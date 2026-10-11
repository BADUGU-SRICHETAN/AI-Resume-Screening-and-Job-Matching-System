import os
import re
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Please check your .env file."
    )


# ============================================================
# 2. LLM
# ============================================================

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0
)


# ============================================================
# 3. CONTACT INFORMATION EXTRACTION
# ============================================================

def extract_email(text):
    """Extract the first valid email address."""

    if not text:
        return None

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    matches = re.findall(pattern, text)

    return matches[0].strip() if matches else None


def extract_phone(text):
    """Extract an Indian mobile number."""

    if not text:
        return None

    patterns = [
        r"\+91[\s-]*[6-9]\d{4}[\s-]?\d{5}",
        r"\b[6-9]\d{4}[\s-]\d{5}\b",
        r"\b[6-9]\d{9}\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)

        if matches:
            return re.sub(
                r"\s+",
                " ",
                matches[0].strip()
            )

    return None


def extract_profile_url(text, platform):
    """Extract a LinkedIn or GitHub URL explicitly present in a resume."""

    if not text:
        return None

    domains = {
        "linkedin": r"(?:www\.)?linkedin\.com",
        "github": r"(?:www\.)?github\.com",
    }

    if platform not in domains:
        return None

    pattern = (
        r"(?i)(?:https?://)?"
        + domains[platform]
        + r"/[^\s<>()\[\]{}\"',;]+"
    )

    matches = re.findall(pattern, text)

    for match in matches:
        url = match.strip().rstrip(".,;:!?)")

        if not url.lower().startswith(("http://", "https://")):
            url = "https://" + url

        return url

    return None


def extract_linkedin(text):
    return extract_profile_url(text, "linkedin")


def extract_github(text):
    return extract_profile_url(text, "github")


# ============================================================
# 4. GET ALL RESUME DOCUMENTS
# ============================================================

def get_all_resume_documents():
    """Retrieve documents and metadata from the existing Chroma store."""

    from rag import vector_store

    data = vector_store.get(
        include=["documents", "metadatas"]
    )

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    resume_documents = []

    for index, document in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        resume_documents.append({
            "text": document or "",
            "metadata": metadata or {}
        })

    return resume_documents


# ============================================================
# 5. GROUP CHUNKS BY CANDIDATE
# ============================================================

def get_all_candidates():
    """
    Group every Chroma document chunk by its source resume.
    Each candidate is evaluated using all available chunks.
    """

    documents = get_all_resume_documents()

    candidates = defaultdict(list)

    for document in documents:
        text = document["text"]
        metadata = document["metadata"]

        source = metadata.get("source", "")

        if not source:
            continue

        candidate_name = os.path.basename(source)

        if text.strip():
            candidates[candidate_name].append(text)

    return dict(candidates)


# ============================================================
# 6. CLEAN CANDIDATE NAME
# ============================================================

def get_display_name(candidate_name):
    """Remove filename extensions and common resume labels."""

    name = os.path.basename(str(candidate_name)).strip()

    name = re.sub(
        r"\.(pdf|docx?)$",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(
        r"\b(resume|cv|profile|curriculum vitae)\b",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(
        r"\s+",
        " ",
        name.replace("_", " ").replace("-", " ")
    )

    return name.strip().title() or "Unknown Candidate"


# ============================================================
# 7. EXTRACT SCORE
# ============================================================

def extract_score(text):
    """Extract and clamp a model-generated score to 0–100."""

    if not text:
        return 0.0

    match = re.search(
        r"SCORE\s*:\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if not match:
        match = re.search(
            r"\b(\d{1,3}(?:\.\d+)?)\s*/\s*100\b",
            text
        )

    if not match:
        return 0.0

    try:
        score = float(match.group(1))
    except (ValueError, TypeError):
        return 0.0

    return max(0.0, min(100.0, score))


# ============================================================
# 8. PARSE MODEL RESPONSE
# ============================================================

def parse_evaluation_response(content):
    """
    Extract the score and explanation.
    Keep the original model response for traceability.
    """

    content = str(content or "").strip()

    score = extract_score(content)

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        content,
        re.IGNORECASE | re.DOTALL
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else content
    )

    return score, reason


# ============================================================
# 9. EVALUATE CANDIDATE FOR SEARCH QUERY
# ============================================================

def evaluate_candidate_for_query(
    candidate_name,
    resume_text,
    query
):
    prompt = f"""
You are a resume-search assistant.

Evaluate the candidate only using the resume text provided.

USER QUERY:
{query}

CANDIDATE:
{candidate_name}

RESUME:
{resume_text}

Instructions:
1. Assess how closely the resume answers the user's query.
2. Return a relevance score from 0 to 100.
3. Base the score on explicit evidence in the resume.
4. Do not invent skills, qualifications, projects or experience.
5. Do not award a high score merely because the resume contains
   common technical keywords.
6. If a required skill is not supported by the resume, do not
   claim that the candidate has that skill.
7. Explain the strongest evidence and any important missing
   information.
8. A score measures relevance, not verified candidate accuracy.

Return exactly this format:

SCORE: <number>

REASON:
<brief evidence-based explanation>
"""

    response = model.invoke(prompt)

    score, reason = parse_evaluation_response(
        response.content
    )

    return {
        "candidate": candidate_name,
        "score": score,
        "reason": reason
    }


# ============================================================
# 10. SEARCH CANDIDATES
# ============================================================

def search_candidates(query, candidates):
    """Evaluate and rank candidates for a natural-language query."""

    results = []

    for candidate_name, chunks in candidates.items():
        resume_text = "\n\n".join(chunks)

        result = evaluate_candidate_for_query(
            candidate_name,
            resume_text,
            query
        )

        results.append(result)

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results


# ============================================================
# 11. DETECT CANDIDATE SEARCH QUERY
# ============================================================

def is_candidate_search_query(query):
    query = query.lower()

    search_keywords = [
        "top candidates",
        "top candidate",
        "top applicants",
        "top applicant",
        "best candidates",
        "best candidate",
        "which candidates",
        "who has",
        "who knows",
        "find candidates",
        "find candidate",
        "candidates with",
        "candidate with",
        "profiles of",
        "top profiles",
        "applicants with",
        "applicant with",
    ]

    return any(
        keyword in query
        for keyword in search_keywords
    )


# ============================================================
# 12. ADD CONTACT DETAILS
# ============================================================

def add_contact_details(result, candidate_chunks):
    """Extract contact information from all available resume chunks."""

    complete_resume = "\n\n".join(candidate_chunks)

    result["email"] = extract_email(complete_resume)
    result["phone"] = extract_phone(complete_resume)
    result["linkedin"] = extract_linkedin(complete_resume)
    result["github"] = extract_github(complete_resume)

    return result


# ============================================================
# 13. CANDIDATE SEARCH
# ============================================================

def candidate_search(query):
    """Search resumes and attach contact information."""

    candidates = get_all_candidates()

    if not candidates:
        return []

    results = search_candidates(
        query,
        candidates
    )

    # Exclude candidates whose evaluation returned no usable score.
    results = [
        result
        for result in results
        if result.get("score", 0) > 0
    ]

    for result in results:
        candidate_name = result["candidate"]

        candidate_chunks = candidates.get(
            candidate_name,
            []
        )

        add_contact_details(
            result,
            candidate_chunks
        )

    return results


# ============================================================
# 14. EXTRACT JOB REQUIREMENTS
# ============================================================

def extract_job_requirements(job_description):
    """Extract requirements from the supplied job description."""

    prompt = f"""
Extract requirements from the following job description.

JOB DESCRIPTION:
{job_description}

Return these categories:

Skills:
Experience:
Tools/Technologies:
Education:
Other Requirements:

Rules:
- Use only requirements stated in the job description.
- Do not add requirements that are not mentioned.
- Distinguish required qualifications from preferred qualifications.
"""

    response = model.invoke(prompt)

    return response.content


# ============================================================
# 15. EVALUATE CANDIDATE FOR JOB DESCRIPTION
# ============================================================

def evaluate_candidate(
    candidate_name,
    resume_text,
    job_description
):
    """
    Compare a complete candidate resume against a job description.

    Score bands are used for interpretation only; they do not
    automatically prove that a candidate is qualified.
    """

    prompt = f"""
You are an evidence-based resume screening assistant.

Compare the resume with the job description.

JOB DESCRIPTION:
{job_description}

CANDIDATE:
{candidate_name}

RESUME:
{resume_text}

Evaluate only information supported by the resume.

Scoring guidance:
- 90–100: Strong evidence for nearly all important requirements.
- 75–89: Good evidence for many important requirements, with
  some gaps.
- 50–74: Partial match; several requirements are missing or unclear.
- 25–49: Limited match to the stated requirements.
- 0–24: Very little relevant evidence.

These score bands are guidance, not a guarantee of qualification.

Important rules:
1. Evaluate required skills, tools, experience, education and
   other job requirements.
2. Give credit only for evidence in the resume.
3. Do not assume a skill just because a related skill is listed.
4. Do not invent work experience, projects, qualifications or
   achievements.
5. Distinguish missing information from confirmed lack of a skill.
6. Consider required qualifications more important than preferred
   qualifications.
7. Explain which important requirements are supported and which
   are missing or unclear.
8. Do not give every candidate a high score.
9. Do not use candidate names, personal details, gender, age,
   religion, caste or other protected characteristics to score.
10. A match score is an estimate, not a verified probability.

Return exactly:

SCORE: <number from 0 to 100>

REASON:
<concise evidence-based explanation>

MATCHED REQUIREMENTS:
- <requirement supported by resume>

MISSING OR UNCLEAR REQUIREMENTS:
- <requirement not supported or not clear in resume>
"""

    response = model.invoke(prompt)

    score, reason = parse_evaluation_response(
        response.content
    )

    return {
        "candidate": candidate_name,
        "score": score,
        "reason": reason
    }


# ============================================================
# 16. SCREEN ALL CANDIDATES
# ============================================================

def screen_candidates(job_description):
    """
    Evaluate every candidate against the job description.

    Returns:
        requirements: extracted job requirements
        results: ranked-ready candidate results
    """

    requirements = extract_job_requirements(
        job_description
    )

    all_candidates = get_all_candidates()

    results = []

    for candidate_name, chunks in all_candidates.items():
        resume_text = "\n\n".join(chunks)

        result = evaluate_candidate(
            candidate_name,
            resume_text,
            job_description
        )

        add_contact_details(
            result,
            chunks
        )

        results.append(result)

    return requirements, results


# ============================================================
# 17. RANK CANDIDATES
# ============================================================

def rank_candidates(candidates):
    """Sort candidates by match score from highest to lowest."""

    return sorted(
        candidates,
        key=lambda item: float(
            item.get("score", 0) or 0
        ),
        reverse=True
    )


# ============================================================
# 18. COUNT MATCHES USING SCORE BANDS
# ============================================================

def calculate_match_counts(candidates):
    """
    Count candidates by score band.

    These are score-based categories, not verified accuracy labels.
    """

    strong_matches = sum(
        1 for candidate in candidates
        if float(candidate.get("score", 0) or 0) >= 80
    )

    potential_matches = sum(
        1 for candidate in candidates
        if 60 <= float(candidate.get("score", 0) or 0) < 80
    )

    low_matches = sum(
        1 for candidate in candidates
        if float(candidate.get("score", 0) or 0) < 60
    )

    return {
        "total_candidates": len(candidates),
        "strong_matches": strong_matches,
        "potential_matches": potential_matches,
        "low_matches": low_matches,
    }
