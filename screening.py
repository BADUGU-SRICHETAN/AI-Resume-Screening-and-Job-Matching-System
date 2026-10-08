import os
import re
import csv

from rag import (
    model,
    retriever
)


# ============================================================
# 1. EXTRACT JOB REQUIREMENTS
# ============================================================

def extract_job_requirements(job_description):

    prompt = f"""
You are an AI-powered recruitment assistant.

Analyze the following Job Description.

Your task is to extract the important requirements without assuming
the job belongs to any particular domain.

Job Description:
----------------
{job_description}
----------------

Return the information in this format:

Job Title:
- 

Required Skills:
- 

Preferred Skills:
- 

Experience:
- 

Education:
- 

Responsibilities:
- 

Tools / Technologies:
- 

Certifications:
- 

Other Requirements:
- 

Important Instructions:
- Extract only information present in the Job Description.
- Do not invent requirements.
- If a category is not mentioned, write "Not specified".
- Keep the extracted information concise.
"""

    response = model.invoke(prompt)

    return response.content


# ============================================================
# 2. RETRIEVE CANDIDATE RESUME CHUNKS
# ============================================================

def retrieve_candidates(job_description):

    job_retrieved_docs = retriever.invoke(
        job_description
    )

    print(
        "Retrieved chunks:",
        len(job_retrieved_docs)
    )

    return job_retrieved_docs


# ============================================================
# 3. GROUP RESUME CHUNKS BY CANDIDATE
# ============================================================

def group_candidates(job_retrieved_docs):

    candidate_documents = {}

    for doc in job_retrieved_docs:

        source = doc.metadata.get(
            "source",
            ""
        )

        if source not in candidate_documents:

            candidate_documents[source] = []

        candidate_documents[source].append(
            doc
        )

    return candidate_documents


# ============================================================
# 4. EVALUATE EACH CANDIDATE
# ============================================================

def evaluate_candidate(
    candidate_name,
    candidate_context,
    requirements
):

    prompt = f"""
You are an AI Resume Screening Assistant.

Evaluate the candidate against the Job Description requirements.

IMPORTANT:
- Use ONLY the candidate evidence provided below.
- Do not assume or invent information.
- Do not give credit for a skill that is not supported by the resume.
- Compare the candidate with the requirements carefully.

Job Requirements:
-----------------
{requirements}
-----------------

Candidate:
----------
{candidate_name}
----------

Candidate Resume Evidence:
--------------------------
{candidate_context}
--------------------------

Return the result in exactly this format:

Match Score: <0-100>%

Matched Requirements:
- 

Missing Requirements:
- 

Candidate Strengths:
- 

Candidate Gaps:
- 

Reason:
- Give a short explanation of why the candidate received this score.
"""

    response = model.invoke(
        prompt
    )

    return response.content


# ============================================================
# 5. SCREEN ALL CANDIDATES
# ============================================================

def screen_candidates(job_description):

    requirements = extract_job_requirements(
        job_description
    )

    print("\nJOB REQUIREMENTS")
    print("=" * 80)

    print(requirements)

    # Retrieve relevant resume chunks
    job_retrieved_docs = retrieve_candidates(
        job_description
    )

    # Group chunks by candidate
    candidate_documents = group_candidates(
        job_retrieved_docs
    )

    print(
        "\nCandidates found:",
        len(candidate_documents)
    )

    candidate_results = {}

    # Evaluate each candidate
    for source, documents in candidate_documents.items():

        candidate_name = os.path.basename(
            source
        )

        candidate_context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        result = evaluate_candidate(
            candidate_name,
            candidate_context,
            requirements
        )

        candidate_results[candidate_name] = {

            "source": source,

            "evaluation": result

        }

        print(
            "\n" + "=" * 80
        )

        print(
            "CANDIDATE:",
            candidate_name
        )

        print(
            "=" * 80
        )

        print(result)

    return (
        requirements,
        candidate_results
    )


# ============================================================
# 6. EXTRACT MATCH SCORE
# ============================================================

def extract_score(evaluation):

    match = re.search(
        r"Match Score:\s*(\d+(?:\.\d+)?)%",
        evaluation,
        re.IGNORECASE
    )

    if match:

        return float(
            match.group(1)
        )

    return 0


# ============================================================
# 7. RANK CANDIDATES
# ============================================================

def rank_candidates(candidate_results):

    ranked_candidates = []

    for candidate_name, data in candidate_results.items():

        evaluation = data["evaluation"]

        score = extract_score(
            evaluation
        )

        ranked_candidates.append({

            "candidate": candidate_name,

            "score": score,

            "source": data["source"],

            "evaluation": evaluation

        })

    # Highest score first
    ranked_candidates = sorted(
        ranked_candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked_candidates


# ============================================================
# 8. EXPORT RESULTS TO CSV
# ============================================================

def export_results_to_csv(
    ranked_candidates,
    filename="screening_results.csv"
):

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Rank",
                "Candidate",
                "Match Score",
                "Matched Requirements",
                "Missing Requirements",
                "Candidate Strengths",
                "Candidate Gaps",
                "Reason",
                "Source"
            ]
        )

        # Create CSV header
        writer.writeheader()

        # Write each candidate
        for rank, candidate in enumerate(
            ranked_candidates,
            start=1
        ):

            evaluation = candidate[
                "evaluation"
            ]

            # -----------------------------
            # Extract Matched Requirements
            # -----------------------------

            matched = re.search(
                r"Matched Requirements:\s*(.*?)(?=\n\s*Missing Requirements:)",
                evaluation,
                re.DOTALL | re.IGNORECASE
            )

            # -----------------------------
            # Extract Missing Requirements
            # -----------------------------

            missing = re.search(
                r"Missing Requirements:\s*(.*?)(?=\n\s*Candidate Strengths:)",
                evaluation,
                re.DOTALL | re.IGNORECASE
            )

            # -----------------------------
            # Extract Candidate Strengths
            # -----------------------------

            strengths = re.search(
                r"Candidate Strengths:\s*(.*?)(?=\n\s*Candidate Gaps:)",
                evaluation,
                re.DOTALL | re.IGNORECASE
            )

            # -----------------------------
            # Extract Candidate Gaps
            # -----------------------------

            gaps = re.search(
                r"Candidate Gaps:\s*(.*?)(?=\n\s*Reason:)",
                evaluation,
                re.DOTALL | re.IGNORECASE
            )

            # -----------------------------
            # Extract Reason
            # -----------------------------

            reason = re.search(
                r"Reason:\s*(.*)",
                evaluation,
                re.DOTALL | re.IGNORECASE
            )

            # -----------------------------
            # Write candidate row
            # -----------------------------

            writer.writerow({

                "Rank":
                    rank,

                "Candidate":
                    candidate["candidate"],

                "Match Score":
                    candidate["score"],

                "Matched Requirements":
                    matched.group(1).strip()
                    if matched
                    else "",

                "Missing Requirements":
                    missing.group(1).strip()
                    if missing
                    else "",

                "Candidate Strengths":
                    strengths.group(1).strip()
                    if strengths
                    else "",

                "Candidate Gaps":
                    gaps.group(1).strip()
                    if gaps
                    else "",

                "Reason":
                    reason.group(1).strip()
                    if reason
                    else "",

                "Source":
                    candidate["source"]
            })

    print(
        f"\nCSV exported successfully: {filename}"
    )


# ============================================================
# 9. TEST THE SCREENING SYSTEM
# ============================================================

if __name__ == "__main__":

    job_description = """

    We are looking for a Data Analyst.

    Responsibilities:
    - Analyze business data
    - Create dashboards and reports
    - Identify business trends
    - Work with stakeholders

    Requirements:
    - SQL
    - Excel
    - Power BI
    - Python
    - Data visualization
    - Analytical skills

    Education:
    - Bachelor's degree in Computer Science,
      Statistics, Mathematics, Business,
      or related field.

    """

    # -----------------------------------------
    # Screen candidates
    # -----------------------------------------

    requirements, candidate_results = screen_candidates(
        job_description
    )

    # -----------------------------------------
    # Rank candidates
    # -----------------------------------------

    ranked_candidates = rank_candidates(
        candidate_results
    )

    # -----------------------------------------
    # Display final ranking
    # -----------------------------------------

    print(
        "\n\nFINAL CANDIDATE RANKING"
    )

    print(
        "=" * 80
    )

    for rank, candidate in enumerate(
        ranked_candidates,
        start=1
    ):

        print(
            f"{rank}. "
            f"{candidate['candidate']} "
            f"→ "
            f"{candidate['score']:.0f}%"
        )

    # -----------------------------------------
    # Export to CSV
    # -----------------------------------------

    export_results_to_csv(
        ranked_candidates
    )