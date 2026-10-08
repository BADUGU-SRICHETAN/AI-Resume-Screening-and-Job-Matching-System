from dotenv import load_dotenv
import os
import re

# ============================================================
# 1. IMPORTS
# ============================================================

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# ============================================================
# 2. OPENAI API KEY
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# 3. LLM
# ============================================================

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)


# ============================================================
# 4. LOAD RESUMES
# ============================================================

file_path = r"C:\Users\badug\OneDrive\Documents\Desktop\Resume Screening RAG Project\Resumes"

loader = DirectoryLoader(
    path=file_path
)

docs = loader.load()

print("Documents loaded:", len(docs))


# ============================================================
# 5. CHUNKING
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=600,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Chunks created:", len(chunks))


# ============================================================
# 6. EMBEDDINGS
# ============================================================

embedding = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY
)


# ============================================================
# 7. LOAD EXISTING CHROMA DATABASE
# ============================================================

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="ResumeRAG"
)

print(
    "Chroma documents:",
    len(vector_store.get()["ids"])
)


# ============================================================
# 8. GLOBAL RETRIEVERS
# ============================================================

retriever1 = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 10,
        "lambda_mult": 0.9
    }
)


retriever2 = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


retriever3 = BM25Retriever.from_documents(
    chunks
)


retriever = EnsembleRetriever(
    retrievers=[
        retriever1,
        retriever2,
        retriever3
    ],
    weights=[
        0.45,
        0.25,
        0.30
    ]
)


# ============================================================
# 9. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    return re.sub(
        r"[^a-z0-9]",
        "",
        text.lower()
    )


# ============================================================
# 10. EXTRACT CANDIDATE NAME
# ============================================================

def extract_candidate_name(query):

    patterns = [
        r"\bof\s+(.+?)(?:\?|$)",
        r"\bfor\s+(.+?)(?:\?|$)",
        r"\babout\s+(.+?)(?:\?|$)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query,
            re.IGNORECASE
        )

        if match:

            return match.group(1).strip()

    return None


# ============================================================
# 11. RAG QUESTION ANSWERING
# ============================================================

def RAG_SYSTEM(query):

    # --------------------------------------------------------
    # Find candidate name
    # --------------------------------------------------------

    candidate_name = extract_candidate_name(query)

    if not candidate_name:

        return (
            "I couldn't find sufficient information "
            "in the retrieved resumes."
        )


    print("\nCandidate:", candidate_name)


    # --------------------------------------------------------
    # Normalize candidate name
    # --------------------------------------------------------

    candidate_name_normalized = normalize_text(
        candidate_name
    )


    # --------------------------------------------------------
    # Find candidate resume
    # --------------------------------------------------------

    candidate_sources = set()

    for doc in chunks:

        source = doc.metadata.get(
            "source",
            ""
        )

        content = doc.page_content


        if candidate_name_normalized in normalize_text(content):

            candidate_sources.add(source)


        elif candidate_name_normalized in normalize_text(source):

            candidate_sources.add(source)


    # --------------------------------------------------------
    # If candidate not found
    # --------------------------------------------------------

    if not candidate_sources:

        return (
            "I couldn't find sufficient information "
            "in the retrieved resumes."
        )


    print("\nResume found:")

    for source in candidate_sources:

        print(source)


    # --------------------------------------------------------
    # Get only candidate chunks
    # --------------------------------------------------------

    candidate_chunks = []

    for doc in chunks:

        source = doc.metadata.get(
            "source",
            ""
        )

        if source in candidate_sources:

            candidate_chunks.append(doc)


    print(
        "\nCandidate chunks:",
        len(candidate_chunks)
    )


    # --------------------------------------------------------
    # Candidate BM25 retriever
    # --------------------------------------------------------

    candidate_bm25 = BM25Retriever.from_documents(
        candidate_chunks
    )

    candidate_bm25.k = 3


    # --------------------------------------------------------
    # Candidate Chroma retriever
    # --------------------------------------------------------

    source = list(candidate_sources)[0]

    candidate_chroma = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 8,
            "lambda_mult": 0.9,
            "filter": {
                "source": source
            }
        }
    )


    # --------------------------------------------------------
    # Hybrid candidate retriever
    # --------------------------------------------------------

    candidate_retriever = EnsembleRetriever(

        retrievers=[
            candidate_chroma,
            candidate_bm25
        ],

        weights=[
            0.60,
            0.40
        ]
    )


    # --------------------------------------------------------
    # Retrieve documents using actual query
    # --------------------------------------------------------

    retrieved_docs = candidate_retriever.invoke(
        query
    )


    print(
        "\nRetrieved chunks:",
        len(retrieved_docs)
    )


    # --------------------------------------------------------
    # Create context
    # --------------------------------------------------------

    context_text = "\n\n".join(

        f"""
Source: {doc.metadata.get("source")}

{doc.page_content}
"""

        for doc in retrieved_docs
    )


    # --------------------------------------------------------
    # Grounded RAG prompt
    # --------------------------------------------------------

    prompt = f"""
You are a professional AI Resume Retrieval Assistant.

You are provided with resume excerpts retrieved from a
candidate's resume.

These retrieved resume excerpts are your ONLY source
of information.

Instructions:

- Answer the user's question using ONLY the retrieved
  resume information.
- Do not use your own knowledge.
- Do not guess.
- Do not infer information that is not explicitly present.
- Do not invent any information.
- Carefully read all retrieved excerpts before answering.
- If the requested information is present, provide it.
- Keep the answer concise and factual.

If the requested information is not present in the
retrieved resume information, respond exactly:

"I couldn't find sufficient information in the retrieved resumes."

Retrieved Resume Information
============================

{context_text}

============================

Question:
{query}

Answer:
"""


    # --------------------------------------------------------
    # LLM response
    # --------------------------------------------------------

    response = model.invoke(
        prompt
    )


    return response.content


# ============================================================
# 12. TEST
# ============================================================

if __name__ == "__main__":

    query = "What are the skills of Srichetan Badugu?"

    answer = RAG_SYSTEM(query)

    print("\nAnswer:")
    print(answer)