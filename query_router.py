from rag import RAG_SYSTEM

from screening import (
    is_candidate_search_query,
    candidate_search
)


# ============================================================
# PROCESS USER QUERY
# ============================================================

def process_query(query):

    if not query or not query.strip():

        return {
            "type": "resume_qa",
            "answer": (
                "I couldn't find sufficient information "
                "in the retrieved resumes."
            )
        }

    query = query.strip()

    # ========================================================
    # CANDIDATE SEARCH / RANKING
    # ========================================================

    if is_candidate_search_query(
        query
    ):

        results = candidate_search(
            query
        )

        return {
            "type": "candidate_search",
            "results": results
        }

    # ========================================================
    # NORMAL RESUME QUESTION
    # ========================================================

    answer = RAG_SYSTEM(
        query
    )

    return {
        "type": "resume_qa",
        "answer": answer
    }