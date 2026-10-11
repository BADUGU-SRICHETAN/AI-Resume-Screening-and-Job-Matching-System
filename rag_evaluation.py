import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag_backup import RAG_SYSTEM_WITH_CONTEXT
from evaluation import load_evaluation_dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing from the project .env file."
    )

RESULTS_DIR = PROJECT_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt-4o-mini"

FALLBACK_MESSAGE = (
    "I couldn't find sufficient information "
    "in the retrieved resumes."
)

# Timeout prevents individual requests from waiting indefinitely.
judge = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    temperature=0,
    timeout=60,
    max_retries=1,
)

# One timestamped report and one continually updated report.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

REPORT_PATH = RESULTS_DIR / f"rag_evaluation_{RUN_TIMESTAMP}.json"
LATEST_REPORT_PATH = RESULTS_DIR / "latest_rag_evaluation.json"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def safe_text(value):
    """Convert common values into readable text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "\n\n".join(
            safe_text(item)
            for item in value
            if item is not None
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return str(value)


def normalize_contexts(contexts):
    """Normalize retrieved context into a list of strings."""

    if contexts is None:
        return []

    if isinstance(contexts, str):
        return [contexts] if contexts.strip() else []

    if not isinstance(contexts, list):
        return []

    normalized = []

    for context in contexts:
        text = safe_text(context).strip()

        if text:
            normalized.append(text)

    return normalized


def parse_json_response(response_text):
    """Parse JSON returned by the evaluation model."""

    response_text = response_text.strip()

    if response_text.startswith("```"):
        lines = response_text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response_text = "\n".join(lines).strip()

        if response_text.lower().startswith("json"):
            response_text = response_text[4:].strip()

    try:
        result = json.loads(response_text)

    except json.JSONDecodeError as error:
        raise ValueError(
            "The evaluation model returned invalid JSON: "
            f"{response_text[:500]}"
        ) from error

    if not isinstance(result, dict):
        raise ValueError(
            "The evaluation response must be a JSON object."
        )

    return result


def validate_score(value, metric_name):
    """Validate an integer score between 1 and 5."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 5
    ):
        raise ValueError(
            f"Invalid {metric_name} score: {value}. "
            "Expected an integer between 1 and 5."
        )

    return value


def save_report(results, total_cases):
    """Save results after every test, including partial runs."""

    successful = [
        item
        for item in results
        if item["evaluation_status"] == "success"
    ]

    def average(metric):
        values = [
            item[metric]
            for item in successful
            if item.get(metric) is not None
        ]

        return (
            round(sum(values) / len(values), 2)
            if values
            else None
        )

    correctness_count = sum(
        1
        for item in successful
        if item.get("correctness") is not None
    )

    report = {
        "project": "Resumind AI",
        "evaluation_type": "Custom LLM-as-a-judge RAG evaluation",
        "model": MODEL_NAME,
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "total_test_cases": total_cases,
        "completed_test_cases": len(results),
        "successful_evaluations": len(successful),
        "failed_evaluations": len(results) - len(successful),
        "correctness_evaluations": correctness_count,
        "average_scores": {
            "faithfulness": average("faithfulness"),
            "answer_relevancy": average("answer_relevancy"),
            "correctness": average("correctness"),
        },
        "results": results,
    }

    # Save the timestamped report.
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Update the latest report after every test case.
    with LATEST_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# 3. EVALUATE ONE ANSWER
# ============================================================

def evaluate_rag_answer(
    question,
    answer,
    contexts,
    reference_answer="",
):
    """
    Evaluate an answer using an LLM judge.

    Metrics:
        Faithfulness: 1-5
        Answer relevancy: 1-5
        Reference-based correctness: 1-5 or None
    """

    context_list = normalize_contexts(contexts)
    context_text = "\n\n".join(context_list)

    if not context_text.strip():
        context_text = (
            "No relevant resume context was retrieved."
        )

    if not isinstance(answer, str) or not answer.strip():
        answer = FALLBACK_MESSAGE

    has_reference = (
        isinstance(reference_answer, str)
        and bool(reference_answer.strip())
    )

    reference_text = (
        reference_answer.strip()
        if has_reference
        else "No verified reference answer was provided."
    )

    prompt = f"""
You are a careful evaluator of a resume question-answering RAG system.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_text[:12000]}

GENERATED ANSWER:
{answer[:5000]}

REFERENCE ANSWER:
{reference_text[:3000]}

Evaluate these metrics on a scale of 1 to 5.

FAITHFULNESS:
Does the answer contain factual claims supported by the retrieved context?

1 = Most claims are unsupported.
2 = Many claims are unsupported.
3 = Some claims are supported, but important problems remain.
4 = Almost all claims are supported.
5 = All factual claims are supported.

ANSWER RELEVANCY:
Does the answer address the question?

1 = Completely irrelevant.
2 = Mostly irrelevant.
3 = Partially relevant.
4 = Relevant with minor omissions.
5 = Direct and complete.

CORRECTNESS:
Compare the answer against the reference answer.

1 = Incorrect or substantially contradictory.
2 = Many important facts are missing or incorrect.
3 = Partially correct.
4 = Mostly correct with minor omissions.
5 = Correct and complete relative to the reference.

RULES:
- Use the retrieved context to judge faithfulness.
- Use the reference answer to judge correctness.
- If there is no reference answer, correctness must be null.
- Different wording is acceptable when the meaning is correct.
- Do not reward unsupported claims or invented candidate names.
- Do not assume that every candidate with Python also has SQL.
- If context is missing, do not assume the answer is factually supported.
- An appropriate insufficient-information response can be faithful.
- Return integer scores between 1 and 5, except correctness may be null.
- Return only valid JSON, without Markdown.

Required JSON:
{{
    "faithfulness": 1,
    "answer_relevancy": 1,
    "correctness": null,
    "explanation": "Brief explanation"
}}
"""

    try:
        response = judge.invoke(prompt)

    except Exception as error:
        raise RuntimeError(
            f"OpenAI evaluation request failed: {error}"
        ) from error

    scores = parse_json_response(
        safe_text(response.content)
    )

    faithfulness = validate_score(
        scores.get("faithfulness"),
        "faithfulness",
    )

    answer_relevancy = validate_score(
        scores.get("answer_relevancy"),
        "answer_relevancy",
    )

    correctness = scores.get("correctness")

    if not has_reference:
        correctness = None

    else:
        correctness = validate_score(
            correctness,
            "correctness",
        )

    explanation = safe_text(
        scores.get("explanation", "")
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "correctness": correctness,
        "explanation": explanation,
    }


# ============================================================
# 4. LOAD THE EVALUATION DATASET
# ============================================================

def prepare_dataset():
    """Load and validate evaluation questions."""

    dataset = load_evaluation_dataset()

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(
            "No evaluation questions found. "
            "Check evaluation_dataset.json."
        )

    valid_records = []

    for index, item in enumerate(dataset, start=1):

        if not isinstance(item, dict):
            print(f"Skipping invalid test case {index}.")
            continue

        question = item.get("question", "")

        if not isinstance(question, str) or not question.strip():
            print(f"Skipping test case {index}: empty question.")
            continue

        reference = item.get("ground_truth", "")

        if reference is None:
            reference = ""

        if not isinstance(reference, str):
            reference = safe_text(reference)

        valid_records.append({
            "question": question.strip(),
            "ground_truth": reference.strip(),
        })

    if not valid_records:
        raise ValueError(
            "No valid evaluation questions were found."
        )

    return valid_records


# ============================================================
# 5. RUN THE EVALUATION
# ============================================================

def run_rag_evaluation():

    dataset = prepare_dataset()
    total_cases = len(dataset)
    results = []

    print("\n" + "=" * 70)
    print("             RESUMIND AI - RAG EVALUATION")
    print("=" * 70)

    print("Model:", MODEL_NAME)
    print("Total test cases:", total_cases)
    print("Reports will be saved to:", RESULTS_DIR)

    try:
        for index, item in enumerate(dataset, start=1):

            question = item["question"]
            reference_answer = item["ground_truth"]

            print("\n" + "-" * 70)
            print(f"Test case {index}/{total_cases}")
            print("Question:", question)

            try:
                # Run the actual RAG pipeline.
                rag_output = RAG_SYSTEM_WITH_CONTEXT(question)

                if not isinstance(rag_output, dict):
                    raise TypeError(
                        "RAG_SYSTEM_WITH_CONTEXT must return a dictionary."
                    )

                answer = safe_text(
                    rag_output.get("answer", "")
                ).strip()

                if not answer:
                    answer = FALLBACK_MESSAGE

                contexts = normalize_contexts(
                    rag_output.get("contexts", [])
                )

                sources = rag_output.get("sources", [])

                if not isinstance(sources, list):
                    sources = [safe_text(sources)]

                print("Retrieved chunks:", len(contexts))
                print("Sources:", sources)

                print("\nEvaluating answer...")

                scores = evaluate_rag_answer(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    reference_answer=reference_answer,
                )

                result = {
                    "question": question,
                    "reference_answer": reference_answer or None,
                    "generated_answer": answer,
                    "retrieved_chunk_count": len(contexts),
                    "contexts": contexts,
                    "sources": sources,
                    "faithfulness": scores["faithfulness"],
                    "answer_relevancy": scores["answer_relevancy"],
                    "correctness": scores["correctness"],
                    "explanation": scores["explanation"],
                    "evaluation_status": "success",
                }

                results.append(result)

                print("\nGenerated answer:")
                print(answer)

                print(
                    "\nFaithfulness:",
                    scores["faithfulness"],
                    "/ 5",
                )

                print(
                    "Answer relevancy:",
                    scores["answer_relevancy"],
                    "/ 5",
                )

                print(
                    "Correctness:",
                    (
                        f"{scores['correctness']} / 5"
                        if scores["correctness"] is not None
                        else "N/A - no reference answer"
                    ),
                )

                print("Explanation:", scores["explanation"])

            except Exception as error:

                error_message = (
                    f"{type(error).__name__}: {error}"
                )

                print("\nTest case failed:", error_message)

                results.append({
                    "question": question,
                    "reference_answer": reference_answer or None,
                    "generated_answer": None,
                    "retrieved_chunk_count": 0,
                    "contexts": [],
                    "sources": [],
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "correctness": None,
                    "explanation": error_message,
                    "evaluation_status": "failed",
                })

            # Save progress even when a test case fails.
            save_report(results, total_cases)

            print("\nProgress saved.")

            # Short pause between test cases.
            if index < total_cases:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

    finally:
        # Always save whatever results are available.
        save_report(results, total_cases)

        successful = [
            result
            for result in results
            if result["evaluation_status"] == "success"
        ]

        def average(metric):
            values = [
                result[metric]
                for result in successful
                if result[metric] is not None
            ]

            return (
                round(sum(values) / len(values), 2)
                if values
                else None
            )

        print("\n" + "=" * 70)
        print("                 EVALUATION SUMMARY")
        print("=" * 70)

        print("Total test cases:", total_cases)
        print("Completed test cases:", len(results))
        print("Successful evaluations:", len(successful))
        print("Failed evaluations:", len(results) - len(successful))

        print(
            "Average faithfulness:",
            average("faithfulness"),
            "/ 5",
        )

        print(
            "Average answer relevancy:",
            average("answer_relevancy"),
            "/ 5",
        )

        print(
            "Average correctness:",
            average("correctness"),
            "/ 5",
        )

        print("\nTimestamped report:")
        print(REPORT_PATH)

        print("\nLatest report:")
        print(LATEST_REPORT_PATH)

        print("\nEvaluation finished.")


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_rag_evaluation()