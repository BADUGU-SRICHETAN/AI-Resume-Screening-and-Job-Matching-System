import json
import csv
from pathlib import Path
from datetime import datetime

from query_router import process_query
from evaluation import (
    evaluate_llm_answer,
    load_evaluation_dataset,
    calculate_average_scores
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).parent

RESULTS_DIR = PROJECT_DIR / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

    dataset = load_evaluation_dataset()
    evaluation_results = []

    print("\n" + "=" * 60)
    print("       RESUMIND AI - LLM EVALUATION")
    print("=" * 60)

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        reference_answer = item["ground_truth"]

        print(f"\nTest case {index}/{len(dataset)}")
        print("Question:", question)

        try:
            # Run the question through your actual application
            response = process_query(question)

            # Evaluate normal RAG questions only
            if response.get("type") != "resume_qa":

                print(
                    "Skipped: This question was routed "
                    "to candidate search."
                )
                continue

            answer = response.get("answer", "")

            # Evaluate the generated answer
            scores = evaluate_llm_answer(
                question=question,
                answer=answer,
                reference_answer=reference_answer
            )

            result = {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": answer,
                **scores
            }

            evaluation_results.append(result)

            print("Generated answer:", answer)
            print("Correctness:", scores["correctness"], "/ 5")
            print("Relevance:", scores["relevance"], "/ 5")
            print(
                "Instruction following:",
                scores["instruction_following"],
                "/ 5"
            )

        except Exception as error:

            print("Evaluation failed:", str(error))

            evaluation_results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": "",
                "correctness": None,
                "relevance": None,
                "instruction_following": None,
                "explanation": str(error)
            })

    # ========================================================
    # CALCULATE AVERAGES
    # ========================================================

    successful_results = [
        result
        for result in evaluation_results
        if isinstance(result.get("correctness"), int)
        and not isinstance(result.get("correctness"), bool)
    ]

    averages = calculate_average_scores(
        successful_results
    )

    print("\n" + "=" * 60)
    print("AVERAGE EVALUATION SCORES")
    print("=" * 60)

    for metric, score in averages.items():
        print(f"{metric}: {score}/5")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = RESULTS_DIR / (
        f"llm_evaluation_{timestamp}.json"
    )

    csv_path = RESULTS_DIR / (
        f"llm_evaluation_{timestamp}.csv"
    )

    report = {
        "evaluated_at": datetime.now().isoformat(),
        "total_test_cases": len(dataset),
        "successful_evaluations": len(successful_results),
        "average_scores": averages,
        "results": evaluation_results
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    fieldnames = [
        "question",
        "reference_answer",
        "generated_answer",
        "correctness",
        "relevance",
        "instruction_following",
        "explanation"
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(evaluation_results)

    print("\nEvaluation completed.")
    print("JSON report:", json_path)
    print("CSV report:", csv_path)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_evaluation()