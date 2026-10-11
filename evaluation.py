import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize the evaluation model
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is missing. Check your project's .env file."
    )

evaluation_model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0
)


def evaluate_llm_answer(question, answer, reference_answer):
    """
    Evaluate an LLM-generated answer using an LLM-as-judge.

    Returns:
        correctness: estimated score from 1 to 5
        relevance: estimated score from 1 to 5
        instruction_following: estimated score from 1 to 5
        explanation: judge's explanation
    """

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an evaluator for a resume-screening RAG application.

            Evaluate the generated answer against the reference answer.

            Give each metric an integer score from 1 to 5.

            1. Correctness:
               Is the answer factually consistent with the reference?

            2. Relevance:
               Does the answer address the question?

            3. Instruction following:
               Does the answer follow the question's requirements?

            Important:
            - Do not reward unsupported claims.
            - Do not assume information that is absent.
            - Treat the reference answer as the evaluation target.
            - Explain limitations when the reference answer is vague.

            Return ONLY valid JSON in this format:
            {{
                "correctness": 1,
                "relevance": 1,
                "instruction_following": 1,
                "explanation": "Brief explanation"
            }}
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Reference answer:
            {reference_answer}

            Generated answer:
            {answer}
            """
        )
    ])

    response = (prompt | evaluation_model).invoke({
        "question": question,
        "reference_answer": reference_answer,
        "answer": answer
    })

    try:
        result = json.loads(response.content)

        for metric in [
            "correctness",
            "relevance",
            "instruction_following"
        ]:
            score = result.get(metric)

            if (
                not isinstance(score, int)
                or isinstance(score, bool)
                or not 1 <= score <= 5
            ):
                raise ValueError(
                    f"Invalid evaluation score for {metric}"
                )

        return result

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            "The evaluation model returned invalid evaluation JSON. "
            f"Details: {exc}"
        ) from exc


def load_evaluation_dataset():
    """Load test cases from evaluation_dataset.json."""

    dataset_path = Path(__file__).parent / "evaluation_dataset.json"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("The evaluation dataset must be a JSON list.")

    for index, item in enumerate(dataset):
        if not all(
            item.get(field)
            for field in ["question", "ground_truth"]
        ):
            raise ValueError(
                f"Test case {index + 1} must contain "
                "'question' and 'ground_truth'."
            )

    return dataset


def calculate_average_scores(results):
    """Calculate average LLM-judge scores across test cases."""

    metrics = [
        "correctness",
        "relevance",
        "instruction_following"
    ]

    if not results:
        return {metric: 0.0 for metric in metrics}

    return {
        metric: round(
            sum(result[metric] for result in results) / len(results),
            2
        )
        for metric in metrics
    }