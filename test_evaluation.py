from evaluation import (
    evaluate_llm_answer,
    load_evaluation_dataset,
    calculate_average_scores
)


# Load the test dataset
dataset = load_evaluation_dataset()

print("Number of test cases:", len(dataset))

# Test one sample first
sample = dataset[0]

generated_answer = """
Python, SQL, and machine learning are listed as technical skills.
"""

result = evaluate_llm_answer(
    question=sample["question"],
    answer=generated_answer,
    reference_answer=sample["ground_truth"]
)

print("\nEvaluation result:")
print(result)

# Demonstrate average calculation
average_scores = calculate_average_scores([result])

print("\nAverage scores:")
print(average_scores)