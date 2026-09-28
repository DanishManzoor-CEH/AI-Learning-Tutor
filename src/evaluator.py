import json
import re
from pathlib import Path
from typing import Dict, List


# ============================================================
# DATASET LOADING
# ============================================================

def load_evaluation_questions(
    evaluation_file: str = "data/evaluation/questions.json",
) -> List[Dict]:
    """
    Load evaluation questions from JSON.
    """

    path = Path(
        evaluation_file
    )

    if not path.exists():
        return []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str,
) -> str:
    """
    Normalize text for lightweight lexical evaluation.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# KEYWORD COVERAGE
# ============================================================

def keyword_coverage(
    answer: str,
    expected_keywords: List[str],
) -> float:
    """
    Calculate the percentage of expected keywords
    found in the generated answer.
    """

    if not expected_keywords:
        return 0.0

    normalized_answer = normalize_text(
        answer
    )

    found = 0

    for keyword in expected_keywords:

        normalized_keyword = normalize_text(
            keyword
        )

        if normalized_keyword in normalized_answer:
            found += 1

    return found / len(
        expected_keywords
    )


# ============================================================
# TOKEN F1
# ============================================================

def token_f1_score(
    prediction: str,
    reference: str,
) -> float:
    """
    Calculate token-level F1.

    This is a lightweight lexical metric and does not
    represent semantic correctness by itself.
    """

    prediction_tokens = set(
        normalize_text(
            prediction
        ).split()
    )

    reference_tokens = set(
        normalize_text(
            reference
        ).split()
    )

    if not prediction_tokens:
        return 0.0

    if not reference_tokens:
        return 0.0

    common = (
        prediction_tokens
        & reference_tokens
    )

    if not common:
        return 0.0

    precision = (
        len(common)
        / len(prediction_tokens)
    )

    recall = (
        len(common)
        / len(reference_tokens)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


# ============================================================
# GROUNDING HEURISTIC
# ============================================================

def grounding_score(
    answer: str,
    context: str,
) -> float:
    """
    Estimate lexical overlap between answer and retrieved
    context.

    This is only a heuristic indicator. It is NOT a complete
    factuality or hallucination metric.
    """

    answer_tokens = set(
        normalize_text(
            answer
        ).split()
    )

    context_tokens = set(
        normalize_text(
            context
        ).split()
    )

    if not answer_tokens:
        return 0.0

    if not context_tokens:
        return 0.0

    overlap = (
        answer_tokens
        & context_tokens
    )

    return (
        len(overlap)
        / len(answer_tokens)
    )


# ============================================================
# RETRIEVAL SUCCESS
# ============================================================

def retrieval_success(
    search_results: List[Dict],
) -> bool:
    """
    Determine whether relevant retrieval results exist.
    """

    return bool(
        search_results
    )


# ============================================================
# AVERAGE
# ============================================================

def calculate_average(
    values: List[float],
) -> float:

    if not values:
        return 0.0

    return sum(values) / len(
        values
    )


# ============================================================
# AUTOMATIC SUMMARY
# ============================================================

def summarize_results(
    results: List[Dict],
) -> Dict:
    """
    Calculate aggregate automatic evaluation statistics.
    """

    if not results:

        return {
            "total_questions": 0,
            "baseline_keyword_coverage": 0.0,
            "rag_keyword_coverage": 0.0,
            "baseline_token_f1": 0.0,
            "rag_token_f1": 0.0,
            "rag_grounding_score": 0.0,
            "retrieval_success_rate": 0.0,
        }

    return {
        "total_questions": len(
            results
        ),

        "baseline_keyword_coverage":
            calculate_average(
                [
                    result[
                        "baseline_keyword_coverage"
                    ]
                    for result in results
                ]
            ),

        "rag_keyword_coverage":
            calculate_average(
                [
                    result[
                        "rag_keyword_coverage"
                    ]
                    for result in results
                ]
            ),

        "baseline_token_f1":
            calculate_average(
                [
                    result[
                        "baseline_token_f1"
                    ]
                    for result in results
                ]
            ),

        "rag_token_f1":
            calculate_average(
                [
                    result[
                        "rag_token_f1"
                    ]
                    for result in results
                ]
            ),

        "rag_grounding_score":
            calculate_average(
                [
                    result[
                        "rag_grounding_score"
                    ]
                    for result in results
                ]
            ),

        "retrieval_success_rate":
            calculate_average(
                [
                    1.0
                    if result[
                        "retrieval_success"
                    ]
                    else 0.0
                    for result in results
                ]
            ),
    }


# ============================================================
# HUMAN EVALUATION SCORING
# ============================================================

HUMAN_EVALUATION_DIMENSIONS = [
    "factual_correctness",
    "groundedness",
    "educational_usefulness",
    "clarity",
    "difficulty_appropriateness",
    "overall_quality",
]


def calculate_human_evaluation_average(
    evaluation: Dict,
) -> float:
    """
    Calculate average human score across the six
    evaluation dimensions.

    Each dimension uses a 1–5 scale.
    """

    values = []

    for dimension in HUMAN_EVALUATION_DIMENSIONS:

        value = evaluation.get(
            dimension
        )

        if value is None:
            continue

        try:
            values.append(
                float(value)
            )

        except (
            ValueError,
            TypeError,
        ):
            continue

    if not values:
        return 0.0

    return sum(values) / len(
        values
    )


def calculate_interpretation(
    average_score: float,
) -> str:
    """
    Provide a neutral descriptive interpretation of
    the average score.
    """

    if average_score == 0:
        return "Not evaluated"

    if average_score < 2:
        return "Low"

    if average_score < 3:
        return "Below moderate"

    if average_score < 4:
        return "Moderate"

    if average_score < 4.5:
        return "High"

    return "Very high"
