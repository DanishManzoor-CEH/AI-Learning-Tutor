import json
import re
from pathlib import Path
from typing import Dict, List


# ============================================================
# EVALUATION DATASET
# ============================================================

def load_evaluation_questions(
    evaluation_file: str = "data/evaluation/questions.json",
) -> List[Dict]:
    """
    Load the research evaluation dataset.
    """

    path = Path(evaluation_file)

    if not path.exists():
        return []

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, list):
            return []

        return data

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return []


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for simple lexical evaluation.
    """

    if not text:
        return ""

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
    Measure how many expected keywords appear
    in the generated answer.

    This is a lexical heuristic, not a semantic
    correctness metric.
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

        if (
            normalized_keyword
            and normalized_keyword in normalized_answer
        ):
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
    Calculate lexical token-level F1.

    This metric measures overlap between the generated
    answer and the reference answer.
    """

    prediction_tokens = set(
        normalize_text(prediction).split()
    )

    reference_tokens = set(
        normalize_text(reference).split()
    )

    if (
        not prediction_tokens
        or not reference_tokens
    ):
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
        / (precision + recall)
    )


# ============================================================
# GROUNDING SCORE
# ============================================================

def grounding_score(
    answer: str,
    context: str,
) -> float:
    """
    Estimate how much of the generated answer overlaps
    lexically with retrieved context.

    This is a heuristic and does not prove factual grounding.
    """

    answer_tokens = set(
        normalize_text(answer).split()
    )

    context_tokens = set(
        normalize_text(context).split()
    )

    if (
        not answer_tokens
        or not context_tokens
    ):
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
    Determine whether the retrieval system returned
    at least one result.
    """

    return bool(search_results)


# ============================================================
# AVERAGE
# ============================================================

def calculate_average(
    values: List[float],
) -> float:

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# OVERALL EXPERIMENT SUMMARY
# ============================================================

def summarize_results(
    results: List[Dict],
) -> Dict:
    """
    Calculate overall research metrics.
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
        "total_questions": len(results),

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
# CATEGORY-LEVEL ANALYSIS
# ============================================================

def summarize_by_category(
    results: List[Dict],
) -> Dict[str, Dict]:
    """
    Calculate research metrics separately for each
    evaluation category.
    """

    grouped = {}

    for result in results:

        category = result.get(
            "category",
            "Unknown",
        )

        if category not in grouped:
            grouped[category] = []

        grouped[category].append(
            result
        )

    category_summary = {}

    for category, category_results in grouped.items():

        baseline_keyword = calculate_average(
            [
                result[
                    "baseline_keyword_coverage"
                ]
                for result in category_results
            ]
        )

        rag_keyword = calculate_average(
            [
                result[
                    "rag_keyword_coverage"
                ]
                for result in category_results
            ]
        )

        baseline_f1 = calculate_average(
            [
                result[
                    "baseline_token_f1"
                ]
                for result in category_results
            ]
        )

        rag_f1 = calculate_average(
            [
                result[
                    "rag_token_f1"
                ]
                for result in category_results
            ]
        )

        grounding = calculate_average(
            [
                result[
                    "rag_grounding_score"
                ]
                for result in category_results
            ]
        )

        retrieval = calculate_average(
            [
                1.0
                if result[
                    "retrieval_success"
                ]
                else 0.0
                for result in category_results
            ]
        )

        category_summary[category] = {
            "questions": len(
                category_results
            ),
            "baseline_keyword_coverage":
                baseline_keyword,
            "rag_keyword_coverage":
                rag_keyword,
            "keyword_improvement":
                rag_keyword - baseline_keyword,
            "baseline_token_f1":
                baseline_f1,
            "rag_token_f1":
                rag_f1,
            "token_f1_improvement":
                rag_f1 - baseline_f1,
            "rag_grounding_score":
                grounding,
            "retrieval_success_rate":
                retrieval,
        }

    return category_summary


# ============================================================
# IMPROVEMENT CALCULATION
# ============================================================

def calculate_improvement(
    baseline: float,
    rag: float,
) -> float:
    """
    Absolute improvement from baseline to RAG.
    """

    return rag - baseline


def calculate_relative_improvement(
    baseline: float,
    rag: float,
) -> float:
    """
    Relative improvement percentage.

    Returns 0 when baseline is zero.
    """

    if baseline == 0:
        return 0.0

    return (
        (rag - baseline)
        / baseline
        * 100
    )


# ============================================================
# HUMAN EVALUATION
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

    values = []

    for dimension in (
        HUMAN_EVALUATION_DIMENSIONS
    ):

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
