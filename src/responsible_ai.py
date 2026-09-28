import re
from typing import Dict, List


# ============================================================
# RESPONSIBLE AI CONFIGURATION
# ============================================================

UNSAFE_PATTERNS = [
    r"\bhow\s+to\s+deploy\s+(a\s+)?malware\b",
    r"\bhow\s+to\s+create\s+(a\s+)?malware\b",
    r"\bcreate\s+(a\s+)?ransomware\b",
    r"\bdeploy\s+ransomware\b",
    r"\bsteal\s+passwords\b",
    r"\bsteal\s+credentials\b",
    r"\bcredential\s+stealing\b",
    r"\bphishing\s+kit\b",
    r"\bcreate\s+a\s+phishing\s+page\b",
    r"\bwrite\s+malware\b",
    r"\bbuild\s+malware\b",
]


UNCERTAINTY_PHRASES = [
    "i don't know",
    "i do not know",
    "not enough information",
    "insufficient information",
    "not available",
    "couldn't find",
    "could not find",
    "cannot determine",
    "unable to determine",
    "not supported by",
    "knowledge base does not",
]


SOURCE_PHRASES = [
    "source",
    "sources",
    "according to",
    "page",
    "document",
]


def detect_unsafe_request(
    question: str,
) -> bool:
    """
    Detect potentially unsafe cybersecurity requests.

    This is a lightweight rule-based safety layer.
    It is not a complete cybersecurity safety classifier.
    """

    normalized_question = (
        question.lower().strip()
    )

    for pattern in UNSAFE_PATTERNS:

        if re.search(
            pattern,
            normalized_question,
        ):
            return True

    return False


def build_safe_response(
    question: str,
) -> str:
    """
    Return a safe response for potentially harmful
    cybersecurity requests.
    """

    return """
I can help with cybersecurity concepts, defensive security,
threat detection, secure configuration, vulnerability
management, and ethical security testing.

I can't provide instructions for deploying malware, stealing
credentials, creating phishing infrastructure, or carrying
out harmful attacks.

For a safe alternative, I can explain the attack concept,
how defenders detect it, and how to protect systems against it.
""".strip()


def contains_uncertainty_language(
    answer: str,
) -> bool:
    """
    Check whether an answer contains language indicating
    uncertainty or knowledge limitations.
    """

    normalized_answer = (
        answer.lower().strip()
    )

    return any(
        phrase in normalized_answer
        for phrase in UNCERTAINTY_PHRASES
    )


def contains_source_language(
    answer: str,
) -> bool:
    """
    Check whether an answer contains source-related language.
    """

    normalized_answer = (
        answer.lower().strip()
    )

    return any(
        phrase in normalized_answer
        for phrase in SOURCE_PHRASES
    )


def evaluate_responsible_ai(
    question: str,
    answer: str,
    has_retrieved_context: bool,
) -> Dict:
    """
    Run lightweight responsible-AI checks.

    Returns diagnostic indicators rather than claiming
    complete safety or factuality.
    """

    unsafe_request = detect_unsafe_request(
        question
    )

    uncertainty_present = (
        contains_uncertainty_language(
            answer
        )
    )

    source_language_present = (
        contains_source_language(
            answer
        )
    )

    checks = {
        "unsafe_request_detected":
            unsafe_request,

        "uncertainty_language_present":
            uncertainty_present,

        "source_language_present":
            source_language_present,

        "retrieved_context_available":
            has_retrieved_context,
    }

    warnings: List[str] = []

    if unsafe_request:

        warnings.append(
            "Potentially unsafe cybersecurity request detected."
        )

    if (
        not has_retrieved_context
        and not uncertainty_present
        and not unsafe_request
    ):

        warnings.append(
            "Answer may not clearly communicate "
            "the absence of supporting knowledge-base evidence."
        )

    if (
        has_retrieved_context
        and not source_language_present
    ):

        warnings.append(
            "Answer does not contain obvious source-related language."
        )

    checks["warnings"] = warnings

    return checks
