from typing import List, Dict


def build_rag_context(
    search_results: List[Dict],
) -> str:
    """
    Convert retrieved FAISS results into context
    that can be provided to the language model.
    """

    if not search_results:
        return ""

    context_parts = []

    for number, result in enumerate(
        search_results,
        start=1,
    ):

        source = result.get(
            "source",
            "Unknown source",
        )

        page = result.get(
            "page",
            "Unknown page",
        )

        text = result.get(
            "text",
            "",
        )

        context_parts.append(
            f"""
SOURCE {number}

Document: {source}
Page: {page}

Content:
{text}
"""
        )

    return "\n".join(context_parts)


def build_rag_prompt(
    question: str,
    context: str,
    student_level: str,
    learning_mode: str,
    response_length: str,
) -> str:
    """
    Build the grounded prompt for the LLM.
    """

    return f"""
You are an AI Learning Tutor specializing in cybersecurity education.

The student asked:

{question}

Student level:
{student_level}

Learning mode:
{learning_mode}

Preferred response length:
{response_length}

You have been provided with information retrieved from
the cybersecurity learning knowledge base.

================ KNOWLEDGE BASE =================

{context}

===================================================

IMPORTANT GROUNDING RULES:

1. Use the provided knowledge base as the primary source
   for factual claims.

2. Do not claim that information came from the knowledge
   base unless it is actually supported by the retrieved
   content.

3. Do not invent citations, document names, or page numbers.

4. If the knowledge base does not contain enough information
   to answer the question, clearly say that the available
   learning material does not contain sufficient information.

5. You may provide a small amount of general explanation
   when useful, but clearly distinguish it from information
   supported by the retrieved material.

6. Never fabricate facts to make the answer appear complete.

7. Adapt the explanation to the student's level.

8. For cybersecurity topics, keep the explanation educational,
   defensive, and responsible.

9. At the end, provide a "Sources" section containing only
   the documents and pages that were actually used.

10. If a source was not useful for the answer, do not cite it.

Answer the student's question now.
"""
