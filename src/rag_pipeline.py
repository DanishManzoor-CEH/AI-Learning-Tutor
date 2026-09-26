from typing import List, Dict


def build_rag_context(
    search_results: List[Dict],
) -> str:
    """
    Convert retrieved search results into a structured
    context block for the language model.
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

        similarity = result.get(
            "similarity",
            0.0,
        )

        context_parts.append(
            f"""
SOURCE {number}

Document: {source}
Page: {page}
Similarity: {similarity:.3f}

Content:
{text}
"""
        )

    return "\n".join(
        context_parts
    )


def build_rag_prompt(
    question: str,
    context: str,
    student_level: str,
    learning_mode: str,
    response_length: str,
) -> str:
    """
    Build a strict knowledge-grounded prompt.

    The model is instructed NOT to answer from general
    knowledge when the knowledge base does not contain
    sufficient information.
    """

    return f"""
You are an AI Learning Tutor for cybersecurity education.

Your task is to answer the student's question using ONLY
the information contained in the provided cybersecurity
knowledge base.

STUDENT QUESTION:
{question}

STUDENT LEVEL:
{student_level}

LEARNING MODE:
{learning_mode}

RESPONSE LENGTH:
{response_length}

================ KNOWLEDGE BASE =================

{context}

====================================================

STRICT GROUNDING RULES:

1. Use the retrieved knowledge-base content as the
   authoritative source for your answer.

2. Do NOT use your general world knowledge to answer
   information that is not supported by the retrieved
   knowledge-base content.

3. If the student's question is unrelated to cybersecurity
   or is not supported by the retrieved knowledge base,
   clearly say:

   "I couldn't find sufficient information about this
   question in the current cybersecurity learning
   knowledge base."

4. Do NOT guess.

5. Do NOT fabricate facts.

6. Do NOT fabricate citations.

7. Do NOT invent document names.

8. Do NOT invent page numbers.

9. Only mention a source when the retrieved content from
   that source actually supports the answer.

10. If multiple sources support the answer, mention only
    the relevant sources.

11. Adapt the explanation to the student's level.

12. Keep cybersecurity explanations educational,
    responsible, and defensive.

13. If the knowledge base is insufficient, do not try to
    make the answer look complete.

14. The source information is included below. Use it
    carefully and faithfully.

15. At the end of a supported answer, provide:

    Sources:
    - Document name — Page X

16. If the knowledge base does not support the answer,
    do NOT provide a Sources section.

Answer the student now.
"""
