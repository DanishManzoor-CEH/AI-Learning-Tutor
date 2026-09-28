import io
import re

import pandas as pd
import streamlit as st
from groq import Groq

from src.document_loader import (
    load_pdf_documents,
    get_document_statistics,
)

from src.text_chunker import (
    chunk_documents,
)

from src.vector_store import (
    build_vector_store,
    search_vector_store,
)

from src.rag_pipeline import (
    build_rag_context,
    build_rag_prompt,
)

from src.evaluator import (
    load_evaluation_questions,
    keyword_coverage,
    token_f1_score,
    grounding_score,
    retrieval_success,
    summarize_results,
    summarize_by_category,
    calculate_improvement,
    calculate_relative_improvement,
    HUMAN_EVALUATION_DIMENSIONS,
    calculate_human_evaluation_average,
    calculate_interpretation,
)

from src.responsible_ai import (
    detect_unsafe_request,
    build_safe_response,
    evaluate_responsible_ai,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Learning Tutor",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

DOCUMENTS_DIR = "data/documents"

EVALUATION_FILE = (
    "data/evaluation/questions.json"
)

TOP_K = 4


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI Learning Tutor specializing in cybersecurity education.

Your purpose is to help students understand cybersecurity concepts
clearly, accurately, and responsibly.

Teaching principles:

1. Adapt explanations to the student's level.
2. Explain concepts step by step.
3. Use simple examples when appropriate.
4. Encourage understanding rather than memorization.
5. Ask a short follow-up question when useful.
6. Clearly communicate uncertainty.
7. Never invent sources, citations, page numbers, or facts.
8. For retrieved knowledge-base answers, prioritize the supplied
   evidence.
9. Keep cybersecurity explanations educational and defensive.
10. Do not provide harmful operational instructions for malware,
    credential theft, phishing infrastructure, or other attacks.

When knowledge-base evidence is insufficient, say so clearly.
"""


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        return None

    return Groq(
        api_key=api_key
    )


client = get_groq_client()


# ============================================================
# KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge_base():

    documents = load_pdf_documents(
        DOCUMENTS_DIR
    )

    chunks = chunk_documents(
        documents,
        chunk_size=800,
        chunk_overlap=150,
    )

    if not chunks:

        return (
            None,
            [],
            documents,
        )

    # IMPORTANT:
    # build_vector_store returns BOTH the FAISS index
    # and the metadata list.
    vector_index, chunk_metadata = (
        build_vector_store(chunks)
    )

    return (
        vector_index,
        chunk_metadata,
        documents,
    )


(
    vector_index,
    chunk_metadata,
    loaded_documents,
) = load_knowledge_base()


# ============================================================
# MODEL GENERATION
# ============================================================

def call_llm(
    prompt: str,
    temperature: float = 0.2,
) -> str:

    if client is None:

        return (
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
        )

        return response.choices[
            0
        ].message.content

    except Exception as error:

        return (
            f"Model request failed: {error}"
        )


# ============================================================
# BASELINE LLM
# ============================================================

def generate_baseline_response(
    question: str,
    student_level: str,
    learning_mode: str,
    response_length: str,
) -> str:

    prompt = f"""
Answer the following cybersecurity learning question.

Student level:
{student_level}

Learning mode:
{learning_mode}

Preferred response length:
{response_length}

Question:
{question}

Important:
This is the baseline condition.

Do NOT use the project's PDF knowledge base.
Do not invent citations.
If you are uncertain, communicate uncertainty.

Provide a clear educational answer.
"""

    return call_llm(
        prompt,
        temperature=0.2,
    )


# ============================================================
# RAG RESPONSE
# ============================================================

def generate_rag_response(
    question: str,
    student_level: str,
    learning_mode: str,
    response_length: str,
):

    if vector_index is None:

        return (
            "The knowledge base is currently unavailable.",
            [],
            "",
        )

    search_results = search_vector_store(
        vector_index,
        chunk_metadata,
        question,
        top_k=TOP_K,
    )

    context = build_rag_context(
        search_results
    )

    prompt = build_rag_prompt(
        question=question,
        context=context,
        student_level=student_level,
        learning_mode=learning_mode,
        response_length=response_length,
    )

    answer = call_llm(
        prompt,
        temperature=0.2,
    )

    return (
        answer,
        search_results,
        context,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "human_evaluations" not in st.session_state:

    st.session_state[
        "human_evaluations"
    ] = []


if "research_results" not in st.session_state:

    st.session_state[
        "research_results"
    ] = []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎓 AI Learning Tutor"
)

st.sidebar.markdown(
    """
**Research Project**

RAG-Based Responsible AI Tutor
for Cybersecurity Learning
"""
)

student_level = st.sidebar.selectbox(
    "Student Level",
    [
        "Beginner",
        "Intermediate",
        "Advanced",
    ],
)

learning_mode = st.sidebar.selectbox(
    "Learning Mode",
    [
        "Explain",
        "Simple Explanation",
        "Technical Explanation",
        "Socratic Learning",
    ],
)

response_length = st.sidebar.selectbox(
    "Response Length",
    [
        "Short",
        "Medium",
        "Detailed",
    ],
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Knowledge Base"
)

stats = get_document_statistics(
    DOCUMENTS_DIR
)

st.sidebar.write(
    f"PDF files: {stats['pdf_count']}"
)

st.sidebar.write(
    f"Loaded pages: {stats['page_count']}"
)

st.sidebar.write(
    f"Indexed chunks: {len(chunk_metadata)}"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Model: {MODEL_NAME}"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎓 AI Learning Tutor"
)

st.markdown(
    """
### RAG-Based Responsible AI Tutor for Cybersecurity Learning

This research prototype compares a **baseline LLM**
with a **Retrieval-Augmented Generation (RAG)** tutor
using a cybersecurity knowledge base.
"""
)


# ============================================================
# TABS
# ============================================================

tab_tutor, tab_eval, tab_research, tab_responsible = (
    st.tabs(
        [
            "🤖 AI Tutor",
            "📊 Evaluation",
            "🔬 Research Results",
            "🛡️ Responsible AI",
        ]
    )
)


# ============================================================
# TAB 1 — AI TUTOR
# ============================================================

with tab_tutor:

    st.header(
        "Ask the AI Learning Tutor"
    )

    question = st.text_area(
        "Enter your cybersecurity question:",
        placeholder=(
            "Example: What is the TCP three-way handshake?"
        ),
        height=120,
    )

    if st.button(
        "Ask Tutor",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif detect_unsafe_request(
            question
        ):

            st.warning(
                "This request requires a safe educational response."
            )

            st.write(
                build_safe_response(
                    question
                )
            )

        else:

            with st.spinner(
                "Generating grounded answer..."
            ):

                (
                    answer,
                    search_results,
                    context,
                ) = generate_rag_response(
                    question,
                    student_level,
                    learning_mode,
                    response_length,
                )

            st.subheader(
                "Tutor Answer"
            )

            st.write(
                answer
            )

            st.markdown("---")

            st.subheader(
                "Retrieved Sources"
            )

            if search_results:

                for index, result in enumerate(
                    search_results,
                    start=1,
                ):

                    st.markdown(
                        f"""
**Source {index}**

- Document: `{result.get('source')}`
- Page: `{result.get('page')}`
- Similarity: `{result.get('similarity', 0):.4f}`
"""
                    )

                    with st.expander(
                        "View retrieved content"
                    ):

                        st.write(
                            result.get(
                                "text",
                                "",
                            )
                        )

            else:

                st.info(
                    "No knowledge-base context was retrieved."
                )


# ============================================================
# TAB 2 — EVALUATION
# ============================================================

with tab_eval:

    st.header(
        "📊 Automated Evaluation"
    )

    evaluation_questions = (
        load_evaluation_questions(
            EVALUATION_FILE
        )
    )

    st.write(
        f"Evaluation dataset: "
        f"**{len(evaluation_questions)} questions**"
    )

    if not evaluation_questions:

        st.error(
            "No evaluation questions were found."
        )

    elif client is None:

        st.error(
            "GROQ_API_KEY is not configured."
        )

    else:

        st.markdown(
            """
The experiment compares:

**Baseline:** GPT-OSS without the project knowledge base.

**RAG:** GPT-OSS with FAISS-retrieved cybersecurity context.
"""
        )

        run_experiment = st.button(
            "▶ Run Phase 5 Experiment",
            type="primary",
        )

        if run_experiment:

            progress = st.progress(
                0
            )

            status = st.empty()

            experiment_results = []

            for position, item in enumerate(
                evaluation_questions,
                start=1,
            ):

                question_text = item[
                    "question"
                ]

                category = item.get(
                    "category",
                    "Unknown",
                )

                status.write(
                    f"Evaluating "
                    f"{position}/"
                    f"{len(evaluation_questions)}: "
                    f"{question_text}"
                )

                # ----------------------------
                # BASELINE
                # ----------------------------

                baseline_answer = (
                    generate_baseline_response(
                        question_text,
                        "Intermediate",
                        "Explain",
                        "Medium",
                    )
                )

                # ----------------------------
                # RAG
                # ----------------------------

                (
                    rag_answer,
                    search_results,
                    rag_context,
                ) = generate_rag_response(
                    question_text,
                    "Intermediate",
                    "Explain",
                    "Medium",
                )

                # ----------------------------
                # METRICS
                # ----------------------------

                expected_keywords = item.get(
                    "expected_keywords",
                    [],
                )

                reference_answer = item.get(
                    "reference_answer",
                    "",
                )

                baseline_keyword = (
                    keyword_coverage(
                        baseline_answer,
                        expected_keywords,
                    )
                )

                rag_keyword = (
                    keyword_coverage(
                        rag_answer,
                        expected_keywords,
                    )
                )

                baseline_f1 = (
                    token_f1_score(
                        baseline_answer,
                        reference_answer,
                    )
                )

                rag_f1 = (
                    token_f1_score(
                        rag_answer,
                        reference_answer,
                    )
                )

                rag_grounding = (
                    grounding_score(
                        rag_answer,
                        rag_context,
                    )
                )

                retrieval_ok = (
                    retrieval_success(
                        search_results
                    )
                )

                experiment_results.append(
                    {
                        "id": item.get(
                            "id",
                            position,
                        ),
                        "category": category,
                        "question": question_text,
                        "baseline_answer":
                            baseline_answer,
                        "rag_answer":
                            rag_answer,
                        "baseline_keyword_coverage":
                            baseline_keyword,
                        "rag_keyword_coverage":
                            rag_keyword,
                        "baseline_token_f1":
                            baseline_f1,
                        "rag_token_f1":
                            rag_f1,
                        "rag_grounding_score":
                            rag_grounding,
                        "retrieval_success":
                            retrieval_ok,
                    }
                )

                progress.progress(
                    position
                    / len(
                        evaluation_questions
                    )
                )

            status.success(
                "Phase 5 experiment completed."
            )

            st.session_state[
                "research_results"
            ] = experiment_results


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        results = st.session_state[
            "research_results"
        ]

        if results:

            summary = summarize_results(
                results
            )

            st.markdown("---")

            st.subheader(
                "Overall Experimental Results"
            )

            baseline_keyword = summary[
                "baseline_keyword_coverage"
            ]

            rag_keyword = summary[
                "rag_keyword_coverage"
            ]

            baseline_f1 = summary[
                "baseline_token_f1"
            ]

            rag_f1 = summary[
                "rag_token_f1"
            ]

            grounding = summary[
                "rag_grounding_score"
            ]

            retrieval = summary[
                "retrieval_success_rate"
            ]

            col1, col2, col3, col4 = st.columns(
                4
            )

            col1.metric(
                "Baseline Keyword Coverage",
                f"{baseline_keyword:.2%}",
            )

            col2.metric(
                "RAG Keyword Coverage",
                f"{rag_keyword:.2%}",
                f"{calculate_relative_improvement(baseline_keyword, rag_keyword):+.1f}%",
            )

            col3.metric(
                "Baseline Token F1",
                f"{baseline_f1:.3f}",
            )

            col4.metric(
                "RAG Token F1",
                f"{rag_f1:.3f}",
                f"{calculate_relative_improvement(baseline_f1, rag_f1):+.1f}%",
            )

            col5, col6 = st.columns(
                2
            )

            col5.metric(
                "RAG Grounding Score",
                f"{grounding:.2%}",
            )

            col6.metric(
                "Retrieval Success Rate",
                f"{retrieval:.2%}",
            )

            st.markdown("---")

            st.subheader(
                "Research Interpretation"
            )

            keyword_difference = (
                rag_keyword
                - baseline_keyword
            )

            f1_difference = (
                rag_f1
                - baseline_f1
            )

            st.write(
                f"""
Across **{summary['total_questions']} evaluation questions**:

- Baseline keyword coverage: **{baseline_keyword:.2%}**
- RAG keyword coverage: **{rag_keyword:.2%}**
- Absolute keyword coverage difference: **{keyword_difference:+.2%}**
- Baseline token F1: **{baseline_f1:.3f}**
- RAG token F1: **{rag_f1:.3f}**
- Absolute token F1 difference: **{f1_difference:+.3f}**
- RAG grounding heuristic: **{grounding:.2%}**
- Retrieval success rate: **{retrieval:.2%}**
"""
            )

            st.info(
                "These automatic metrics are lexical heuristics. "
                "They do not independently prove factual correctness, "
                "absence of hallucination, or educational quality. "
                "Human evaluation is therefore included separately."
            )

            # ------------------------------------------------
            # CATEGORY ANALYSIS
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "Category-Level Analysis"
            )

            category_summary = (
                summarize_by_category(
                    results
                )
            )

            category_rows = []

            for category, values in (
                category_summary.items()
            ):

                category_rows.append(
                    {
                        "Category": category,
                        "Questions": values[
                            "questions"
                        ],
                        "Baseline Keyword":
                            values[
                                "baseline_keyword_coverage"
                            ],
                        "RAG Keyword":
                            values[
                                "rag_keyword_coverage"
                            ],
                        "Keyword Improvement":
                            values[
                                "keyword_improvement"
                            ],
                        "Baseline F1":
                            values[
                                "baseline_token_f1"
                            ],
                        "RAG F1":
                            values[
                                "rag_token_f1"
                            ],
                        "F1 Improvement":
                            values[
                                "token_f1_improvement"
                            ],
                        "RAG Grounding":
                            values[
                                "rag_grounding_score"
                            ],
                        "Retrieval Success":
                            values[
                                "retrieval_success_rate"
                            ],
                    }
                )

            category_df = pd.DataFrame(
                category_rows
            )

            st.dataframe(
                category_df.style.format(
                    {
                        "Baseline Keyword":
                            "{:.2%}",
                        "RAG Keyword":
                            "{:.2%}",
                        "Keyword Improvement":
                            "{:+.2%}",
                        "Baseline F1":
                            "{:.3f}",
                        "RAG F1":
                            "{:.3f}",
                        "F1 Improvement":
                            "{:+.3f}",
                        "RAG Grounding":
                            "{:.2%}",
                        "Retrieval Success":
                            "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            # ------------------------------------------------
            # DOWNLOAD RESULTS
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "Download Experimental Data"
            )

            results_df = pd.DataFrame(
                results
            )

            csv_data = results_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(
                label="⬇️ Download Full Experiment CSV",
                data=csv_data,
                file_name=(
                    "phase5_experiment_results.csv"
                ),
                mime="text/csv",
            )


# ============================================================
# TAB 3 — RESEARCH RESULTS
# ============================================================

with tab_research:

    st.header(
        "🔬 Research Results"
    )

    results = st.session_state[
        "research_results"
    ]

    if not results:

        st.info(
            "Run the Phase 5 experiment from the "
            "Evaluation tab first."
        )

    else:

        summary = summarize_results(
            results
        )

        st.subheader(
            "Research Metrics"
        )

        research_table = pd.DataFrame(
            {
                "Metric": [
                    "Number of questions",
                    "Baseline keyword coverage",
                    "RAG keyword coverage",
                    "Baseline token F1",
                    "RAG token F1",
                    "RAG grounding score",
                    "Retrieval success rate",
                ],
                "Value": [
                    summary[
                        "total_questions"
                    ],
                    f"{summary['baseline_keyword_coverage']:.2%}",
                    f"{summary['rag_keyword_coverage']:.2%}",
                    f"{summary['baseline_token_f1']:.3f}",
                    f"{summary['rag_token_f1']:.3f}",
                    f"{summary['rag_grounding_score']:.2%}",
                    f"{summary['retrieval_success_rate']:.2%}",
                ],
            }
        )

        st.table(
            research_table
        )

        st.markdown("---")

        st.subheader(
            "Research Question"
        )

        st.write(
            """
**How can Retrieval-Augmented Generation improve the
factual accuracy, source-groundedness, and learning-support
quality of an LLM-based cybersecurity tutoring system?**
"""
        )

        st.markdown("---")

        st.subheader(
            "Baseline vs RAG"
        )

        baseline_keyword = summary[
            "baseline_keyword_coverage"
        ]

        rag_keyword = summary[
            "rag_keyword_coverage"
        ]

        baseline_f1 = summary[
            "baseline_token_f1"
        ]

        rag_f1 = summary[
            "rag_token_f1"
        ]

        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Keyword Coverage",
                    "Token F1",
                ],
                "Baseline LLM": [
                    baseline_keyword,
                    baseline_f1,
                ],
                "RAG Tutor": [
                    rag_keyword,
                    rag_f1,
                ],
            }
        )

        st.dataframe(
            comparison_df.style.format(
                {
                    "Baseline LLM":
                        "{:.3f}",
                    "RAG Tutor":
                        "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("---")

        st.subheader(
            "Human Evaluation"
        )

        if not st.session_state[
            "human_evaluations"
        ]:

            st.info(
                "No human evaluation has been recorded yet."
            )

        else:

            human_df = pd.DataFrame(
                st.session_state[
                    "human_evaluations"
                ]
            )

            st.dataframe(
                human_df,
                use_container_width=True,
            )

            numeric_columns = [
                column
                for column in HUMAN_EVALUATION_DIMENSIONS
                if column in human_df.columns
            ]

            if numeric_columns:

                human_average = (
                    human_df[
                        numeric_columns
                    ]
                    .mean()
                    .mean()
                )

                st.metric(
                    "Average Human Evaluation",
                    f"{human_average:.2f} / 5",
                )

                st.caption(
                    calculate_interpretation(
                        human_average
                    )
                )

        st.markdown("---")

        st.subheader(
            "Important Research Limitation"
        )

        st.write(
            """
The automatic evaluation in this prototype is based mainly
on lexical overlap and retrieval heuristics. These metrics
are useful for controlled experimentation but cannot fully
measure semantic correctness, hallucination, pedagogical
quality, or student learning gains.

A stronger future study would use multiple human evaluators,
larger datasets, statistical significance testing, and
controlled user studies.
"""
        )


# ============================================================
# TAB 4 — RESPONSIBLE AI
# ============================================================

with tab_responsible:

    st.header(
        "🛡️ Responsible AI"
    )

    st.markdown(
        """
This section evaluates responsible behavior of the tutor,
including unsafe-request handling, uncertainty communication,
source awareness, and knowledge-base availability.
"""
    )

    st.subheader(
        "Safety Test"
    )

    safety_question = st.text_input(
        "Enter a cybersecurity request to test:"
    )

    if st.button(
        "Run Responsible AI Check"
    ):

        if not safety_question.strip():

            st.warning(
                "Enter a question first."
            )

        else:

            unsafe = detect_unsafe_request(
                safety_question
            )

            if unsafe:

                safety_answer = (
                    build_safe_response(
                        safety_question
                    )
                )

            else:

                (
                    safety_answer,
                    safety_results,
                    safety_context,
                ) = generate_rag_response(
                    safety_question,
                    student_level,
                    learning_mode,
                    response_length,
                )

            safety_report = (
                evaluate_responsible_ai(
                    safety_question,
                    safety_answer,
                    bool(
                        vector_index
                    ),
                )
            )

            st.subheader(
                "Generated Response"
            )

            st.write(
                safety_answer
            )

            st.subheader(
                "Responsible AI Checks"
            )

            st.json(
                safety_report
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    """
AI Learning Tutor — RAG-Based Responsible AI Tutor
for Cybersecurity Learning | Research Prototype
"""
)
