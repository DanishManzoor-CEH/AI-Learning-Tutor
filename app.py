import json
from datetime import datetime

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .research-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #f8f9fa;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 AI Learning Tutor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    RAG-Based Responsible AI Tutor for Cybersecurity Learning
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GROQ API
# ============================================================

try:

    groq_api_key = st.secrets[
        "GROQ_API_KEY"
    ]

except Exception:

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY to Streamlit Cloud Secrets."
    )

    st.stop()


client = Groq(
    api_key=groq_api_key
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an educational AI tutor specializing in
cybersecurity.

Your role is to help students understand cybersecurity
concepts clearly and responsibly.

Important principles:

- Be accurate.
- Be educational.
- Adapt explanations to the student's level.
- Encourage understanding.
- Do not fabricate information.
- Do not fabricate citations.
- Clearly distinguish knowledge-base-supported
  information from unsupported information.
- Remain defensive and responsible when discussing
  cybersecurity.
"""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Learning Settings"
    )

    student_level = st.selectbox(
        "Student Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
        ],
        index=0,
    )

    learning_mode = st.selectbox(
        "Learning Mode",
        [
            "Explain",
            "Simple Explanation",
            "Technical Explanation",
            "Socratic Learning",
        ],
        index=0,
    )

    response_length = st.selectbox(
        "Response Length",
        [
            "Short",
            "Medium",
            "Detailed",
        ],
        index=1,
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = load_pdf_documents()

statistics = get_document_statistics()


# ============================================================
# CREATE CHUNKS
# ============================================================

chunks = chunk_documents(
    documents,
    chunk_size=800,
    chunk_overlap=150,
)


# ============================================================
# BUILD VECTOR STORE
# ============================================================

vector_index = None

if chunks:

    try:

        vector_index = build_vector_store(
            tuple(chunks)
        )

    except Exception as error:

        st.error(
            "Could not build the knowledge base."
        )

        st.error(
            str(error)
        )


# ============================================================
# TUTOR RESPONSE — RAG
# ============================================================

def generate_rag_response(
    question,
    student_level,
    learning_mode,
    response_length,
    search_results,
):
    """
    Generate a knowledge-grounded response.
    """

    if not search_results:

        return (
            "I couldn't find sufficient information "
            "about this question in the current "
            "cybersecurity learning knowledge base."
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

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",

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

            temperature=0.2,

            max_tokens=1800,
        )

        return response.choices[
            0
        ].message.content

    except Exception as error:

        return (
            "Error generating RAG response: "
            f"{error}"
        )


# ============================================================
# TUTOR RESPONSE — BASELINE
# ============================================================

def generate_baseline_response(
    question,
    student_level,
    learning_mode,
    response_length,
):
    """
    Generate a baseline response using the same LLM
    WITHOUT retrieval.

    This provides a comparison condition for the research.
    """

    prompt = f"""
You are an AI Learning Tutor specializing in cybersecurity.

Answer the following student question directly.

Student question:
{question}

Student level:
{student_level}

Learning mode:
{learning_mode}

Response length:
{response_length}

Answer using your general model knowledge.

Do not use or mention a cybersecurity PDF knowledge base.

Do not fabricate citations.
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",

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

            temperature=0.2,

            max_tokens=1800,
        )

        return response.choices[
            0
        ].message.content

    except Exception as error:

        return (
            "Error generating baseline response: "
            f"{error}"
        )


# ============================================================
# SIDEBAR KNOWLEDGE BASE STATUS
# ============================================================

with st.sidebar:

    st.divider()

    st.header(
        "📚 Knowledge Base"
    )

    st.metric(
        "PDF Documents",
        statistics["pdf_count"],
    )

    st.metric(
        "Pages",
        statistics["page_count"],
    )

    st.metric(
        "Text Chunks",
        len(chunks),
    )

    if statistics["documents"]:

        st.caption(
            "Documents:"
        )

        for document in statistics[
            "documents"
        ]:

            st.caption(
                f"📄 {document}"
            )


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🤖 AI Tutor",
        "🔬 Evaluation",
        "📊 Research Results",
    ]
)


# ============================================================
# TAB 1 — AI TUTOR
# ============================================================

with tab1:

    st.subheader(
        "🤖 Ask Your AI Tutor"
    )

    question = st.text_area(
        "Enter your cybersecurity question",
        placeholder=(
            "Example: What is DNS?"
        ),
        height=120,
    )

    ask_button = st.button(
        "🚀 Ask AI Tutor",
        type="primary",
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        elif vector_index is None:

            st.warning(
                "The cybersecurity knowledge base "
                "is not available."
            )

        else:

            with st.spinner(
                "🔎 Searching knowledge base..."
            ):

                search_results = search_vector_store(
                    vector_index,
                    chunks,
                    question,
                    top_k=4,
                )

            if not search_results:

                st.warning(
                    """
                    I couldn't find sufficient information
                    about this question in the current
                    cybersecurity learning knowledge base.
                    """
                )

            else:

                with st.spinner(
                    "🧠 Generating grounded response..."
                ):

                    answer = generate_rag_response(
                        question,
                        student_level,
                        learning_mode,
                        response_length,
                        search_results,
                    )

                st.subheader(
                    "🤖 Tutor Response"
                )

                st.markdown(
                    answer
                )

                st.divider()

                st.subheader(
                    "📚 Retrieved Evidence"
                )

                for number, result in enumerate(
                    search_results,
                    start=1,
                ):

                    with st.expander(
                        f"Source {number}: "
                        f"{result['source']} "
                        f"— Page {result['page']}"
                    ):

                        st.write(
                            f"**Similarity:** "
                            f"{result['similarity']:.3f}"
                        )

                        st.write(
                            result["text"]
                        )


# ============================================================
# TAB 2 — EVALUATION
# ============================================================

with tab2:

    st.header(
        "🔬 RAG Evaluation"
    )

    st.write(
        """
        This experiment compares two systems:

        **Baseline:** GPT-OSS without retrieval

        **RAG:** GPT-OSS with retrieved cybersecurity
        knowledge-base context.

        Both systems use the same underlying language model.
        The main experimental difference is the presence or
        absence of retrieval.
        """
    )

    st.info(
        """
        Evaluation metrics are lightweight research-prototype
        metrics. Keyword coverage and token F1 measure overlap
        with reference answers. Grounding score is a lexical
        heuristic and is NOT a complete factuality metric.
        """
    )

    evaluation_questions = (
        load_evaluation_questions()
    )

    st.write(
        f"Evaluation questions loaded: "
        f"**{len(evaluation_questions)}**"
    )

    if evaluation_questions:

        preview_rows = []

        for item in evaluation_questions:

            preview_rows.append(
                {
                    "ID": item["id"],
                    "Question": item["question"],
                    "Topic": ", ".join(
                        item.get(
                            "expected_topics",
                            [],
                        )
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                preview_rows
            ),
            use_container_width=True,
        )

    run_evaluation = st.button(
        "▶️ Run Full Evaluation",
        type="primary",
    )

    if run_evaluation:

        if not evaluation_questions:

            st.error(
                "No evaluation questions were found."
            )

        elif vector_index is None:

            st.error(
                "Knowledge base is unavailable."
            )

        else:

            results = []

            progress_bar = st.progress(
                0
            )

            status = st.empty()

            total = len(
                evaluation_questions
            )

            for index, item in enumerate(
                evaluation_questions,
                start=1,
            ):

                question_id = item[
                    "id"
                ]

                question = item[
                    "question"
                ]

                reference_answer = item[
                    "reference_answer"
                ]

                expected_keywords = item[
                    "expected_keywords"
                ]

                status.write(
                    f"Evaluating {question_id}: "
                    f"{question}"
                )

                # --------------------------------------------
                # BASELINE
                # --------------------------------------------

                baseline_answer = (
                    generate_baseline_response(
                        question,
                        student_level,
                        learning_mode,
                        response_length,
                    )
                )

                # --------------------------------------------
                # RETRIEVAL
                # --------------------------------------------

                search_results = (
                    search_vector_store(
                        vector_index,
                        chunks,
                        question,
                        top_k=4,
                    )
                )

                # --------------------------------------------
                # RAG
                # --------------------------------------------

                rag_answer = (
                    generate_rag_response(
                        question,
                        student_level,
                        learning_mode,
                        response_length,
                        search_results,
                    )
                )

                # --------------------------------------------
                # CONTEXT
                # --------------------------------------------

                context = build_rag_context(
                    search_results
                )

                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                baseline_keyword_score = (
                    keyword_coverage(
                        baseline_answer,
                        expected_keywords,
                    )
                )

                rag_keyword_score = (
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
                        context,
                    )
                )

                retrieval_ok = (
                    retrieval_success(
                        search_results
                    )
                )

                results.append(
                    {
                        "id": question_id,
                        "question": question,

                        "baseline_answer":
                            baseline_answer,

                        "rag_answer":
                            rag_answer,

                        "baseline_keyword_coverage":
                            baseline_keyword_score,

                        "rag_keyword_coverage":
                            rag_keyword_score,

                        "baseline_token_f1":
                            baseline_f1,

                        "rag_token_f1":
                            rag_f1,

                        "rag_grounding_score":
                            rag_grounding,

                        "retrieval_success":
                            retrieval_ok,

                        "retrieved_sources":
                            [
                                f"{r['source']} "
                                f"Page {r['page']}"
                                for r in search_results
                            ],
                    }
                )

                progress_bar.progress(
                    index / total
                )

            status.success(
                "Evaluation completed successfully."
            )

            # --------------------------------------------
            # SAVE TO SESSION STATE
            # --------------------------------------------

            st.session_state[
                "evaluation_results"
            ] = results

            st.session_state[
                "evaluation_timestamp"
            ] = datetime.now().isoformat(
                timespec="seconds"
            )


# ============================================================
# TAB 3 — RESEARCH RESULTS
# ============================================================

with tab3:

    st.header(
        "📊 Research Results"
    )

    results = st.session_state.get(
        "evaluation_results",
        [],
    )

    if not results:

        st.info(
            """
            No evaluation results are available yet.

            Go to the Evaluation tab and click:

            **Run Full Evaluation**
            """
        )

    else:

        summary = summarize_results(
            results
        )

        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Baseline Keyword Coverage",
                f"{summary['baseline_keyword_coverage']:.2%}",
            )

        with col2:

            st.metric(
                "RAG Keyword Coverage",
                f"{summary['rag_keyword_coverage']:.2%}",
            )

        with col3:

            st.metric(
                "Baseline Token F1",
                f"{summary['baseline_token_f1']:.2%}",
            )

        with col4:

            st.metric(
                "RAG Token F1",
                f"{summary['rag_token_f1']:.2%}",
            )

        st.divider()

        col5, col6 = st.columns(2)

        with col5:

            st.metric(
                "RAG Grounding Heuristic",
                f"{summary['rag_grounding_score']:.2%}",
            )

        with col6:

            st.metric(
                "Retrieval Success Rate",
                f"{summary['retrieval_success_rate']:.2%}",
            )

        # ----------------------------------------------------
        # COMPARISON TABLE
        # ----------------------------------------------------

        st.subheader(
            "Baseline vs RAG"
        )

        comparison_rows = []

        for result in results:

            comparison_rows.append(
                {
                    "Question ID":
                        result["id"],

                    "Question":
                        result["question"],

                    "Baseline Keyword Coverage":
                        round(
                            result[
                                "baseline_keyword_coverage"
                            ],
                            3,
                        ),

                    "RAG Keyword Coverage":
                        round(
                            result[
                                "rag_keyword_coverage"
                            ],
                            3,
                        ),

                    "Baseline Token F1":
                        round(
                            result[
                                "baseline_token_f1"
                            ],
                            3,
                        ),

                    "RAG Token F1":
                        round(
                            result[
                                "rag_token_f1"
                            ],
                            3,
                        ),

                    "RAG Grounding":
                        round(
                            result[
                                "rag_grounding_score"
                            ],
                            3,
                        ),

                    "Retrieval":
                        "Yes"
                        if result[
                            "retrieval_success"
                        ]
                        else "No",
                }
            )

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # BAR CHART
        # ----------------------------------------------------

        st.subheader(
            "Average Metric Comparison"
        )

        chart_data = pd.DataFrame(
            {
                "System": [
                    "Baseline",
                    "RAG",
                ],

                "Keyword Coverage": [
                    summary[
                        "baseline_keyword_coverage"
                    ],
                    summary[
                        "rag_keyword_coverage"
                    ],
                ],

                "Token F1": [
                    summary[
                        "baseline_token_f1"
                    ],
                    summary[
                        "rag_token_f1"
                    ],
                ],
            }
        )

        st.bar_chart(
            chart_data.set_index(
                "System"
            )
        )

        # ----------------------------------------------------
        # INDIVIDUAL RESULTS
        # ----------------------------------------------------

        st.subheader(
            "Detailed Evaluation Results"
        )

        for result in results:

            with st.expander(
                f"{result['id']} — "
                f"{result['question']}"
            ):

                st.markdown(
                    "### Baseline Answer"
                )

                st.write(
                    result[
                        "baseline_answer"
                    ]
                )

                st.markdown(
                    "### RAG Answer"
                )

                st.write(
                    result[
                        "rag_answer"
                    ]
                )

                st.markdown(
                    "### Metrics"
                )

                metric_data = {
                    "Baseline Keyword Coverage":
                        result[
                            "baseline_keyword_coverage"
                        ],

                    "RAG Keyword Coverage":
                        result[
                            "rag_keyword_coverage"
                        ],

                    "Baseline Token F1":
                        result[
                            "baseline_token_f1"
                        ],

                    "RAG Token F1":
                        result[
                            "rag_token_f1"
                        ],

                    "RAG Grounding":
                        result[
                            "rag_grounding_score"
                        ],
                }

                st.json(
                    metric_data
                )

                st.markdown(
                    "### Retrieved Sources"
                )

                if result[
                    "retrieved_sources"
                ]:

                    for source in result[
                        "retrieved_sources"
                    ]:

                        st.write(
                            f"📄 {source}"
                        )

                else:

                    st.write(
                        "No sources retrieved."
                    )

        # ----------------------------------------------------
        # DOWNLOAD RESULTS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "⬇️ Export Evaluation"
        )

        results_json = json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            label="Download Detailed JSON Results",
            data=results_json,
            file_name=(
                "rag_evaluation_results.json"
            ),
            mime="application/json",
        )

        csv_data = (
            comparison_df
            .to_csv(
                index=False
            )
        )

        st.download_button(
            label="Download Comparison CSV",
            data=csv_data,
            file_name=(
                "rag_baseline_comparison.csv"
            ),
            mime="text/csv",
        )

        # ----------------------------------------------------
        # RESEARCH INTERPRETATION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🔬 Research Interpretation"
        )

        st.markdown(
            """
            This experiment compares the same language model
            under two conditions:

            **Condition A — Baseline**

            The model receives only the student's question.

            **Condition B — RAG**

            The model receives the student's question plus
            retrieved passages from the cybersecurity
            knowledge base.

            The purpose is to investigate whether retrieval
            improves source-grounded answering.

            The current automatic metrics are intended for
            prototype evaluation. Human assessment should be
            added in a later phase to evaluate factual
            correctness, educational usefulness, clarity,
            and quality of explanations.
            """
        )


# ============================================================
# RESEARCH FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="research-box">

    <strong>Research Question</strong>

    <br><br>

    How can Retrieval-Augmented Generation improve the
    factual accuracy, source-groundedness, and
    learning-support quality of an LLM-based cybersecurity
    tutoring system?

    <br><br>

    <strong>Phase 3 Experimental Design</strong>

    <br><br>

    LLM-only baseline → RAG system → Automatic comparison →
    Research results

    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "AI Learning Tutor — RAG-based responsible AI "
    "education research project."
)
