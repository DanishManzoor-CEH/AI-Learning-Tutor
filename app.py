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
You are an educational AI tutor specializing in cybersecurity.

Your role is to help students understand cybersecurity
concepts clearly, accurately, and responsibly.

Important principles:

- Be accurate.
- Be educational.
- Adapt explanations to the student's level.
- Encourage understanding.
- Do not fabricate information.
- Do not fabricate citations.
- Do not claim that unsupported information came from
  the knowledge base.
- Clearly communicate uncertainty when evidence is insufficient.
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
# CHUNK DOCUMENTS
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
# RAG RESPONSE
# ============================================================

def generate_rag_response(
    question,
    student_level,
    learning_mode,
    response_length,
    search_results,
):
    """
    Generate a grounded RAG response.
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
# BASELINE RESPONSE
# ============================================================

def generate_baseline_response(
    question,
    student_level,
    learning_mode,
    response_length,
):
    """
    Generate an LLM-only baseline response.
    """

    prompt = f"""
You are an AI Learning Tutor specializing in cybersecurity.

Answer this student question using general model knowledge.

Question:
{question}

Student level:
{student_level}

Learning mode:
{learning_mode}

Response length:
{response_length}

Do not use a document knowledge base.
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
# SIDEBAR KNOWLEDGE BASE
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

        for document in statistics[
            "documents"
        ]:

            st.caption(
                f"📄 {document}"
            )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🤖 AI Tutor",
        "🔬 Evaluation",
        "📊 Research Results",
        "🛡️ Responsible AI",
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

        elif detect_unsafe_request(
            question
        ):

            st.warning(
                "This request requires a safety-restricted response."
            )

            st.markdown(
                build_safe_response(
                    question
                )
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

                answer = (
                    "I couldn't find sufficient information "
                    "about this question in the current "
                    "cybersecurity learning knowledge base."
                )

                st.warning(
                    answer
                )

                responsible_result = (
                    evaluate_responsible_ai(
                        question,
                        answer,
                        False,
                    )
                )

                if responsible_result[
                    "warnings"
                ]:

                    with st.expander(
                        "🛡️ Responsible AI Check"
                    ):

                        for warning in responsible_result[
                            "warnings"
                        ]:

                            st.write(
                                f"⚠️ {warning}"
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

                # --------------------------------------------
                # RESPONSIBLE AI CHECK
                # --------------------------------------------

                responsible_result = (
                    evaluate_responsible_ai(
                        question,
                        answer,
                        True,
                    )
                )

                if responsible_result[
                    "warnings"
                ]:

                    with st.expander(
                        "🛡️ Responsible AI Diagnostics"
                    ):

                        for warning in responsible_result[
                            "warnings"
                        ]:

                            st.write(
                                f"⚠️ {warning}"
                            )

                # --------------------------------------------
                # SOURCES
                # --------------------------------------------

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
                            f"Similarity: "
                            f"{result['similarity']:.3f}"
                        )

                        st.write(
                            result["text"]
                        )


# ============================================================
# TAB 2 — AUTOMATIC EVALUATION
# ============================================================

with tab2:

    st.header(
        "🔬 Automatic Evaluation"
    )

    st.write(
        """
        This experiment compares an LLM-only baseline with
        the RAG system using the same GPT-OSS model.
        """
    )

    evaluation_questions = (
        load_evaluation_questions()
    )

    st.write(
        f"Evaluation questions: "
        f"**{len(evaluation_questions)}**"
    )

    run_evaluation = st.button(
        "▶️ Run Full Evaluation",
        type="primary",
    )

    if run_evaluation:

        if not evaluation_questions:

            st.error(
                "No evaluation questions found."
            )

        elif vector_index is None:

            st.error(
                "Knowledge base is unavailable."
            )

        else:

            results = []

            progress = st.progress(
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
                    f"Evaluating {item['id']} — "
                    f"{question}"
                )

                baseline_answer = (
                    generate_baseline_response(
                        question,
                        student_level,
                        learning_mode,
                        response_length,
                    )
                )

                search_results = (
                    search_vector_store(
                        vector_index,
                        chunks,
                        question,
                        top_k=4,
                    )
                )

                rag_answer = (
                    generate_rag_response(
                        question,
                        student_level,
                        learning_mode,
                        response_length,
                        search_results,
                    )
                )

                context = build_rag_context(
                    search_results
                )

                results.append(
                    {
                        "id":
                            item["id"],

                        "question":
                            question,

                        "evaluation_type":
                            item.get(
                                "evaluation_type",
                                "in_scope",
                            ),

                        "baseline_answer":
                            baseline_answer,

                        "rag_answer":
                            rag_answer,

                        "baseline_keyword_coverage":
                            keyword_coverage(
                                baseline_answer,
                                expected_keywords,
                            ),

                        "rag_keyword_coverage":
                            keyword_coverage(
                                rag_answer,
                                expected_keywords,
                            ),

                        "baseline_token_f1":
                            token_f1_score(
                                baseline_answer,
                                reference_answer,
                            ),

                        "rag_token_f1":
                            token_f1_score(
                                rag_answer,
                                reference_answer,
                            ),

                        "rag_grounding_score":
                            grounding_score(
                                rag_answer,
                                context,
                            ),

                        "retrieval_success":
                            retrieval_success(
                                search_results
                            ),

                        "retrieved_sources":
                            [
                                f"{r['source']} "
                                f"Page {r['page']}"
                                for r in search_results
                            ],
                    }
                )

                progress.progress(
                    index / total
                )

            status.success(
                "Automatic evaluation completed."
            )

            st.session_state[
                "evaluation_results"
            ] = results

            st.session_state[
                "evaluation_timestamp"
            ] = datetime.now().isoformat(
                timespec="seconds"
            )


# ============================================================
# TAB 3 — RESEARCH RESULTS + HUMAN EVALUATION
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
            Run the automatic evaluation first.

            Then this section will allow you to review
            and manually evaluate generated answers.
            """
        )

    else:

        summary = summarize_results(
            results
        )

        # ----------------------------------------------------
        # AUTOMATIC METRICS
        # ----------------------------------------------------

        st.subheader(
            "Automatic Metrics"
        )

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

        col5, col6 = st.columns(2)

        with col5:

            st.metric(
                "RAG Grounding Heuristic",
                f"{summary['rag_grounding_score']:.2%}",
            )

        with col6:

            st.metric(
                "Retrieval Success",
                f"{summary['retrieval_success_rate']:.2%}",
            )

        st.divider()

        # ----------------------------------------------------
        # HUMAN EVALUATION
        # ----------------------------------------------------

        st.header(
            "👤 Human Evaluation"
        )

        st.write(
            """
            Review each RAG answer and rate it from 1 to 5.

            **1 = Very poor**

            **2 = Poor**

            **3 = Acceptable**

            **4 = Good**

            **5 = Excellent**

            Use the same criteria for every answer to reduce
            evaluator inconsistency.
            """
        )

        human_results = st.session_state.get(
            "human_evaluation_results",
            [],
        )

        for result in results:

            result_id = result[
                "id"
            ]

            with st.expander(
                f"{result_id} — "
                f"{result['question']}"
            ):

                st.markdown(
                    "### RAG Answer"
                )

                st.write(
                    result[
                        "rag_answer"
                    ]
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
                        "No retrieved sources."
                    )

                st.divider()

                st.markdown(
                    "### Rate this answer"
                )

                scores = {}

                scores[
                    "factual_correctness"
                ] = st.slider(
                    "1. Factual Correctness",
                    1,
                    5,
                    3,
                    key=f"{result_id}_factual",
                )

                scores[
                    "groundedness"
                ] = st.slider(
                    "2. Groundedness / Source Support",
                    1,
                    5,
                    3,
                    key=f"{result_id}_grounding",
                )

                scores[
                    "educational_usefulness"
                ] = st.slider(
                    "3. Educational Usefulness",
                    1,
                    5,
                    3,
                    key=f"{result_id}_education",
                )

                scores[
                    "clarity"
                ] = st.slider(
                    "4. Clarity",
                    1,
                    5,
                    3,
                    key=f"{result_id}_clarity",
                )

                scores[
                    "difficulty_appropriateness"
                ] = st.slider(
                    "5. Appropriate Difficulty",
                    1,
                    5,
                    3,
                    key=f"{result_id}_difficulty",
                )

                scores[
                    "overall_quality"
                ] = st.slider(
                    "6. Overall Quality",
                    1,
                    5,
                    3,
                    key=f"{result_id}_overall",
                )

                comments = st.text_area(
                    "Evaluator Comments",
                    key=f"{result_id}_comments",
                    placeholder=(
                        "Example: Accurate explanation, "
                        "but could provide a simpler example."
                    ),
                )

                if st.button(
                    f"Save Evaluation — {result_id}",
                    key=f"save_{result_id}",
                ):

                    average_score = (
                        calculate_human_evaluation_average(
                            scores
                        )
                    )

                    human_record = {
                        "id":
                            result_id,

                        "question":
                            result["question"],

                        "factual_correctness":
                            scores[
                                "factual_correctness"
                            ],

                        "groundedness":
                            scores[
                                "groundedness"
                            ],

                        "educational_usefulness":
                            scores[
                                "educational_usefulness"
                            ],

                        "clarity":
                            scores[
                                "clarity"
                            ],

                        "difficulty_appropriateness":
                            scores[
                                "difficulty_appropriateness"
                            ],

                        "overall_quality":
                            scores[
                                "overall_quality"
                            ],

                        "average_score":
                            average_score,

                        "interpretation":
                            calculate_interpretation(
                                average_score
                            ),

                        "comments":
                            comments,
                    }

                    existing = [
                        item
                        for item in human_results
                        if item["id"] != result_id
                    ]

                    existing.append(
                        human_record
                    )

                    human_results = existing

                    st.session_state[
                        "human_evaluation_results"
                    ] = human_results

                    st.success(
                        f"Evaluation saved for {result_id}."
                    )

        # ----------------------------------------------------
        # HUMAN RESULTS SUMMARY
        # ----------------------------------------------------

        if human_results:

            st.divider()

            st.header(
                "📈 Human Evaluation Summary"
            )

            human_df = pd.DataFrame(
                human_results
            )

            st.dataframe(
                human_df,
                use_container_width=True,
            )

            average_human_score = (
                human_df[
                    "average_score"
                ].mean()
            )

            st.metric(
                "Average Human Evaluation Score",
                f"{average_human_score:.2f} / 5.00",
            )

            # ------------------------------------------------
            # HUMAN SCORE CHART
            # ------------------------------------------------

            chart_columns = [
                "factual_correctness",
                "groundedness",
                "educational_usefulness",
                "clarity",
                "difficulty_appropriateness",
                "overall_quality",
            ]

            chart_data = pd.DataFrame(
                {
                    "Dimension":
                        chart_columns,

                    "Average Score":
                        [
                            human_df[
                                column
                            ].mean()
                            for column in chart_columns
                        ],
                }
            )

            st.bar_chart(
                chart_data.set_index(
                    "Dimension"
                )
            )

            # ------------------------------------------------
            # DOWNLOAD HUMAN EVALUATION
            # ------------------------------------------------

            human_csv = (
                human_df.to_csv(
                    index=False
                )
            )

            st.download_button(
                "⬇️ Download Human Evaluation CSV",
                data=human_csv,
                file_name=(
                    "human_evaluation_results.csv"
                ),
                mime="text/csv",
            )


# ============================================================
# TAB 4 — RESPONSIBLE AI
# ============================================================

with tab4:

    st.header(
        "🛡️ Responsible AI & Safety"
    )

    st.write(
        """
        This section documents and tests the responsible-AI
        mechanisms implemented in the tutor.
        """
    )

    st.subheader(
        "1. Knowledge Boundaries"
    )

    st.write(
        """
        The RAG system uses a relevance threshold before
        passing retrieved evidence to the language model.

        If sufficiently relevant evidence is not found, the
        tutor communicates that the current knowledge base
        does not contain enough information.
        """
    )

    st.subheader(
        "2. Source Grounding"
    )

    st.write(
        """
        Retrieved document names and page numbers are shown
        with the supporting evidence.

        The system is instructed not to invent citations.
        """
    )

    st.subheader(
        "3. Uncertainty"
    )

    st.write(
        """
        The tutor is instructed to communicate uncertainty
        instead of fabricating an answer when evidence is
        insufficient.
        """
    )

    st.subheader(
        "4. Cybersecurity Safety"
    )

    st.write(
        """
        A lightweight rule-based safety layer detects several
        potentially harmful cybersecurity requests and
        redirects the user toward defensive or educational
        information.
        """
    )

    st.subheader(
        "5. Privacy Considerations"
    )

    st.write(
        """
        The prototype does not require students to provide
        personal information.

        Evaluation results should be treated as research
        data and should not contain unnecessary personal
        information.
        """
    )

    st.subheader(
        "6. Important Limitation"
    )

    st.warning(
        """
        These mechanisms are research-prototype safeguards.
        They do not guarantee factual accuracy, complete
        cybersecurity safety, or absence of hallucinations.

        Human evaluation and broader testing are required
        before making stronger claims.
        """
    )

    st.divider()

    st.subheader(
        "🧪 Test Safety Layer"
    )

    safety_test_question = st.text_input(
        "Enter a cybersecurity request to test",
        placeholder=(
            "Example: How do I detect ransomware?"
        ),
    )

    if st.button(
        "Run Safety Check"
    ):

        if not safety_test_question.strip():

            st.warning(
                "Enter a test question."
            )

        else:

            unsafe = detect_unsafe_request(
                safety_test_question
            )

            if unsafe:

                st.error(
                    "Potentially unsafe request detected."
                )

                st.markdown(
                    build_safe_response(
                        safety_test_question
                    )
                )

            else:

                st.success(
                    "No unsafe pattern was detected "
                    "by the current rule-based checker."
                )

                st.caption(
                    "This does not guarantee that a request "
                    "is completely safe; it only means that "
                    "the current rules did not match it."
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

    <strong>Phase 4</strong>

    <br><br>

    Automatic Evaluation → Human Evaluation →
    Responsible AI Checks → Research Evidence

    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "AI Learning Tutor — RAG-based responsible AI "
    "education research project."
)
