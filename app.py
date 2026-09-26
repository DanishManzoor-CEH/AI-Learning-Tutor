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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Learning Tutor",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CUSTOM STYLING
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
# GROQ CLIENT
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
        "Please add GROQ_API_KEY to "
        "Streamlit Cloud → Settings → Secrets."
    )

    st.stop()


client = Groq(
    api_key=groq_api_key
)


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

    st.divider()

    st.header(
        "📚 Knowledge Base"
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = load_pdf_documents()


# ============================================================
# DOCUMENT STATISTICS
# ============================================================

statistics = get_document_statistics()


with st.sidebar:

    st.metric(
        "PDF Documents",
        statistics["pdf_count"],
    )

    st.metric(
        "Indexed Pages",
        statistics["page_count"],
    )

    if statistics["documents"]:

        st.caption(
            "Available documents:"
        )

        for document_name in statistics[
            "documents"
        ]:

            st.caption(
                f"📄 {document_name}"
            )

    else:

        st.warning(
            "No PDF documents found."
        )


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

        # Tuple is used so Streamlit can cache
        # the vector store reliably.

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

else:

    st.warning(
        "No usable text was extracted from the PDFs."
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
- Avoid unnecessary jargon.
- Adapt explanations to the student's level.
- Encourage understanding rather than memorization.
- Do not fabricate information.
- Respect the retrieved knowledge-base evidence.
- Do not claim that information came from a source unless
  that source actually supports the statement.
- If the knowledge base does not contain enough information,
  clearly communicate that limitation.
- For cybersecurity topics, remain defensive and educational.
"""


# ============================================================
# GENERATE RAG RESPONSE
# ============================================================

def generate_tutor_response(
    question,
    student_level,
    learning_mode,
    response_length,
    search_results,
):
    """
    Generate a response using retrieved knowledge-base
    content.

    If there are no sufficiently relevant search results,
    the model is explicitly prevented from answering from
    general knowledge.
    """

    # --------------------------------------------------------
    # NO RELEVANT KNOWLEDGE
    # --------------------------------------------------------

    if not search_results:

        return (
            "I couldn't find sufficient information about "
            "this question in the current cybersecurity "
            "learning knowledge base.\n\n"
            "Please ask a question related to the "
            "cybersecurity topics covered by the uploaded "
            "learning materials."
        )

    # --------------------------------------------------------
    # BUILD RAG CONTEXT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

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
            "⚠️ An error occurred while generating "
            "the tutor response.\n\n"
            f"Error: {error}"
        )


# ============================================================
# KNOWLEDGE BASE INFORMATION
# ============================================================

st.subheader(
    "📚 Cybersecurity Knowledge Base"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "PDF Documents",
        statistics["pdf_count"],
    )

with col2:

    st.metric(
        "Pages",
        statistics["page_count"],
    )

with col3:

    st.metric(
        "Text Chunks",
        len(chunks),
    )


if not documents:

    st.info(
        """
        No cybersecurity PDF documents are currently
        available.

        Add trusted cybersecurity learning material to:

        `data/documents/`
        """
    )


# ============================================================
# ASK AI TUTOR
# ============================================================

st.divider()

st.subheader(
    "🤖 Ask Your AI Tutor"
)

question = st.text_area(
    "Enter your cybersecurity question",
    placeholder=(
        "Example: What is DNS and what role does it "
        "play in computer networks?"
    ),
    height=120,
)


ask_button = st.button(
    "🚀 Ask AI Tutor",
    type="primary",
)


# ============================================================
# ASK BUTTON
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    elif vector_index is None:

        st.warning(
            "The cybersecurity knowledge base "
            "is not available yet."
        )

    else:

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching the cybersecurity knowledge base..."
        ):

            search_results = search_vector_store(
                vector_index,
                chunks,
                question,
                top_k=4,
            )

        # ----------------------------------------------------
        # NO RELEVANT RESULTS
        # ----------------------------------------------------

        if not search_results:

            st.warning(
                """
                I couldn't find sufficient information
                about this question in the current
                cybersecurity knowledge base.
                """
            )

            st.caption(
                "The tutor does not answer from general "
                "knowledge when the knowledge base does "
                "not contain sufficiently relevant evidence."
            )

        # ----------------------------------------------------
        # RELEVANT RESULTS FOUND
        # ----------------------------------------------------

        else:

            with st.spinner(
                "🧠 Generating a grounded tutor response..."
            ):

                answer = generate_tutor_response(
                    question=question,
                    student_level=student_level,
                    learning_mode=learning_mode,
                    response_length=response_length,
                    search_results=search_results,
                )

            # ------------------------------------------------
            # RESPONSE
            # ------------------------------------------------

            st.subheader(
                "🤖 Tutor Response"
            )

            st.markdown(
                answer
            )

            # ------------------------------------------------
            # RETRIEVED SOURCES
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📚 Retrieved Evidence"
            )

            st.caption(
                "These are the knowledge-base passages "
                "retrieved for this question."
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
                        f"**Document:** "
                        f"{result['source']}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{result['page']}"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{result['similarity']:.3f}"
                    )

                    st.write(
                        result["text"]
                    )


# ============================================================
# RESEARCH INFORMATION
# ============================================================

st.divider()

st.subheader(
    "🔬 Research Project"
)

st.markdown(
    """
    <div class="research-box">

    <strong>Research Question:</strong><br>

    How can Retrieval-Augmented Generation improve the
    factual accuracy, source-groundedness, and
    learning-support quality of an LLM-based cybersecurity
    tutoring system?

    <br><br>

    <strong>Current Architecture:</strong>

    <br><br>

    Student Question → Embedding → FAISS Retrieval →
    Relevant Cybersecurity Chunks → GPT-OSS →
    Grounded Tutor Response

    </div>
    """,
    unsafe_allow_html=True,
)


st.caption(
    "AI Learning Tutor — RAG-based responsible AI "
    "education research project."
)
