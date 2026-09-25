import streamlit as st
from groq import Groq

from src.document_loader import (
    load_pdf_documents,
    get_document_statistics,
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
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        background-color: #f5f7fa;
        margin-top: 10px;
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
    '<div class="subtitle">'
    "A responsible AI tutor for cybersecurity learning"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# GROQ CLIENT
# ============================================================

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Please add it to Streamlit Cloud Secrets."
    )
    st.stop()


client = Groq(api_key=groq_api_key)


# ============================================================
# SIDEBAR - LEARNING SETTINGS
# ============================================================

with st.sidebar:

    st.header("⚙️ Learning Settings")

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
# SIDEBAR - KNOWLEDGE BASE
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("📚 Knowledge Base")

stats = get_document_statistics()

st.sidebar.write(
    f"📄 PDFs: {stats['pdf_count']}"
)

st.sidebar.write(
    f"📑 Pages loaded: {stats['page_count']}"
)

if stats["documents"]:

    st.sidebar.success(
        "Knowledge base available"
    )

    with st.sidebar.expander(
        "View documents"
    ):

        for document in stats["documents"]:

            st.write(
                f"• {document}"
            )

else:

    st.sidebar.warning(
        "No PDF documents found."
    )


# ============================================================
# LOAD KNOWLEDGE BASE DOCUMENTS
# ============================================================

documents = load_pdf_documents()


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an AI Learning Tutor specializing in cybersecurity education.

Your purpose is to help students understand cybersecurity concepts clearly,
accurately, and responsibly.

Student level:
{student_level}

Learning mode:
{learning_mode}

Response length:
{response_length}

Teaching principles:

1. Explain concepts accurately.
2. Adapt explanations to the student's level.
3. Use simple examples when appropriate.
4. Explain technical terminology.
5. Encourage understanding rather than memorization.
6. Ask a short follow-up question when useful.
7. Never intentionally provide false information.
8. If you are uncertain about a factual claim, clearly indicate uncertainty.
9. For cybersecurity topics, keep explanations educational and defensive.
10. Do not claim that information came from a source unless it actually did.
"""


# ============================================================
# RESPONSE LENGTH INSTRUCTIONS
# ============================================================

if response_length == "Short":

    length_instruction = """
Keep the answer concise.
Use a short explanation and one simple example when useful.
"""

elif response_length == "Detailed":

    length_instruction = """
Provide a detailed explanation.
Use headings, examples, technical details, and practical context where useful.
"""

else:

    length_instruction = """
Provide a balanced explanation.
Explain the main concept clearly and include a useful example.
"""


# ============================================================
# LEARNING MODE INSTRUCTIONS
# ============================================================

if learning_mode == "Simple Explanation":

    mode_instruction = """
Use very simple language.
Explain difficult technical terms in plain language.
Assume the student is still developing their fundamentals.
"""

elif learning_mode == "Technical Explanation":

    mode_instruction = """
Provide a technically detailed explanation.
Include relevant protocols, mechanisms, components, and security considerations.
"""

elif learning_mode == "Socratic Learning":

    mode_instruction = """
Teach using a Socratic approach.
Instead of immediately giving every detail, guide the student with questions
and hints that encourage them to reason about the concept.
"""

else:

    mode_instruction = """
Explain the concept directly and clearly.
Use examples when they improve understanding.
"""


# ============================================================
# AI RESPONSE FUNCTION
# ============================================================

def generate_tutor_response(
    question,
    student_level,
    learning_mode,
    response_length,
):
    """
    Generate an educational response using Groq.
    """

    prompt = f"""
Student question:

{question}

Student level:
{student_level}

Learning mode:
{learning_mode}

Response length:
{response_length}

{length_instruction}

{mode_instruction}

Please answer the student's question as an educational cybersecurity tutor.

When appropriate, structure the answer using:

- Explanation
- Key Points
- Example
- Security Relevance
- Quick Check

Do not unnecessarily include every section if it does not help answer the question.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
            temperature=0.3,
            max_tokens=1500,
        )

        return response.choices[0].message.content

    except Exception as error:

        return (
            "⚠️ An error occurred while generating the response.\n\n"
            f"Error: {error}"
        )


# ============================================================
# MAIN KNOWLEDGE BASE INFORMATION
# ============================================================

st.divider()

st.subheader("📚 Cybersecurity Knowledge Base")

if documents:

    st.write(
        f"The tutor currently has access to "
        f"**{len(documents)} PDF pages**."
    )

    with st.expander(
        "🔎 View loaded knowledge sources"
    ):

        for document in documents:

            source = document.get(
                "source",
                "Unknown source",
            )

            page = document.get(
                "page",
                "Unknown page",
            )

            st.write(
                f"📄 **{source}** — Page {page}"
            )

else:

    st.info(
        "No PDF documents are currently available. "
        "Add cybersecurity PDFs to `data/documents/`."
    )


# ============================================================
# ASK YOUR AI TUTOR
# ============================================================

st.divider()

st.subheader("💬 Ask Your AI Tutor")

question = st.text_area(
    "What would you like to learn?",
    placeholder=(
        "For example: What is cybersecurity? "
        "Explain the TCP three-way handshake."
    ),
    height=120,
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🚀 Ask AI Tutor",
    type="primary",
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "🧠 Your AI Tutor is preparing an answer..."
        ):

            answer = generate_tutor_response(
                question=question,
                student_level=student_level,
                learning_mode=learning_mode,
                response_length=response_length,
            )

        st.subheader("🤖 Tutor Response")

        st.markdown(answer)


# ============================================================
# RESEARCH INFORMATION
# ============================================================

st.divider()

with st.expander(
    "🔬 About this Research Project"
):

    st.write(
        """
        This project explores how Large Language Models and
        Retrieval-Augmented Generation can support cybersecurity education.

        The research direction focuses on:

        • Educational AI
        • Large Language Models
        • Retrieval-Augmented Generation (RAG)
        • Source-grounded answers
        • Responsible AI
        • Hallucination reduction
        • Adaptive learning
        • Student feedback
        • Cybersecurity education
        """
    )

    st.markdown(
        """
        **Planned research question:**

        How can Retrieval-Augmented Generation improve the factual
        accuracy, source-groundedness, and learning-support quality
        of an LLM-based cybersecurity tutoring system?
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Learning Tutor | RAG-Based Responsible AI Tutor "
    "for Cybersecurity Learning"
)
