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
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
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

        .feature-card {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #dddddd;
            margin-bottom: 12px;
        }

        .source-box {
            padding: 12px;
            border-radius: 8px;
            background-color: #f5f5f5;
            margin-top: 10px;
        }

        .disclaimer {
            font-size: 13px;
            color: #777777;
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
# INTRODUCTION
# ============================================================

st.info(
    """
    **Welcome to the AI Learning Tutor!**

    This project is being developed as a research-oriented
    educational AI system. Its goal is to help students learn
    cybersecurity concepts through explanations, examples,
    hints, questions, and personalized learning support.

    🚧 Phase 1: LLM Foundation
    """
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.header("🎓 Learning Settings")

    st.divider()

    student_level = st.selectbox(
        "Student Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
        ],
    )

    learning_mode = st.selectbox(
        "Learning Mode",
        [
            "Explain",
            "Simple Explanation",
            "Technical Explanation",
            "Socratic Learning",
        ],
    )
    response_length = st.selectbox(
    "Response Length",
    ["Short", "Medium", "Detailed"],
    index=1
)



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

    st.sidebar.success("Knowledge base available")

    with st.sidebar.expander("View documents"):

        for document in stats["documents"]:
            st.write(f"• {document}")

else:
st.sidebar.warning(
        "No PDF documents found."
    )

documents = load_pdf_documents()


    
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
st.markdown("### 🔬 Project Status")
st.markdown(
        """
        **Phase 1**
        
        ✅ Streamlit UI  
        ✅ Groq LLM  
        ✅ Adaptive learning level  
        ✅ Learning modes  
        
        **Coming Next**
        
        ⏳ PDF knowledge base  
        ⏳ RAG  
        ⏳ FAISS  
        ⏳ Source citations  
        ⏳ Quiz generation  
        ⏳ Learning analytics  
        ⏳ Evaluation
        """
    )


# ============================================================
# GROQ API CONFIGURATION
# ============================================================

try:

    groq_api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        """
        🔑 **Groq API key not configured.**

        Please add your Groq API key to:

        **Streamlit Cloud → App Settings → Secrets**

        Use:

        `GROQ_API_KEY="your_api_key_here"`
        """
    )

    st.stop()


# ============================================================
# INITIALIZE GROQ CLIENT
# ============================================================

client = Groq(api_key=groq_api_key)


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(
    student_level,
    learning_mode,
    response_length,
):

    length_instruction = {
        "Short": "Keep the response concise and focused.",
        "Medium": "Provide a balanced explanation with useful examples.",
        "Detailed": "Provide a detailed educational explanation with examples and important technical details.",
    }

    mode_instruction = {
        "Explain": """
        Explain the concept clearly and logically.
        Include a practical cybersecurity example when appropriate.
        """,

        "Simple Explanation": """
        Explain the concept using simple language.
        Use an analogy or real-world example when helpful.
        Avoid unnecessary technical complexity.
        """,

        "Technical Explanation": """
        Provide a technically detailed explanation.
        Include relevant protocols, mechanisms, architecture,
        security implications, and practical examples.
        """,

        "Socratic Learning": """
        Do not immediately provide the complete answer.
        Guide the student through the concept using questions,
        hints, and reasoning.
        Encourage the student to think about the problem.
        """
    }

    return f"""
You are an AI Learning Tutor specializing in cybersecurity.

Your role is to support learning rather than simply provide
answers.

STUDENT LEVEL:
{student_level}

LEARNING MODE:
{learning_mode}

RESPONSE LENGTH:
{response_length}

EDUCATIONAL PRINCIPLES:

1. Explain concepts accurately.
2. Adapt explanations to the student's level.
3. Encourage understanding rather than memorization.
4. Use practical cybersecurity examples.
5. Break difficult concepts into smaller steps.
6. Ask a useful follow-up question when appropriate.
7. Clearly distinguish facts from examples or assumptions.
8. Never invent references or sources.
9. If you are uncertain, explicitly acknowledge uncertainty.
10. Do not claim that information comes from a document because
    no knowledge documents have been connected yet.
11. Avoid unnecessarily revealing sensitive information.
12. For cybersecurity topics, remain within legitimate educational
    and defensive contexts.

LEARNING MODE INSTRUCTIONS:

{mode_instruction[learning_mode]}

RESPONSE LENGTH INSTRUCTIONS:

{length_instruction[response_length]}

When appropriate, structure the answer as:

### Explanation

### Example

### Key Points

### Think About This

The final section should encourage active learning.
"""


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("💬 Ask Your AI Tutor")

question = st.text_area(
    "What would you like to learn?",
    placeholder=(
        "Example:\n"
        "Explain the TCP three-way handshake and why it is important."
    ),
    height=150,
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🚀 Ask Tutor",
    type="primary",
    use_container_width=True,
)


# ============================================================
# GENERATE RESPONSE
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a cybersecurity question first."
        )

    else:

        system_prompt = build_system_prompt(
            student_level=student_level,
            learning_mode=learning_mode,
            response_length=response_length,
        )

        try:

            with st.spinner(
                "🤖 Your AI tutor is thinking..."
            ):

                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": question,
                        },
                    ],

                    temperature=0.3,
                    max_tokens=2000,
                )

                answer = completion.choices[0].message.content

            st.success("Tutor response generated.")

            st.markdown("## 📚 Tutor Response")

            st.markdown(answer)

        except Exception as error:

            st.error(
                "An error occurred while generating the response."
            )

            with st.expander("Technical error details"):

                st.code(str(error))


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("🔬 About This Project")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="feature-card">

        ### 🧠 Adaptive Learning

        Responses are adapted to the student's
        selected learning level.

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">

        ### 🤖 LLM-Based Tutor

        Uses a large language model to provide
        interactive educational support.

        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">

        ### 🔬 Research-Oriented

        Future versions will include RAG,
        evaluation, learning analytics,
        and responsible AI experiments.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="disclaimer">

    🎓 AI Learning Tutor — Phase 1

    This system is an educational prototype. AI-generated
    responses may contain errors and should be verified
    against reliable learning resources.

    </div>
    """,
    unsafe_allow_html=True,
)
