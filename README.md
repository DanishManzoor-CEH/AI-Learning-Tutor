# 🎓 AI Learning Tutor

## RAG-Based Responsible AI Tutor for Cybersecurity Education

An AI-powered educational assistant designed to support students
in learning cybersecurity concepts through adaptive explanations,
interactive feedback, guided learning, quizzes, and source-grounded
responses.

This project is being developed as a research-oriented portfolio
project exploring the use of Large Language Models (LLMs),
Retrieval-Augmented Generation (RAG), and responsible AI for
educational support.

---

# 🎯 Project Objective

The objective is to develop an AI-supported learning system that
can:

- Explain cybersecurity concepts
- Adapt explanations to different student levels
- Provide practical examples
- Give learning hints
- Generate quizzes
- Provide personalized feedback
- Retrieve information from trusted educational documents
- Provide source-grounded answers
- Measure learning performance
- Reduce hallucinations
- Protect student privacy

---

# 🧠 Planned Architecture

```text
Student
   │
   ▼
Question
   │
   ▼
Intent Detection
   │
   ▼
Educational RAG
   │
   ├── Documents
   ├── Chunking
   ├── Embeddings
   └── Vector Database
   │
   ▼
Large Language Model
   │
   ▼
Educational Response
   │
   ├── Explanation
   ├── Example
   ├── Hint
   ├── References
   └── Follow-up Question
   │
   ▼
Quiz / Learning Check
   │
   ▼
Learning Evaluation
