import streamlit as st
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    """
    Load and cache the Sentence Transformer embedding model.

    The model is loaded once and reused across Streamlit reruns.
    """

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    return model


def create_embeddings(
    texts,
):
    """
    Convert text into normalized embedding vectors.
    """

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings
