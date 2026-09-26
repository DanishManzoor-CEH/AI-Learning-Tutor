from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st

from src.embeddings import create_embeddings


# Initial threshold for Phase 2.
#
# Because embeddings are normalized and FAISS uses
# inner-product similarity, the score is approximately
# cosine similarity.
#
# This value can later be calibrated using the evaluation set.
DEFAULT_SIMILARITY_THRESHOLD = 0.45


@st.cache_resource
def build_vector_store(
    chunks: Tuple[Dict, ...],
):
    """
    Build and cache a FAISS vector index.

    The index is cached so Streamlit does not rebuild
    the embeddings every time the user asks a question.
    """

    chunks_list = list(chunks)

    if not chunks_list:

        raise ValueError(
            "No document chunks were provided."
        )

    texts = [
        chunk["text"]
        for chunk in chunks_list
    ]

    embeddings = create_embeddings(
        texts
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def search_vector_store(
    index,
    chunks: List[Dict],
    query: str,
    top_k: int = 4,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """
    Search the FAISS vector store.

    Only results whose similarity score is equal to or
    greater than the threshold are returned.

    This prevents unrelated questions from being sent
    to the LLM with irrelevant context.
    """

    if index is None:
        return []

    if not chunks:
        return []

    if not query.strip():
        return []

    query_embedding = create_embeddings(
        [query]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    number_of_results = min(
        top_k,
        len(chunks),
    )

    scores, indices = index.search(
        query_embedding,
        number_of_results,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        similarity = float(
            score
        )

        # Reject weak/unrelated matches.
        if similarity < similarity_threshold:
            continue

        chunk = chunks[
            index_position
        ].copy()

        chunk["similarity"] = similarity

        results.append(
            chunk
        )

    return results
