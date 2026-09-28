from typing import List, Dict, Tuple

import faiss
import numpy as np

from src.embeddings import create_embeddings


def build_vector_store(
    chunks: List[Dict],
) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Build a FAISS vector index from document chunks.

    Returns:
        vector_index:
            FAISS similarity-search index.

        chunk_metadata:
            The original chunk dictionaries containing
            text, source, page, and chunk_id.
    """

    if not chunks:
        raise ValueError(
            "No document chunks were provided."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Create normalized embeddings
    embeddings = create_embeddings(
        texts
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional array."
        )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of document chunks."
        )

    dimension = embeddings.shape[1]

    # Inner Product works as cosine similarity
    # because embeddings are normalized.
    vector_index = faiss.IndexFlatIP(
        dimension
    )

    vector_index.add(
        embeddings
    )

    # IMPORTANT:
    # Return BOTH the FAISS index and the
    # corresponding chunk metadata.
    return vector_index, chunks


def search_vector_store(
    index,
    chunks: List[Dict],
    query: str,
    top_k: int = 4,
) -> List[Dict]:
    """
    Search the FAISS vector index and return
    the most relevant document chunks.
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

        if index_position >= len(chunks):
            continue

        chunk = chunks[
            index_position
        ].copy()

        chunk["similarity"] = float(
            score
        )

        results.append(
            chunk
        )

    return results
