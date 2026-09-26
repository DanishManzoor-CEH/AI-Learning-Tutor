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
        FAISS index
        Chunk metadata
    """

    if not chunks:
        raise ValueError(
            "No document chunks were provided."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
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

    index.add(embeddings)

    return index, chunks


def search_vector_store(
    index,
    chunks: List[Dict],
    query: str,
    top_k: int = 4,
) -> List[Dict]:
    """
    Search the FAISS index for the most relevant chunks.
    """

    if index is None:
        return []

    if not chunks:
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

        chunk = chunks[index_position].copy()

        chunk["similarity"] = float(
            score
        )

        results.append(chunk)

    return results
