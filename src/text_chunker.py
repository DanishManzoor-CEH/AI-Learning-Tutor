from typing import List, Dict


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Split page-level documents into smaller overlapping chunks.

    Each chunk keeps its original source and page metadata.
    """

    chunks = []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    for document in documents:

        text = document.get("text", "").strip()

        if not text:
            continue

        source = document.get(
            "source",
            "Unknown source",
        )

        page = document.get(
            "page",
            "Unknown page",
        )

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length,
            )

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "page": page,
                        "chunk_id": len(chunks),
                    }
                )

            if end >= text_length:
                break

            start = end - chunk_overlap

    return chunks
