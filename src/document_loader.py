from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def load_pdf_documents(documents_dir: str = "data/documents") -> List[Dict]:
    """
    Load PDF files from the documents directory.

    Each PDF page becomes a separate document record.

    Returns:
        List of dictionaries containing:
        - text
        - source
        - page
    """

    documents = []

    documents_path = Path(documents_dir)

    if not documents_path.exists():
        return documents

    pdf_files = sorted(documents_path.glob("*.pdf"))

    for pdf_path in pdf_files:

        try:
            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):

                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                text = text.strip()

                if not text:
                    continue

                documents.append(
                    {
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number,
                    }
                )

        except Exception as error:
            print(
                f"Could not read PDF '{pdf_path.name}': {error}"
            )

    return documents


def get_document_statistics(
    documents_dir: str = "data/documents"
) -> Dict:
    """
    Return basic statistics about the PDF knowledge base.
    """

    documents_path = Path(documents_dir)

    if not documents_path.exists():
        return {
            "pdf_count": 0,
            "page_count": 0,
            "documents": [],
        }

    pdf_files = sorted(documents_path.glob("*.pdf"))

    documents = load_pdf_documents(documents_dir)

    return {
        "pdf_count": len(pdf_files),
        "page_count": len(documents),
        "documents": [pdf.name for pdf in pdf_files],
    }
