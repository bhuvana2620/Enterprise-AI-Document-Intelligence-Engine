# src/ingestion/load_documents.py

import io
from pathlib import Path

from src.ingestion.text_cleaner import clean_text


# ---------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------
def extract_pdf_text_with_pypdf(file_path: str) -> str:
    """
    Extract selectable text from normal text-based PDFs.

    PyPDF2 is imported lazily so TXT/DOCX uploads do not fail if PyPDF2
    is missing in the local environment.
    """
    try:
        from PyPDF2 import PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyPDF2 is required for PDF uploads. Install it with: pip install PyPDF2"
        ) from exc

    reader = PdfReader(file_path)

    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text and page_text.strip():
            text_parts.append(f"\n\n[PAGE {page_number}]\n{page_text}")

    return "\n".join(text_parts)


def extract_pdf_text_with_ocr(file_path: str) -> str:
    """
    OCR fallback for scanned/image-based PDFs such as EAD cards,
    government forms, screenshots, and scanned documents.

    OCR dependencies are imported lazily because they are only needed
    for scanned PDFs.
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image

    except ImportError as exc:
        raise ImportError(
            "OCR dependencies are missing. Install them with:\n"
            "brew install tesseract\n"
            "python -m pip install pymupdf pytesseract pillow"
        ) from exc

    doc = fitz.open(file_path)
    text_parts = []

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)

        # Higher scale gives better OCR accuracy.
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = Image.open(io.BytesIO(pix.tobytes("png")))
        page_text = pytesseract.image_to_string(image)

        if page_text and page_text.strip():
            text_parts.append(f"\n\n[PAGE {page_number + 1}]\n{page_text}")

    return "\n".join(text_parts)


def load_pdf(file_path: str) -> str:
    """
    Load text from PDF.

    Free Render mode:
    - Supports normal text-based PDFs.
    - Does not run OCR fallback because OCR can exceed free-tier memory.
    """
    text = extract_pdf_text_with_pypdf(file_path)

    if len(text.strip()) < 200:
        raise ValueError(
            "This PDF appears to be scanned or image-based. "
            "OCR is disabled on the free deployment because it can exceed memory limits. "
            "Please upload a text-based PDF or convert it to TXT first."
        )

    return text


# ---------------------------------------------------
# TXT Loader
# ---------------------------------------------------
def load_txt(file_path: str) -> str:
    """
    Load plain text files.

    Uses errors='ignore' so weird encodings do not crash the upload pipeline.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


# ---------------------------------------------------
# DOCX Loader
# ---------------------------------------------------
def load_docx(file_path: str) -> str:
    """
    Load text from DOCX.

    python-docx is imported lazily so TXT/PDF uploads do not fail if docx
    is missing in the local environment.
    """
    try:
        from docx import Document
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "python-docx is required for DOCX uploads. Install it with: pip install python-docx"
        ) from exc

    doc = Document(file_path)

    paragraphs = []

    for para in doc.paragraphs:
        if para.text and para.text.strip():
            paragraphs.append(para.text.strip())

    return "\n".join(paragraphs)


# ---------------------------------------------------
# Universal Document Loader
# ---------------------------------------------------
def load_document(file_path: str) -> str:
    """
    Automatically detect file type, extract text, and clean it.

    Supported:
    - .pdf
    - .txt
    - .md
    - .log
    - .docx
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        text = load_pdf(file_path)

    elif ext in {".txt", ".md", ".log"}:
        text = load_txt(file_path)

    elif ext == ".docx":
        text = load_docx(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    cleaned_text = clean_text(text)

    if not cleaned_text:
        print(f"[WARN] No extractable text found in document: {file_path}", flush=True)

    return cleaned_text


# ---------------------------------------------------
# Test Runner
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test document text extraction.")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to PDF, TXT, MD, LOG, or DOCX file."
    )

    args = parser.parse_args()

    text = load_document(args.file)

    print("\n===== EXTRACTED TEXT PREVIEW =====\n")
    print(text[:3000])
    print(f"\nTotal extracted characters: {len(text)}")