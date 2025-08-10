import fitz  # PyMuPDF


def pdf_to_pages(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text().strip())
    return pages
