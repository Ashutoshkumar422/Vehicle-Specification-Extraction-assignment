import io
from pathlib import Path
from typing import List, Dict, Any, Union, Callable, Optional
import pymupdf


class PDFProcessor:
    """Extracts text content and page metadata from automotive service manual PDFs."""

    def __init__(self):
        pass

    def extract_pages(
        self,
        source: Union[str, Path, bytes, io.BytesIO],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts text from each page of the PDF.
        
        Args:
            source: File path (str/Path) or raw PDF bytes.
            progress_callback: Optional callable(current_page, total_pages).
            max_pages: Optional maximum number of pages to process.

        Returns:
            List of dicts: [{'page_number': 1, 'text': '...', 'char_count': 123}]
        """
        doc = None
        try:
            if isinstance(source, (str, Path)):
                doc = pymupdf.open(str(source))
            elif isinstance(source, bytes):
                doc = pymupdf.open(stream=source, filetype="pdf")
            elif isinstance(source, io.BytesIO):
                doc = pymupdf.open(stream=source.getvalue(), filetype="pdf")
            else:
                raise ValueError("Unsupported source type for PDFProcessor")

            total_pages = len(doc)
            pages_to_read = total_pages if max_pages is None else min(total_pages, max_pages)
            extracted_pages = []

            for page_idx in range(pages_to_read):
                page_num = page_idx + 1  # 1-indexed page numbering
                page = doc[page_idx]
                text = page.get_text("text") or ""
                
                extracted_pages.append({
                    "page_number": page_num,
                    "text": text,
                    "char_count": len(text)
                })

                if progress_callback:
                    progress_callback(page_num, pages_to_read)

            return extracted_pages

        finally:
            if doc is not None:
                doc.close()

    def get_document_info(self, source: Union[str, Path, bytes, io.BytesIO]) -> Dict[str, Any]:
        """Extracts basic metadata from the PDF without reading all pages."""
        doc = None
        try:
            if isinstance(source, (str, Path)):
                doc = pymupdf.open(str(source))
            elif isinstance(source, bytes):
                doc = pymupdf.open(stream=source, filetype="pdf")
            elif isinstance(source, io.BytesIO):
                doc = pymupdf.open(stream=source.getvalue(), filetype="pdf")
            else:
                raise ValueError("Unsupported source type")

            return {
                "page_count": len(doc),
                "metadata": doc.metadata,
                "is_encrypted": doc.is_encrypted
            }
        finally:
            if doc is not None:
                doc.close()
