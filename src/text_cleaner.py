import re
from typing import List, Dict, Any


class TextCleaner:
    """Cleans and standardizes extracted service manual text while preserving specs and tables."""

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Cleans a single text string by normalizing unicode, formatting bullets,
        removing excessive whitespace, and retaining numbers and units.
        """
        if not raw_text:
            return ""

        # Normalize carriage returns and non-breaking spaces
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\xa0", " ").replace("\u200b", "")

        # Standardize non-standard bullet symbols commonly found in workshop PDFs (e.g., 'z ', '')
        text = re.sub(r'^[ \t]*[z•\-\*\u2022\uf0b7]\s*', '• ', text, flags=re.MULTILINE)
        
        # Replace unicode replacement characters
        text = text.replace("\ufffd", "-")

        # Collapse horizontal whitespace (spaces/tabs) while preserving newlines
        text = re.sub(r'[ \t]+', ' ', text)

        # Normalize excessive consecutive newlines (keep max 2 for paragraph separation)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading and trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        cleaned = "\n".join(lines).strip()

        return cleaned

    @classmethod
    def clean_pages(cls, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cleans a list of page dicts produced by PDFProcessor.
        Preserves page_number and metadata.
        """
        cleaned_pages = []
        for p in pages:
            cleaned_text = cls.clean_text(p.get("text", ""))
            cleaned_pages.append({
                "page_number": p.get("page_number"),
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "is_empty": len(cleaned_text.strip()) == 0
            })
        return cleaned_pages
