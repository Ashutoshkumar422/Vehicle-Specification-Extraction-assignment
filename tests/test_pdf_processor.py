import pytest
import pymupdf
from src.pdf_processor import PDFProcessor


@pytest.fixture
def sample_pdf_bytes():
    """Generates a small 2-page in-memory PDF for unit testing."""
    doc = pymupdf.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Brake System Specifications\nBrake Caliper Bolt: 35 Nm")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Engine Fluid Capacities\nEngine Oil: 6.0 qt (5.7 L)")
    
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_processor_extract_pages(sample_pdf_bytes):
    processor = PDFProcessor()
    pages = processor.extract_pages(sample_pdf_bytes)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Brake Caliper Bolt: 35 Nm" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Engine Oil: 6.0 qt" in pages[1]["text"]


def test_pdf_processor_metadata(sample_pdf_bytes):
    processor = PDFProcessor()
    info = processor.get_document_info(sample_pdf_bytes)

    assert info["page_count"] == 2
    assert info["is_encrypted"] is False
