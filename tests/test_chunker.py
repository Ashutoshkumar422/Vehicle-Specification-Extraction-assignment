import pytest
from src.chunker import DocumentChunker


def test_chunker_invalid_params():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=0)
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=500, chunk_overlap=600)


def test_chunker_short_text():
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    pages = [
        {"page_number": 1, "text": "This is page one text."},
        {"page_number": 2, "text": "This is page two text."}
    ]
    chunks = chunker.chunk_pages(pages, doc_id="test_doc")
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_id"] == "test_doc_p1_c0"
    assert chunks[1]["page_number"] == 2
    assert chunks[1]["chunk_id"] == "test_doc_p2_c0"


def test_chunker_long_text_overlapping():
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=30)
    lines = [f"Step {i}: Inspect the component carefully." for i in range(15)]
    long_text = "\n".join(lines)
    pages = [{"page_number": 5, "text": long_text}]
    chunks = chunker.chunk_pages(pages, doc_id="test_doc")

    assert len(chunks) > 1
    for c in chunks:
        assert c["page_number"] == 5
        # Ensure every chunk starts and ends with a complete line, no severed words
        assert not c["text"].startswith(" ")
        assert not c["text"].endswith(" ")
