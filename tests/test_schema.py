import pytest
from pydantic import ValidationError
from src.schema import SpecItem, EvidenceChunk, ExtractionResponse


def test_spec_item_valid():
    item = SpecItem(
        component="Brake Caliper Bolt",
        spec_type="Torque",
        value="35",
        unit="Nm",
        page_number=24,
        source_context="Brake Caliper Bolt 35 Nm",
        confidence=0.95
    )
    assert item.component == "Brake Caliper Bolt"
    assert item.spec_type == "Torque"
    assert item.value == "35"
    assert item.unit == "Nm"
    assert item.page_number == 24
    assert item.confidence == 0.95


def test_spec_item_missing_required():
    with pytest.raises(ValidationError):
        SpecItem(component="Brake Caliper Bolt")


def test_extraction_response_structure():
    chunk = EvidenceChunk(
        chunk_id="doc_p24_c0",
        page_number=24,
        text="Brake Caliper Bolt 35 Nm",
        similarity_score=0.88
    )
    spec = SpecItem(
        component="Brake Caliper Bolt",
        spec_type="Torque",
        value="35",
        unit="Nm",
        page_number=24
    )
    resp = ExtractionResponse(
        query="Torque for brake caliper bolt",
        status="success",
        specifications=[spec],
        evidence_chunks=[chunk],
        extraction_method="rule_based_fallback",
        message="Found 1 spec"
    )

    data = resp.model_dump()
    assert data["status"] == "success"
    assert len(data["specifications"]) == 1
    assert data["specifications"][0]["unit"] == "Nm"
    assert len(data["evidence_chunks"]) == 1
