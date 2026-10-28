import pytest
from src.schema import EvidenceChunk
from src.llm_extractor import LLMExtractor


def test_rule_based_extractor_torque_table():
    extractor = LLMExtractor()
    chunk = EvidenceChunk(
        chunk_id="c1",
        page_number=24,
        text=(
            "Torque Specifications\n"
            "Description\n"
            "Nm lb-ft lb-in\n"
            "Brake disc shield bolts\n"
            "17\n"
            "-\n"
            "150\n"
            "Brake hose bracket bolt\n"
            "12\n"
            "-\n"
            "106\n"
            "Lower ball joint nut\n"
            "175 129\n"
            "-"
        ),
        similarity_score=0.92
    )

    resp = extractor.extract(
        query="torque for brake disc shield bolts",
        evidence_chunks=[chunk],
        use_llm=False
    )

    assert resp.status == "success"
    # Must ONLY return Brake disc shield bolts, NOT Brake hose bracket bolt or Lower ball joint nut!
    assert len(resp.specifications) == 1
    spec = resp.specifications[0]
    assert spec.component == "Brake disc shield bolts"
    assert spec.value == "17"
    assert spec.unit == "Nm"
    assert spec.page_number == 24
    assert "150 lb-in" in (spec.notes or "")


def test_rule_based_extractor_antonym_rejection():
    extractor = LLMExtractor()
    chunk = EvidenceChunk(
        chunk_id="c1",
        page_number=24,
        text=(
            "Torque Specifications\n"
            "Description\n"
            "Nm lb-ft lb-in\n"
            "Upper ball joint nut\n"
            "115 85\n"
            "-\n"
            "Lower ball joint nut\n"
            "175 129\n"
            "-"
        ),
        similarity_score=0.90
    )

    resp = extractor.extract(
        query="lower ball joint nut torque",
        evidence_chunks=[chunk],
        use_llm=False
    )

    assert resp.status == "success"
    assert len(resp.specifications) == 1
    spec = resp.specifications[0]
    assert spec.component == "Lower ball joint nut"
    assert spec.value == "175"
    assert spec.unit == "Nm"
    assert "129 lb-ft" in (spec.notes or "")


def test_rule_based_extractor_part_number():
    extractor = LLMExtractor()
    chunk = EvidenceChunk(
        chunk_id="c132",
        page_number=132,
        text=(
            "Item\n"
            "Part Number\n"
            "Description\n"
            "1\n"
            "W506545\n"
            "Shock absorber lower bolt\n"
            "2\n"
            "W520214\n"
            "Shock absorber lower nut\n"
            "3\n"
            "18125\n"
            "Shock absorber"
        ),
        similarity_score=0.91
    )

    resp = extractor.extract(
        query="part number for shock absorber lower nut",
        evidence_chunks=[chunk],
        use_llm=False
    )

    assert resp.status == "success"
    assert len(resp.specifications) == 1
    spec = resp.specifications[0]
    assert spec.component == "Shock absorber lower nut"
    assert spec.spec_type == "Part Number"
    assert spec.value == "W520214"
    assert spec.page_number == 132


def test_rule_based_extractor_fluid_capacity():
    extractor = LLMExtractor()
    chunk = EvidenceChunk(
        chunk_id="c308",
        page_number=308,
        text="Remove the filler plug, fill the axle with 2.8L (6.0 pt) of axle lubricant and install the filler plug.",
        similarity_score=0.88
    )

    resp = extractor.extract(
        query="rear axle lubricant fluid capacity",
        evidence_chunks=[chunk],
        use_llm=False
    )

    assert resp.status == "success"
    assert len(resp.specifications) == 1
    spec = resp.specifications[0]
    assert "Axle" in spec.component
    assert spec.spec_type == "Fluid Capacity"
    assert spec.value == "2.8"
    assert spec.unit == "L"
    assert "6.0 pt" in (spec.notes or "")


def test_rule_based_extractor_not_found():
    extractor = LLMExtractor()
    chunk = EvidenceChunk(
        chunk_id="c2",
        page_number=5,
        text="Check vehicle alignment and tire wear patterns on regular roads.",
        similarity_score=0.6
    )

    resp = extractor.extract(
        query="submarine ballast tank pressure specification",
        evidence_chunks=[chunk],
        use_llm=False
    )

    assert resp.status == "not_found"
    assert len(resp.specifications) == 0
    assert "No matching specification found" in resp.message


def test_adjacent_row_isolation_brake_disc_shield():
    """Explicitly verifies adjacent row rejection for Brake disc shield bolts vs Brake hose bracket bolt."""
    extractor = LLMExtractor()
    table_text = (
        "Description\n"
        "Nm lb-ft lb-in\n"
        "Brake disc shield bolts\n"
        "17\n"
        "—\n"
        "150\n"
        "Brake hose bracket bolt\n"
        "12\n"
        "—\n"
        "106\n"
        "Lower arm forward and rearward nuts\n"
        "350 258\n"
        "—"
    )
    chunk = EvidenceChunk(
        chunk_id="c24",
        page_number=24,
        text=table_text,
        similarity_score=0.95
    )

    resp = extractor.extract(
        query="What is the torque for brake disc shield bolts?",
        evidence_chunks=[chunk],
        use_llm=False
    )

    assert resp.status == "success"
    assert len(resp.specifications) == 1
    item = resp.specifications[0]
    assert item.component == "Brake disc shield bolts"
    assert item.value == "17"
    assert item.unit == "Nm"
    assert item.page_number == 24
    # Adjacent rows MUST NOT be present
    comps = [s.component for s in resp.specifications]
    assert "Brake hose bracket bolt" not in comps
    assert "Lower arm forward and rearward nuts" not in comps


def test_fluid_capacity_table_header_cleaning():
    """Verifies that table headers (e.g., 'Fill Capacity') do not contaminate component names."""
    extractor = LLMExtractor()
    table_text = (
        "Material Item Specification Fill Capacity\n"
        "High Performance DOT 3 Motor Vehicle Brake Fluid PM-1-C WSS-M6C62-A 1,420 ml (3 pt)"
    )
    chunk = EvidenceChunk(
        chunk_id="c636",
        page_number=636,
        text=table_text,
        similarity_score=0.90
    )

    resp = extractor.extract(
        query="Brake fluid capacity",
        evidence_chunks=[chunk],
        use_llm=False
    )

    assert resp.status == "success"
    assert len(resp.specifications) >= 1
    spec = resp.specifications[0]
    assert not spec.component.lower().startswith("pacity")
    assert not spec.component.lower().startswith("fill")
    assert "Brake Fluid" in spec.component
    assert spec.spec_type == "Fluid Capacity"
    assert spec.page_number == 636


def test_component_match_score_metrics():
    """Tests accuracy, singular/plural stem matching, antonym rejection, and score thresholds."""
    # Exact match
    assert LLMExtractor.compute_component_match_score("brake disc shield bolts", "Brake disc shield bolts") == 1.0

    # Singular query vs plural component
    score_stem = LLMExtractor.compute_component_match_score("brake disc shield bolt", "Brake disc shield bolts")
    assert score_stem >= 0.95

    # Neighboring row: Brake hose bracket bolt vs Brake disc shield bolts
    score_neighbor = LLMExtractor.compute_component_match_score("brake disc shield bolts", "Brake hose bracket bolt")
    assert score_neighbor < 0.50

    # Antonym rejection
    score_antonym = LLMExtractor.compute_component_match_score("lower ball joint", "Upper ball joint")
    assert score_antonym == 0.0

    # Unrelated query
    score_unrelated = LLMExtractor.compute_component_match_score("submarine ballast tank", "Brake disc shield bolts")
    assert score_unrelated == 0.0
