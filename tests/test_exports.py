import json
from src.schema import ExtractionResponse, SpecItem
from src.export_utils import ExportUtils


def test_export_utils_to_dataframe_and_csv():
    specs = [
        SpecItem(
            component="Brake Caliper Bolt",
            spec_type="Torque",
            value="35",
            unit="Nm",
            page_number=24,
            source_context="Brake Caliper Bolt: 35 Nm"
        ),
        SpecItem(
            component="Engine Oil",
            spec_type="Fluid Capacity",
            value="6.0",
            unit="qt",
            page_number=45,
            source_context="Engine Oil: 6.0 qt"
        )
    ]
    resp = ExtractionResponse(
        query="Vehicle specs",
        status="success",
        specifications=specs,
        evidence_chunks=[],
        extraction_method="rule_based_fallback"
    )

    df = ExportUtils.to_dataframe(resp)
    assert len(df) == 2
    assert "Component" in df.columns
    assert "Spec Type" in df.columns
    assert df.iloc[0]["Component"] == "Brake Caliper Bolt"

    csv_data = ExportUtils.to_csv_str(resp)
    assert "Brake Caliper Bolt,Torque,35,Nm,24" in csv_data


def test_assignment_format_json():
    spec = SpecItem(
        component="Brake Caliper Bolt",
        spec_type="Torque",
        value="35",
        unit="Nm",
        page_number=24
    )
    resp = ExtractionResponse(
        query="Torque for brake caliper bolts",
        status="success",
        specifications=[spec],
        evidence_chunks=[],
        extraction_method="rule_based_fallback"
    )

    json_str = ExportUtils.to_assignment_format_json(resp)
    parsed = json.loads(json_str)

    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0] == {
        "component": "Brake Caliper Bolt",
        "spec_type": "Torque",
        "value": "35",
        "unit": "Nm"
    }
