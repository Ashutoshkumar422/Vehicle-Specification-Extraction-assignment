import json
import io
from typing import List, Dict, Any, Union
import pandas as pd
from src.schema import ExtractionResponse, SpecItem


class ExportUtils:
    """Utilities for serializing extraction responses to JSON, CSV, and Pandas DataFrames."""

    @staticmethod
    def to_dataframe(response: Union[ExtractionResponse, List[SpecItem]]) -> pd.DataFrame:
        """Converts extracted specifications into a Pandas DataFrame for display or tabular analysis."""
        if isinstance(response, ExtractionResponse):
            items = response.specifications
        else:
            items = response

        if not items:
            return pd.DataFrame(columns=[
                "Component", "Spec Type", "Value", "Unit", "PDF Page", "Source Context", "Notes"
            ])

        records = []
        for s in items:
            records.append({
                "Component": s.component,
                "Spec Type": s.spec_type,
                "Value": s.value,
                "Unit": s.unit,
                "PDF Page": s.page_number if s.page_number is not None else "N/A",
                "Source Context": s.source_context or "",
                "Notes": s.notes or ""
            })
        return pd.DataFrame(records)

    @staticmethod
    def to_json_str(response: ExtractionResponse, indent: int = 2) -> str:
        """Serializes full ExtractionResponse to a clean, formatted JSON string."""
        return response.model_dump_json(indent=indent)

    @staticmethod
    def to_csv_str(response: Union[ExtractionResponse, List[SpecItem]]) -> str:
        """Converts extracted specifications to CSV string format."""
        df = ExportUtils.to_dataframe(response)
        return df.to_csv(index=False)

    @staticmethod
    def to_assignment_format_json(response: ExtractionResponse, indent: int = 2) -> str:
        """
        Formats specifications strictly according to the assignment prompt snippet:
        [
          {
            "component": "Brake Caliper Bolt",
            "spec_type": "Torque",
            "value": "35",
            "unit": "Nm"
          }
        ]
        """
        items = [
            {
                "component": s.component,
                "spec_type": s.spec_type,
                "value": s.value,
                "unit": s.unit
            }
            for s in response.specifications
        ]
        return json.dumps(items, indent=indent, ensure_ascii=False)
