from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SpecItem(BaseModel):
    """Structured representation of a single extracted vehicle specification."""
    component: str = Field(..., description="The vehicle part, system, or fastener (e.g., 'Brake Caliper Bolt')")
    spec_type: str = Field(..., description="The type of specification (e.g., 'Torque', 'Fluid Capacity', 'Clearance', 'Part Number')")
    value: str = Field(..., description="The numerical value or range (e.g., '35', '17 - 22', '6.0')")
    unit: str = Field(..., description="Unit of measurement (e.g., 'Nm', 'lb-ft', 'lb-in', 'L', 'qt', 'mm', 'psi')")
    page_number: Optional[int] = Field(None, description="Source page number in the original service manual")
    source_context: Optional[str] = Field(None, description="Exact sentence or line from the document confirming the specification")
    confidence: Optional[float] = Field(None, description="Extraction confidence score between 0.0 and 1.0")
    notes: Optional[str] = Field(None, description="Additional context such as engine variant, model year, or procedure notes")


class EvidenceChunk(BaseModel):
    """Passage retrieved from the document vector store as provenance evidence."""
    chunk_id: str
    page_number: int
    text: str
    similarity_score: float


class ExtractionResponse(BaseModel):
    """Complete response payload returned by the extraction pipeline."""
    query: str
    status: str = Field(..., description="'success', 'partial', or 'not_found'")
    specifications: List[SpecItem] = Field(default_factory=list)
    evidence_chunks: List[EvidenceChunk] = Field(default_factory=list)
    extraction_method: str = Field(..., description="'llm' or 'rule_based_fallback' or 'not_found'")
    message: str = Field(default="")
    document_name: Optional[str] = None
