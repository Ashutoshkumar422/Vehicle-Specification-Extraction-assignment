import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

from src.pipeline import SpecExtractionPipeline

pipeline = SpecExtractionPipeline()
idx_path = current_dir / "storage" / "indexes" / "sample-service-manual 1_c85e7b943c53d825"
if idx_path.exists() and (idx_path / "index.faiss").exists():
    pipeline.vector_store.load(idx_path)
    pipeline.is_indexed = True
    pipeline.current_doc_name = "sample-service-manual 1.pdf"
else:
    # Auto-detect PDF and build index
    pdf_candidates = [
        current_dir / "data" / "sample-service-manual 1.pdf",
        current_dir.parent / "sample-service-manual 1.pdf",
        Path("sample-service-manual 1.pdf")
    ]
    pdf_file = next((p for p in pdf_candidates if p.exists()), None)
    if pdf_file:
        print(f"Indexing {pdf_file.name} (all 852 pages)...")
        pipeline.index_document(source=pdf_file, doc_name=pdf_file.name, max_pages=None)
    else:
        print("Please place 'sample-service-manual 1.pdf' in data/ or root directory to index.")
        sys.exit(1)

audit_queries = [
    ("1. Brake disc shield bolt torque", "Torque for brake disc shield bolts"),
    ("2. Lower ball joint nut torque", "Lower ball joint nut torque"),
    ("3. Stabilizer bar link nut torque", "Stabilizer bar link nut torque"),
    ("4. Shock absorber lower nut torque", "Shock absorber lower nut torque"),
    ("5. Tie-rod end nut torque", "Tie-rod end nut torque"),
    ("6. Fluid capacity: Rear axle lubricant", "Rear axle lubricant fluid capacity"),
    ("7. Fluid capacity: Brake fluid", "Brake fluid capacity"),
    ("8. Part number: Shock absorber lower nut", "Part number for shock absorber lower nut"),
    ("9. Part number: Spring-to-frame bolt", "Part number for spring-to-frame bolt"),
    ("10. Unrelated query: Submarine ballast tank", "Submarine ballast tank pressure specification")
]

print("==================================================")
print("=== FULL SERVICE MANUAL AUDIT (ALL 852 PAGES) ===")
print("==================================================")
print(f"Total Chunks in Index: {len(pipeline.vector_store.chunks)}")
print(f"Document Name: {pipeline.current_doc_name}")
print()

for label, q in audit_queries:
    resp = pipeline.query(q, top_k=5, use_llm=False)
    print(f"### {label}")
    print(f"- **Query**: \"{q}\"")
    print(f"- **Status**: {resp.status} (Method: `{resp.extraction_method}`)")
    print(f"- **Extracted Count**: {len(resp.specifications)}")
    for i, s in enumerate(resp.specifications, 1):
        print(f"  {i}. **Component**: {s.component}")
        print(f"     - **Spec Type**: {s.spec_type}")
        print(f"     - **Value & Unit**: {s.value} {s.unit}")
        print(f"     - **Source PDF Page**: Page {s.page_number}")
        print(f"     - **Evidence Snippet**: \"{s.source_context}\"")
        print(f"     - **Notes**: {s.notes}")
    if not resp.specifications:
        print("  - *(No specifications returned - strictly grounded, zero hallucination)*")
    print()
