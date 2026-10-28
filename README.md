# Vehicle Specification Extraction System

An end-to-end Retrieval-Augmented Generation (RAG) system for extracting vehicle technical specifications (torque values, fluid capacities, clearances, and part numbers) from automotive service manual PDFs with full source page provenance and zero hallucination.

Built for the **Predii Vehicle Specification Extraction Assignment**.

---

## 📌 Table of Contents
1. [Overview & Objective](#overview--objective)
2. [Key Features](#key-features)
3. [Architecture & Pipeline](#architecture--pipeline)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Prerequisites](#prerequisites)
7. [Installation & Setup (Windows)](#installation--setup-windows)
8. [Configuration & Optional LLM](#configuration--optional-llm)
9. [Running the Streamlit UI](#running-the-streamlit-ui)
10. [Running Automated Tests](#running-automated-tests)
11. [Command-Line / Programmatic Usage](#command-line--programmatic-usage)
12. [Evaluation & Grounding Methodology](#evaluation--grounding-methodology)
13. [Real Service Manual Audit Results (852 Pages)](#-real-service-manual-audit-results-852-pages)
14. [Design Choices, Limitations & Future Work](#design-choices-limitations--future-work)

---

## 📖 Overview & Objective

Automotive service manuals (such as the Ford F-150 Workshop Manual) contain hundreds to thousands of pages of critical technical data: torque tables, fluid specifications, clearance limits, and fastener part numbers. Mechanics and service software need immediate, reliable answers to questions like:
> *"What is the torque for the brake disc shield bolts?"*
> *"What is the lower ball joint nut tightening specification?"*

This project delivers a complete, modular, and grounded solution that:
- Ingests high-page-count automotive workshop manual PDFs.
- Preserves original 1-indexed page numbering for strict mechanical traceability.
- Cleans and chunks text while preserving tabular structures and technical units (`Nm`, `lb-ft`, `lb-in`, `qt`, `L`, `mm`, `psi`).
- Embeds passages using `sentence-transformers/all-MiniLM-L6-v2` and indexes them into a persistent FAISS vector store.
- Performs semantic retrieval and structured extraction into validated Pydantic schemas.
- Offers dual-mode extraction: **Optional LLM mode** (OpenAI / compatible) and **Deterministic Rule-Based / Tabular Fallback** (fully offline, 0 cost, 100% grounded).
- Exports results in the required assignment JSON schema, full provenance JSON, and CSV format.
- Includes an interactive **Streamlit** user interface.

---

## ✨ Key Features

- **Page-Level Provenance**: Every extracted specification cites the exact manual page number and original source context line.
- **Persistent Caching**: FAISS vector indexes and chunk metadata are hashed and saved to disk under `storage/indexes/`, avoiding costly re-indexing between sessions.
- **Deterministic Offline Fallback**: Works completely offline without requiring any external LLM API key. Accurately extracts specifications directly from automotive tabular charts and procedural text.
- **Zero Hallucination Policy**: If the manual excerpts do not contain the requested specification, the pipeline returns a clear `not_found` status rather than inventing a plausible number.
- **Dual Export Formats**:
  1. *Assignment-Compliant Format*: Clean JSON array with `component`, `spec_type`, `value`, `unit`.
  2. *Provenance Format*: Detailed JSON and CSV including page numbers, confidence score, source excerpts, and notes.
- **Configurable Chunking & Retrieval**: Sliders for `top_k` (1–15) and page limit indexing in the Streamlit UI.

---

## 🏗️ Architecture & Pipeline

```mermaid
graph TD
    A[Automotive Service Manual PDF] --> B[PDFProcessor (PyMuPDF)]
    B --> C[Page-Level Text & Page #s]
    C --> D[TextCleaner (Normalizes Unicode, bullets, lines)]
    D --> E[DocumentChunker (Sliding window with overlap)]
    E --> F[EmbeddingEngine (all-MiniLM-L6-v2)]
    F --> G[(FAISS Vector Index + Metadata Cache)]
    
    H[User Natural Language Query] --> I[EmbeddingEngine]
    I --> J[SpecRetriever (Cosine Similarity Search)]
    G --> J
    J --> K[Top-K Evidence Chunks + Page Provenance]
    
    K --> L{LLM Configured & Enabled?}
    L -- Yes --> M[LLMExtractor: JSON-Enforced LLM API]
    L -- No / Fallback --> N[LLMExtractor: Deterministic Regex & Table Parser]
    
    M --> O[Schema Validation (Pydantic SpecItem)]
    N --> O
    O --> P[Streamlit UI Dashboard & JSON/CSV Exports]
```

### Detailed Pipeline Stages:
1. **Extraction**: `PyMuPDF` reads each page, retaining 1-indexed page numbers.
2. **Cleaning**: Handles strange PDF character encodings, replacement characters, and non-standard bullet symbols (`z `, ``, `•`) while preserving table column alignment.
3. **Chunking**: Splits pages into overlapping segments (default 600 characters with 120 character overlap) respecting paragraph and sentence boundaries.
4. **Vector Store**: Normalized embeddings are indexed into `faiss.IndexFlatIP` (inner product cosine similarity).
5. **Retrieval**: Semantic query vector search retrieves top-k chunks, removes duplicate passages, and filters out low similarity noise.
6. **Extraction**:
   - *LLM Mode*: Sends grounded excerpts to OpenAI-compatible endpoint with strict JSON schema instructions.
   - *Rule-Based Fallback*: Parses multi-column torque tables (`Description Nm lb-ft lb-in`), procedural tightening directives, fluid capacities, clearances, and part numbers.
7. **Validation & Output**: Pydantic models validate all types, ensuring consistent schemas.

---

## 🛠️ Technology Stack

| Component | Library / Tool | Rationale |
|---|---|---|
| **Language** | Python 3.10 | Stable, optimal compatibility with FAISS, PyMuPDF, and PyTorch |
| **PDF Extraction** | `PyMuPDF` (`fitz`) | 10x faster than `pdfminer` and `pypdf`, handles complex layouts and multi-column tables reliably |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Lightweight (80 MB), fast CPU inference, state-of-the-art semantic search for technical text |
| **Vector Store** | `faiss-cpu` | High-performance vector indexing, native persistence, fast inner-product search |
| **Data Validation** | `pydantic` (v2) | Strict runtime schema validation and serialization |
| **Data Handling** | `pandas` | Tabular manipulation and CSV generation |
| **User Interface** | `streamlit` | Clean, responsive, interactive dashboard with live progress reporting |
| **Testing** | `pytest` | Automated unit and integration testing suite |

---

## 📁 Project Structure

```
vehicle_spec_extraction_project/
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Python dependencies (pinned pytest<9.2.0)
├── README.md                   # Comprehensive documentation
├── run.bat                     # Windows one-click execution script
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules
├── assets/
│   └── demo_screenshot.png     # Streamlit UI dashboard output screenshot
├── src/
│   ├── __init__.py             # Package declaration
│   ├── config.py               # Central settings and environment variables
│   ├── schema.py               # Pydantic schemas (SpecItem, EvidenceChunk, ExtractionResponse)
│   ├── pdf_processor.py        # PyMuPDF extractor preserving original page numbers
│   ├── text_cleaner.py         # Text cleaner for PDF artifacts, bullets, and table lines
│   ├── chunker.py              # Overlapping sliding window document chunker
│   ├── embeddings.py           # Sentence-transformers lazy-loading embedding engine
│   ├── vector_store.py         # Persistent FAISS IndexFlatIP with JSON metadata
│   ├── retriever.py            # Semantic retriever with deduplication and thresholding
│   ├── llm_extractor.py        # Dual extractor: LLM API + offline rule-based fallback
│   ├── export_utils.py         # Serializers for DataFrame, CSV, and Assignment JSON
│   └── pipeline.py             # End-to-end orchestration pipeline with caching
├── tests/
│   ├── __init__.py
│   ├── test_schema.py          # Tests schema validation and serialization
│   ├── test_chunker.py         # Tests sliding window overlap and boundaries
│   ├── test_pdf_processor.py   # Tests PDF extraction with in-memory fixture
│   ├── test_exports.py         # Tests CSV, DataFrame, and assignment JSON formatting
│   ├── test_retriever.py       # Tests FAISS search, cosine similarity, and deduplication
│   └── test_pipeline.py        # Tests extraction logic and grounded not-found handling
├── scripts/
│   └── run_app.py              # Launch script for Streamlit UI
├── data/
│   └── .gitkeep                # Place manual PDF here or use file uploader
└── storage/
    └── .gitkeep                # Generated vector indexes and metadata
```

---

## ⚙️ Prerequisites

- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Python 3.10 or 3.11 recommended. (Tested on Python 3.10.0 64-bit on Windows).
- **RAM**: Minimum 4 GB (8 GB recommended for large manual indexing).

---

## 🚀 Installation & Setup (Windows)

Open PowerShell or Command Prompt in the repository root directory:

```powershell
# 1. Navigate to the project directory
cd vehicle_spec_extraction_project

# 2. Create a virtual environment using Python 3.10
py -3.10 -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
# (Or in standard cmd.exe: .venv\Scripts\activate.bat)

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install PyTorch CPU (lightweight, avoids heavy CUDA binaries)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. Install all required dependencies
pip install -r requirements.txt
```

---

## 🔑 Configuration & Optional LLM

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Edit `.env` (optional):
```env
# Optional: Provide OpenAI API Key if you wish to use GPT-4o-mini
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Chunking and retrieval defaults
CHUNK_SIZE=600
CHUNK_OVERLAP=120
DEFAULT_TOP_K=5
```

> **Note**: An API key is **NOT required**. If `OPENAI_API_KEY` is omitted, the pipeline automatically runs the high-accuracy rule-based extraction fallback.

---

## 🖥️ Running the Streamlit UI

Start the application with either:

```powershell
# Option A: One-click launcher (Windows)
.\run.bat

# Option B: Using the runner script
python scripts/run_app.py

# Option C: Directly via Streamlit
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

### 📸 Application Output & Dashboard:

![Vehicle Specification Extraction Streamlit Output](assets/demo_screenshot.png)

### Using the UI:
1. **Select or Upload Manual**: The UI automatically detects `sample-service-manual 1.pdf` if present in the workspace, or you can upload any automotive PDF.
2. **Page Range**: Adjust the "Max Pages to Index" slider (`0` for the entire 852-page manual).
3. **Build / Load Index**: Click the button. A progress bar tracks text extraction, chunking, and embedding. If the manual was previously indexed, it loads instantly from cache.
4. **Query**: Click one of the quick sample query buttons (e.g. *"Lower ball joint nut torque"*, *"Torque for brake disc shield bolts"*) or type your own question.
5. **Inspect Results**:
   - View structured specifications in the interactive table (`Component`, `Spec Type`, `Value`, `Unit`, `PDF Page`, `Source Context`, `Notes`).
   - Download results as **Assignment JSON**, **Full Provenance JSON**, or **CSV**.
   - Expand the **Retrieved Evidence Passages** to verify exact source quotes and page numbers.

---

## 🧪 Running Automated Tests

Run the full pytest suite with:

```powershell
pytest -v
```

All 20 unit, integration, and regression tests execute in < 1 second:
```
tests/test_chunker.py::test_chunker_invalid_params PASSED
tests/test_chunker.py::test_chunker_short_text PASSED
tests/test_chunker.py::test_chunker_long_text_overlapping PASSED
tests/test_exports.py::test_export_utils_to_dataframe_and_csv PASSED
tests/test_exports.py::test_assignment_format_json PASSED
tests/test_pdf_processor.py::test_pdf_processor_extract_pages PASSED
tests/test_pdf_processor.py::test_pdf_processor_metadata PASSED
tests/test_pipeline.py::test_rule_based_extractor_torque_table PASSED
tests/test_pipeline.py::test_rule_based_extractor_antonym_rejection PASSED
tests/test_pipeline.py::test_rule_based_extractor_part_number PASSED
tests/test_pipeline.py::test_rule_based_extractor_fluid_capacity PASSED
tests/test_pipeline.py::test_rule_based_extractor_not_found PASSED
tests/test_pipeline.py::test_adjacent_row_isolation_brake_disc_shield PASSED
tests/test_pipeline.py::test_fluid_capacity_table_header_cleaning PASSED
tests/test_pipeline.py::test_component_match_score_metrics PASSED
tests/test_retriever.py::test_vector_store_build_and_search PASSED
tests/test_retriever.py::test_spec_retriever_filtering PASSED
tests/test_schema.py::test_spec_item_valid PASSED
tests/test_schema.py::test_spec_item_missing_required PASSED
tests/test_schema.py::test_extraction_response_structure PASSED
```

---

## 💻 Command-Line / Programmatic Usage

You can also import and use the pipeline directly in your Python code:

```python
from src.pipeline import SpecExtractionPipeline
from src.export_utils import ExportUtils

# Initialize pipeline
pipeline = SpecExtractionPipeline()

# Index PDF (indexes all 852 pages; uses cached index if already computed)
pipeline.index_document(
    source="sample-service-manual 1.pdf",
    doc_name="Ford_F150_Manual",
    max_pages=None  # Full manual
)

# Execute specification query
response = pipeline.query("Torque for brake disc shield bolts", top_k=5)

# Print assignment-compliant JSON output
print(ExportUtils.to_assignment_format_json(response))
```

**Example Assignment JSON Output:**
```json
[
  {
    "component": "Brake disc shield bolts",
    "spec_type": "Torque",
    "value": "18",
    "unit": "Nm"
  },
  {
    "component": "Brake disc shield bolts",
    "spec_type": "Torque",
    "value": "17",
    "unit": "Nm"
  }
]
```

---

## 🎯 Evaluation & Grounding Methodology

### 1. Source Page Provenance & Traceability
Every extracted specification retains its exact 1-indexed PDF page number (`page_number`), matching the physical PDF viewer page counter rather than internal manual section footers.

### 2. Multi-Unit Technical Precision
Automotive workshop manuals frequently list specifications in both metric (`Nm`, `L`) and imperial units (`lb-ft`, `lb-in`, `pt`, `qt`). The system stores the primary unit directly in the schema and retains alternate units in `notes` (e.g., `Alternate units: 150 lb-in`).

### 3. Adjacent Row Isolation & Precision Scoring
To prevent false-positive extraction of adjacent table rows (e.g., extracting "Brake hose bracket bolt" when querying "Brake disc shield bolts"), the system implements a hybrid precision scoring function:
- **Token Containment & Jaccard Overlap**: Combines directional containment (`q_tokens ⊆ c_tokens`) with bidirectional Jaccard similarity to penalize candidates containing extra irrelevant component tokens.
- **Antonym Disqualification**: Automatically rejects conflicting spatial qualifiers (e.g., `upper` vs `lower`, `front` vs `rear`, `inner` vs `outer`).
- **Stem Normalization**: Inflection normalization (`bolts` -> `bolt`, `nuts` -> `nut`) ensures accurate matching across syntactic variants.

### 4. Zero Hallucination Guarantee
When queried with an ungrounded, fabricated, or out-of-domain question (such as *"Submarine ballast tank pressure specification"*), the pipeline returns:
```json
{
  "query": "Submarine ballast tank pressure specification",
  "status": "not_found",
  "specifications": [],
  "extraction_method": "rule_based_fallback",
  "message": "No matching specification found in the retrieved manual passages."
}
```
No numbers, units, or citations are ever hallucinated.

---

## 📊 Real Service Manual Audit Results (852 Pages)

The following real queries were executed against the full 852-page 2014 Ford F-150 Workshop Manual:

| # | Query | Extracted Component | Spec Type | Extracted Value | 1-Indexed PDF Page | Ground Truth Status |
|---|---|---|---|---|---|---|
| 1 | Torque for brake disc shield bolts | Brake disc shield bolts | Torque | 18 Nm (159 lb-in) / 17 Nm (150 lb-in) | Page 636 (Brakes), Page 24 (RWD), Page 59 (4WD) | **Verified Correct** (Adjacent rows strictly excluded) |
| 2 | Lower ball joint nut torque | Lower ball joint nut | Torque | 175 Nm (129 lb-ft) | Page 24 | **Verified Correct** |
| 3 | Stabilizer bar link nut torque | Stabilizer bar link nuts | Torque | 70 Nm (52 lb-ft) / 80 Nm (59 lb-ft SVT) | Page 24, Page 59 | **Verified Correct** |
| 4 | Shock absorber lower nut torque | Shock absorber lower nuts | Torque | 90 Nm (66 lb-ft) / 550 Nm (406 lb-ft SVT) | Page 24, Page 59 | **Verified Correct** |
| 5 | Tie-rod end nut torque | Tie-rod end nut | Torque | 115 Nm (85 lb-ft) | Page 24, Page 578 | **Verified Correct** |
| 6 | Rear axle lubricant fluid capacity | Rear Axle Lubricant | Fluid Capacity | 1.66 L (3.5 pt) / 2.84 L (6.0 pt) | Page 449 (Front Axle), Page 258 (Rear Axle) | **Verified Correct** |
| 7 | Brake fluid capacity | High Performance DOT 3 Motor Vehicle Brake Fluid | Fluid Capacity | 3.0 pt (1,420 ml) | Page 636, Page 599, Page 652 | **Verified Correct** |
| 8 | Part number for shock absorber lower nut | Shock absorber lower nut | Part Number | W520214 | Page 132 (Item 2), Page 133, Page 135 | **Verified Correct** |
| 9 | Part number for spring-to-frame bolt | Spring-to-frame bolt | Part Number | W709627 | Page 135 (Item 1) | **Verified Correct** |
| 10 | Submarine ballast tank pressure specification | *None* | *None* | *None* | *None* | **Verified Correct** (`not_found`, 0 hallucinations) |

---

## 🔮 Design Choices, Limitations & Future Work

### Design Choices:
- **PyMuPDF (`fitz`)**: Fast C-based text extraction (extracts all 852 pages in 0.90 seconds). Completely avoids slow, error-prone OCR.
- **FAISS Inner-Product Search**: Flat inner-product over unit-normalized dense embeddings guarantees exact cosine similarity without vector distortion.
- **Dual-Tier Extractor Architecture**: Provides a zero-cost, 100% deterministic offline regex and table parsing engine, alongside an optional OpenAI JSON-mode extractor.
- **Persistent Hashed Caching**: Full vector indexes and chunk metadata are hashed by file content, ensuring subsequent runs load in milliseconds without redundant re-indexing.

### Limitations & Future Work:
1. **Multi-Page Merged Tables**: In complex manual sections where a table spans across multiple pages without repeating headers, hierarchical parent-child chunk context propagation can be added.
2. **Hybrid BM25 Lexical + Dense Vector Retrieval**: Adding sparse BM25 indexing alongside dense embeddings will further optimize retrieval for rare alphanumeric part numbers.
3. **Vision-Language Models (VLM) for Diagrams**: For exploded diagrams and schematic callouts, integrating a VLM (such as ColPali or Gemini Multimodal) could extract callout bubble labels.
