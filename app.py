import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add parent directory to path so imports work cleanly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.config import config
from src.pipeline import SpecExtractionPipeline
from src.export_utils import ExportUtils
from src.embeddings import EmbeddingEngine
from src.llm_extractor import LLMExtractor

st.set_page_config(
    page_title="Vehicle Specification Extraction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .evidence-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-family: monospace;
        font-size: 0.88rem;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Initializes and caches the extraction pipeline and embedding model."""
    embedding_engine = EmbeddingEngine(
        model_name=config.embedding_model_name,
        device=config.device
    )
    llm_extractor = LLMExtractor()
    return SpecExtractionPipeline(
        embedding_engine=embedding_engine,
        llm_extractor=llm_extractor
    )


pipeline = get_pipeline()

# Session State Initialization
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "index_stats" not in st.session_state:
    st.session_state.index_stats = {}


# Auto-load existing cached index if available and pipeline is not yet indexed
if not pipeline.is_indexed:
    cached_indexes = sorted(
        [p.parent for p in config.index_dir.glob("*/index.faiss") if p.exists()],
        key=lambda p: (p / "index.faiss").stat().st_mtime,
        reverse=True
    )
    if cached_indexes:
        target_dir = cached_indexes[0]
        if pipeline.vector_store.load(target_dir):
            pipeline.is_indexed = True
            st.session_state.indexed = True
            doc_name = "sample-service-manual 1.pdf"
            meta_path = target_dir / "meta.json"
            if meta_path.exists():
                try:
                    import json
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        doc_name = meta.get("doc_name", doc_name)
                except Exception:
                    pass
            st.session_state.current_doc_name = doc_name
            st.session_state.index_stats = {
                "status": "cached",
                "doc_name": doc_name,
                "total_chunks": len(pipeline.vector_store.chunks)
            }

# --- SIDEBAR: Document Management & Settings ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car-service.png", width=64)
    st.title("Settings & Manual")

    # Document Selection Source
    doc_source = st.radio(
        "Document Source",
        ["Select Existing Workspace PDF", "Upload PDF"],
        index=0
    )

    pdf_source = None
    doc_display_name = ""

    # Look for existing PDFs in workspace or project data folder
    workspace_candidates = [
        Path("../sample-service-manual 1.pdf"),
        Path("sample-service-manual 1.pdf"),
        Path("data/sample-service-manual 1.pdf"),
        Path(__file__).resolve().parent.parent / "sample-service-manual 1.pdf",
        Path(__file__).resolve().parent / "data" / "sample-service-manual 1.pdf"
    ]
    existing_pdfs = [p for p in workspace_candidates if p.exists()]

    if doc_source == "Select Existing Workspace PDF":
        if existing_pdfs:
            selected_pdf = st.selectbox(
                "Detected Service Manuals",
                options=existing_pdfs,
                format_func=lambda p: f"{p.name} ({p.stat().st_size / (1024*1024):.1f} MB)"
            )
            pdf_source = str(selected_pdf)
            doc_display_name = selected_pdf.name
        else:
            st.info("No service manual PDF found in local directory. Please upload one below.")
            doc_source = "Upload PDF"

    if doc_source == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload Service Manual PDF",
            type=["pdf"],
            help="Upload an automotive service manual PDF (e.g., Ford F-150 Workshop Manual)"
        )
        if uploaded_file is not None:
            pdf_source = uploaded_file.getvalue()
            doc_display_name = uploaded_file.name

    # Indexing Controls
    max_pages_option = st.number_input(
        "Max Pages to Index (0 for All Pages)",
        min_value=0,
        max_value=2000,
        value=0,
        help="Select a page limit. Set 0 (default) to index all 852 pages of the manual."
    )
    max_pages = None if max_pages_option == 0 else max_pages_option

    force_reindex = st.checkbox("Force Re-index (ignore cache)", value=False)

    if st.button("🔨 Build / Load Index", type="primary", use_container_width=True):
        if pdf_source is None:
            st.error("Please select or upload a valid PDF file first.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(msg, frac):
                status_text.text(msg)
                progress_bar.progress(min(max(frac, 0.0), 1.0))

            try:
                with st.spinner("Processing document..."):
                    result = pipeline.index_document(
                        source=pdf_source,
                        doc_name=doc_display_name,
                        force_reindex=force_reindex,
                        progress_callback=update_progress,
                        max_pages=max_pages
                    )
                st.session_state.indexed = True
                st.session_state.current_doc_name = doc_display_name
                st.session_state.index_stats = result
                st.success(f"Index ready! ({result.get('total_chunks', 0)} chunks)")
            except Exception as e:
                st.error(f"Error indexing document: {e}")

    # Index Status Card
    st.markdown("---")
    st.subheader("📊 Index Status")
    if pipeline.is_indexed:
        st.markdown(f"**Document:** `{st.session_state.current_doc_name or 'sample-service-manual 1.pdf'}`")
        stats = pipeline.vector_store.get_stats()
        st.markdown(f"**Total Chunks in Index:** `{stats.get('total_chunks', 0)}`")
        st.markdown(f"**Vector Dimension:** `{stats.get('dimension', 384)}`")
        st.markdown('<span class="badge-success">Ready for Queries</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warning">No Document Indexed</span>', unsafe_allow_html=True)

    # Retrieval and LLM Settings
    st.markdown("---")
    st.subheader("⚙️ Query Settings")
    top_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=15, value=config.default_top_k)

    use_llm = st.toggle("Enable LLM Extraction", value=pipeline.llm_extractor.has_llm_credentials)
    llm_key = st.text_input(
        "OpenAI API Key (Optional)",
        value=pipeline.llm_extractor.api_key or "",
        type="password",
        help="Optional. If omitted or invalid, deterministic rule-based extraction fallback is used automatically."
    )
    if llm_key:
        pipeline.llm_extractor.api_key = llm_key.strip()


# --- MAIN AREA ---
st.markdown('<div class="main-title">🚗 Vehicle Specification Extraction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Semantic RAG system for extracting torque values, fluid capacities, clearances, and part numbers with source provenance.</div>',
    unsafe_allow_html=True
)

# Suggested queries
st.markdown("**Sample Specification Queries:**")
sample_cols = st.columns(3)
sample_queries = [
    "Torque for brake disc shield bolts",
    "Lower ball joint nut torque",
    "Tie-rod end nut torque",
    "Rear axle lubricant fluid capacity",
    "Part number for shock absorber lower nut",
    "Submarine ballast tank pressure specification"
]

selected_sample = None
for i, sq in enumerate(sample_queries):
    col = sample_cols[i % 3]
    if col.button(sq, key=f"sq_{i}", use_container_width=True):
        selected_sample = sq

# Query Input
query_text = st.text_input(
    "Enter specification query:",
    value=selected_sample if selected_sample else "",
    placeholder="e.g., Torque for brake disc shield bolts, rear axle fluid capacity, shock absorber part number..."
)

col_search, col_clear = st.columns([1, 6])
execute_search = col_search.button("🔍 Search", type="primary", use_container_width=True)

# Trigger query execution if search button clicked or sample query clicked
active_query = selected_sample if selected_sample else (query_text.strip() if execute_search else "")

if active_query:
    if not pipeline.is_indexed:
        st.warning("⚠️ Please build or load an index from the sidebar before searching.")
    else:
        with st.spinner("Retrieving evidence passages and extracting specifications..."):
            try:
                response = pipeline.query(
                    query_text=active_query,
                    top_k=top_k,
                    use_llm=use_llm
                )
                st.session_state.last_response = response
            except Exception as e:
                st.error(f"Extraction failed: {e}")

# Display Results
response = st.session_state.last_response

if response is not None:
    st.markdown("---")
    
    # Status Banner
    if response.status == "success":
        st.success(f"**Status: Success** | {response.message} (Method: `{response.extraction_method}`)")
    else:
        st.warning(f"**Status: Not Found** | {response.message} (Method: `{response.extraction_method}`)")

    # Specifications Table
    if response.specifications:
        st.subheader("📋 Extracted Specifications")
        df = ExportUtils.to_dataframe(response)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download Buttons
        st.markdown("#### 💾 Export Specifications")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        csv_data = ExportUtils.to_csv_str(response)
        dl_col1.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"vehicle_specs_{Path(response.document_name or 'doc').stem}.csv",
            mime="text/csv",
            use_container_width=True
        )

        json_assignment = ExportUtils.to_assignment_format_json(response)
        dl_col2.download_button(
            label="📥 Download Assignment JSON",
            data=json_assignment,
            file_name=f"specs_assignment_format.json",
            mime="application/json",
            use_container_width=True
        )

        json_full = ExportUtils.to_json_str(response)
        dl_col3.download_button(
            label="📥 Download Full Provenance JSON",
            data=json_full,
            file_name=f"specs_full_provenance.json",
            mime="application/json",
            use_container_width=True
        )

    # Evidence Chunks Section
    st.subheader(f"🔍 Retrieved Evidence Passages ({len(response.evidence_chunks)} chunks)")
    if response.evidence_chunks:
        for idx, chunk in enumerate(response.evidence_chunks, 1):
            with st.expander(f"Passage #{idx} | Page {chunk.page_number} | Score: {chunk.similarity_score:.4f} | Chunk ID: {chunk.chunk_id}"):
                st.code(chunk.text, language="text")
    else:
        st.info("No evidence chunks retrieved.")

    # Raw JSON Debug Viewer
    with st.expander("🛠️ View Raw JSON Response"):
        st.json(response.model_dump())

else:
    if not pipeline.is_indexed:
        st.info("👈 Please select or upload a service manual PDF in the sidebar and click **Build / Load Index** to begin.")
    else:
        st.info("💡 Type a vehicle specification question above or click one of the sample query buttons.")
