import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

from src.config import config
from src.pdf_processor import PDFProcessor
from src.text_cleaner import TextCleaner
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingEngine
from src.vector_store import FAISSVectorStore
from src.retriever import SpecRetriever
from src.llm_extractor import LLMExtractor
from src.schema import ExtractionResponse

logger = logging.getLogger(__name__)


class SpecExtractionPipeline:
    """
    End-to-end pipeline orchestrating PDF extraction, chunking,
    embedding, FAISS indexing, semantic retrieval, and specification extraction.
    """

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        llm_extractor: Optional[LLMExtractor] = None,
        index_dir: Optional[Path] = None
    ):
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = TextCleaner()
        self.chunker = DocumentChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embedding_engine = embedding_engine or EmbeddingEngine(
            model_name=config.embedding_model_name,
            device=config.device
        )
        self.vector_store = FAISSVectorStore()
        self.retriever = SpecRetriever(
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
            similarity_threshold=config.similarity_threshold
        )
        self.llm_extractor = llm_extractor or LLMExtractor()
        self.current_doc_name: Optional[str] = None
        self.current_index_dir: Optional[Path] = None
        self.is_indexed: bool = False

    @staticmethod
    def _compute_hash(data: Union[str, Path, bytes]) -> str:
        """Computes SHA-256 hash of document content or path."""
        hasher = hashlib.sha256()
        if isinstance(data, (str, Path)):
            with open(data, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        elif isinstance(data, bytes):
            hasher.update(data)
        return hasher.hexdigest()[:16]

    def index_document(
        self,
        source: Union[str, Path, bytes],
        doc_name: str = "document",
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Indexes a service manual PDF into FAISS.
        Reuses cached index if available and force_reindex is False.
        """
        doc_hash = self._compute_hash(source)
        slug = f"{Path(doc_name).stem[:24]}_{doc_hash}"
        target_index_dir = config.index_dir / slug

        self.current_doc_name = doc_name
        self.current_index_dir = target_index_dir

        # 1. Check for existing persisted index
        if not force_reindex and FAISSVectorStore.is_index_ready(target_index_dir):
            if progress_callback:
                progress_callback("Loading existing index from cache...", 0.9)
            loaded = self.vector_store.load(target_index_dir)
            if loaded:
                self.is_indexed = True
                stats = self.vector_store.get_stats()
                if progress_callback:
                    progress_callback("Index ready!", 1.0)
                return {
                    "status": "cached",
                    "doc_name": doc_name,
                    "index_dir": str(target_index_dir),
                    "total_chunks": stats["total_chunks"]
                }

        # 2. Extract pages from PDF
        if progress_callback:
            progress_callback("Extracting text from PDF pages...", 0.1)

        def pdf_progress(current, total):
            if progress_callback and total > 0:
                fraction = 0.1 + 0.3 * (current / total)
                progress_callback(f"Extracting page {current} of {total}...", fraction)

        raw_pages = self.pdf_processor.extract_pages(
            source=source,
            progress_callback=pdf_progress,
            max_pages=max_pages
        )

        # 3. Clean pages
        if progress_callback:
            progress_callback("Cleaning text and formatting tables...", 0.45)
        cleaned_pages = self.text_cleaner.clean_pages(raw_pages)

        # 4. Chunk pages
        if progress_callback:
            progress_callback("Creating overlapping chunks...", 0.55)
        chunks = self.chunker.chunk_pages(cleaned_pages, doc_id=slug)

        if not chunks:
            raise ValueError("No text content could be extracted from this PDF.")

        # 5. Embed chunks
        if progress_callback:
            progress_callback(f"Generating embeddings for {len(chunks)} chunks...", 0.65)

        texts_to_embed = [c["text"] for c in chunks]
        embeddings = self.embedding_engine.embed_texts(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # 6. Build and persist FAISS index
        if progress_callback:
            progress_callback("Building FAISS vector index...", 0.85)

        meta = {
            "doc_name": doc_name,
            "doc_hash": doc_hash,
            "page_count": len(raw_pages),
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        }
        self.vector_store.build_index(
            chunks=chunks,
            embeddings=embeddings,
            save_dir=target_index_dir,
            metadata=meta
        )

        self.is_indexed = True
        if progress_callback:
            progress_callback("Indexing completed successfully!", 1.0)

        return {
            "status": "created",
            "doc_name": doc_name,
            "index_dir": str(target_index_dir),
            "total_pages": len(raw_pages),
            "total_chunks": len(chunks)
        }

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        use_llm: bool = True
    ) -> ExtractionResponse:
        """
        Retrieves relevant passages and extracts structured specifications.
        """
        if not self.is_indexed:
            raise RuntimeError("Pipeline must index a document before querying.")

        evidence_chunks = self.retriever.retrieve(query_text, top_k=top_k)

        response = self.llm_extractor.extract(
            query=query_text,
            evidence_chunks=evidence_chunks,
            use_llm=use_llm,
            document_name=self.current_doc_name
        )

        return response
