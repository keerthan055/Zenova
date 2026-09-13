"""API routes for Curated Evidence Retrieval-Augmented Generation (RAG)."""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from zenova.schemas.rag import RAGQueryResult, KnowledgeSource
from zenova.rag.engine import KnowledgeRetrievalEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.rag")

router = APIRouter(prefix="/api/v1/rag", tags=["Retrieval-Augmented Generation"])

# Shared retrieval engine instance
_engine: Optional[KnowledgeRetrievalEngine] = None


def get_rag_engine() -> KnowledgeRetrievalEngine:
    global _engine
    if _engine is None:
        _engine = KnowledgeRetrievalEngine(knowledge_dir="knowledge")
    return _engine


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text to search authoritative knowledge base")
    top_k: int = Field(default=3, ge=1, le=10, description="Maximum number of chunks to retrieve")
    threshold_override: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Optional custom confidence threshold override"
    )


@router.post("/query", response_model=RAGQueryResult)
async def query_knowledge(req: RAGQueryRequest) -> RAGQueryResult:
    """Retrieve evidence-grounded psychoeducational passages with strict anti-hallucination confidence gating."""
    try:
        engine = get_rag_engine()
        result = engine.query(
            text=req.query,
            top_k=req.top_k,
            threshold_override=req.threshold_override
        )
        return result
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")


@router.get("/sources")
async def list_sources() -> Dict[str, Any]:
    """List all curated, clinically approved mental health knowledge sources with provenance metadata."""
    manifest_path = Path("knowledge/metadata/sources_manifest.json")
    if not manifest_path.exists():
        # Fallback: scan sources folder
        sources_dir = Path("knowledge/sources")
        if not sources_dir.exists():
            return {"count": 0, "sources": []}
        sources = []
        for p in sources_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    sources.append(json.load(f))
            except Exception as read_err:
                logger.warning(f"Error reading source file {p}: {read_err}")
        return {"count": len(sources), "sources": sources}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["count"] = data.get("total_sources", len(data.get("sources", [])))
        return data
    except Exception as e:
        logger.error(f"Failed to read sources manifest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load sources manifest: {str(e)}")


@router.get("/status")
async def get_rag_status() -> Dict[str, Any]:
    """Get indexing status, chunk counts, and model metadata for the RAG engine."""
    engine = get_rag_engine()
    manifest_path = Path("knowledge/metadata/sources_manifest.json")
    source_count = 0
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                source_count = data.get("total_sources", data.get("count", 0))
        except Exception:
            pass

    return {
        "is_initialized": engine.is_initialized,
        "indexed_chunks": engine.vector_store.count(),
        "registered_sources": source_count,
        "embedding_dimension": engine.embedding_model.dimension,
        "default_threshold": engine.min_confidence_threshold,
        "vector_store_type": type(engine.vector_store).__name__,
        "embedding_model_type": type(engine.embedding_model).__name__
    }
