"""Unit tests for ZENOVA Curated Evidence RAG Module."""
import json
import pytest
import numpy as np
from pathlib import Path

from zenova.schemas.rag import (
    SourceType,
    EvidenceTier,
    ReviewStatus,
    ReliabilityMetadata,
    KnowledgeSource,
    KnowledgeChunk,
    RAGSearchResult,
    RAGQueryResult
)
from zenova.rag.chunker import DocumentChunker
from zenova.rag.embeddings import TfidfDenseEmbedding, get_embedding_model
from zenova.rag.vectorstore import InMemoryVectorStore, get_vector_store
from zenova.rag.reranker import HybridReranker
from zenova.rag.engine import KnowledgeRetrievalEngine
from zenova.rag.placeholder import RAGPlaceholderEngine


def test_rag_schemas_validation():
    """Verify strict validation and metadata constraints on knowledge sources."""
    meta = ReliabilityMetadata(
        evidence_tier=EvidenceTier.TIER_1_CLINICAL_GUIDELINE,
        reviewed_by="Clinical Review Board",
        review_status=ReviewStatus.APPROVED,
        clinical_guideline_aligned=True
    )
    source = KnowledgeSource(
        source_id="who_test",
        title="WHO Stress Management Guide",
        publisher="World Health Organization",
        url="https://who.int/test",
        publication_date="2020-04-01",
        last_reviewed_date="2024-01-15",
        source_type=SourceType.CLINICAL_GUIDELINE,
        citation="WHO (2020). Doing What Matters in Times of Stress.",
        scope=["stress_management"],
        reliability=meta,
        sections=[{"heading": "Grounding", "content": "Notice your feet on the floor and breathe deeply."}]
    )
    assert source.source_id == "who_test"
    assert source.reliability.review_status == ReviewStatus.APPROVED


def test_chunker_rejects_unapproved_source(tmp_path):
    """Verify chunker strictly rejects unapproved or draft sources."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    unapproved_data = {
        "source_id": "unapproved_src",
        "title": "Random Blog",
        "publisher": "Unknown",
        "url": "https://example.com",
        "publication_date": "2023-01-01",
        "last_reviewed_date": "2023-01-01",
        "source_type": "government_resource",
        "citation": "Unknown (2023)",
        "scope": ["general"],
        "reliability": {
            "evidence_tier": "tier_3_peer_reviewed_literature",
            "reviewed_by": "Self",
            "review_status": "draft",
            "clinical_guideline_aligned": False
        },
        "sections": [{"heading": "Tip", "content": "Just take a nap."}]
    }
    src_file = tmp_path / "unapproved.json"
    with open(src_file, "w", encoding="utf-8") as f:
        json.dump(unapproved_data, f)

    chunks = chunker.chunk_file(str(src_file))
    assert len(chunks) == 0


def test_chunker_processes_approved_source():
    """Verify chunker accurately produces chunks with citation metadata from an approved source."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    who_file = "knowledge/sources/who_stress_management.json"
    assert Path(who_file).exists()

    chunks = chunker.chunk_file(who_file)
    assert len(chunks) >= 3
    for ch in chunks:
        assert ch.source_id == "WHO-STRESS-2020"
        assert "Doing What Matters" in ch.title
        assert ch.publisher == "World Health Organization"
        assert len(ch.content) > 20
        assert ch.word_count > 5


def test_dense_tfidf_embedding():
    """Verify dense TF-IDF embedding produces normalized vectors."""
    embedder = TfidfDenseEmbedding(dimension=16)
    docs = [
        "Take deep breaths and ground yourself to manage stress.",
        "Box breathing involves deep breathing for 4 seconds to reduce stress.",
        "Deep breathing and sleep hygiene build mental resilience and stress recovery."
    ]
    vecs = embedder.embed_documents(docs)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 3
    assert vecs.shape[1] == embedder.dimension

    # Check L2 normalization
    norms = np.linalg.norm(vecs, axis=1)
    for norm in norms:
        assert norm == pytest.approx(1.0, abs=1e-4)

    # Query embedding
    q_vec = embedder.embed_query("How to do box breathing to reduce stress?")
    assert q_vec.shape == (embedder.dimension,)
    assert np.linalg.norm(q_vec) == pytest.approx(1.0, abs=1e-4)


def test_dense_tfidf_save_and_load(tmp_path):
    """Verify embedding model can be saved and reloaded accurately."""
    embedder = TfidfDenseEmbedding(dimension=12)
    docs = ["Mindfulness meditation practice", "Diaphragmatic abdominal breathing"]
    embedder.embed_documents(docs)

    save_path = tmp_path / "embed_model.joblib"
    embedder.save(str(save_path))
    assert save_path.exists()

    reloaded = TfidfDenseEmbedding()
    reloaded.load(str(save_path))
    assert reloaded.is_fitted
    assert reloaded.dimension == embedder.dimension

    vec1 = embedder.embed_query("mindfulness")
    vec2 = reloaded.embed_query("mindfulness")
    np.testing.assert_allclose(vec1, vec2, atol=1e-5)


def test_in_memory_vector_store():
    """Verify in-memory vector store indexing and cosine similarity search."""
    store = InMemoryVectorStore()
    chunk1 = KnowledgeChunk(
        chunk_id="c1",
        source_id="s1",
        source_title="Breathing Guide",
        publisher="NHS",
        section_heading="Box Breathing",
        content="Inhale for 4 seconds, hold for 4, exhale for 4, hold for 4.",
        word_count=14,
        citation="NHS (2023)"
    )
    chunk2 = KnowledgeChunk(
        chunk_id="c2",
        source_id="s2",
        source_title="Sleep Hygiene",
        publisher="CDC",
        section_heading="Consistent Schedule",
        content="Go to bed and wake up at the same time every day.",
        word_count=13,
        citation="CDC (2024)"
    )

    # Unit vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    store.add_chunks([chunk1, chunk2], np.vstack([v1, v2]))
    assert store.count() == 2

    # Query matching chunk 1
    q_vec = np.array([0.9, 0.1, 0.0])
    results = store.search(q_vec, top_k=2, min_score=0.1)
    assert len(results) == 2
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > results[1][1]


def test_hybrid_reranker():
    """Verify hybrid reranker incorporates dense score, lexical overlap, and publisher authority."""
    reranker = HybridReranker()
    chunk = KnowledgeChunk(
        chunk_id="c1",
        source_id="who_1",
        source_title="WHO Guide",
        publisher="World Health Organization",
        section_heading="Grounding",
        content="Grounding exercise helps unhook from difficult thoughts.",
        word_count=8,
        citation="WHO (2020)"
    )
    initial_res = [RAGSearchResult(chunk=chunk, score=0.6)]

    reranked = reranker.rerank(query="grounding exercise unhook thoughts", results=initial_res, top_k=1)
    assert len(reranked) == 1
    # Score should receive boost from exact lexical overlap and tier 1 authority
    assert reranked[0].score > 0.6


def test_retrieval_engine_with_built_knowledge():
    """Verify production KnowledgeRetrievalEngine retrieves relevant passages from the built index."""
    engine = KnowledgeRetrievalEngine(knowledge_dir="knowledge", min_confidence_threshold=0.30)
    assert engine.is_initialized
    assert engine.vector_store.count() >= 18

    # Query on box breathing
    result = engine.query("box breathing technique 4 seconds hold")
    assert result.is_confident is True
    assert result.highest_score >= 0.30
    assert len(result.results) >= 1
    assert len(result.formatted_context) >= 1
    assert len(result.cited_sources) >= 1

    # Verify citation fields
    top_src = result.cited_sources[0]
    assert "title" in top_src
    assert "publisher" in top_src
    assert "url" in top_src
    assert "citation" in top_src


def test_retrieval_engine_anti_hallucination_gating():
    """Verify retrieval engine enforces anti-hallucination gate on irrelevant queries."""
    engine = KnowledgeRetrievalEngine(knowledge_dir="knowledge", min_confidence_threshold=0.30)

    # Completely off-domain technical jargon unrelated to stress/anxiety/sleep/coping
    result = engine.query("quantum mechanics nuclear magnetic resonance astrophysics rocket propulsion")
    assert result.is_confident is False
    assert len(result.results) == 0
    assert len(result.formatted_context) == 0
    assert "NO_VERIFIED_KNOWLEDGE_FOUND" in result.status_message


def test_rag_placeholder_engine():
    """Verify RAGPlaceholderEngine provides transparent fallback."""
    placeholder = RAGPlaceholderEngine()
    res = placeholder.query("Tell me about breathing")
    assert res.is_confident is False
    assert len(res.results) == 0
    assert len(res.formatted_context) == 0
    assert "placeholder active" in res.status_message
