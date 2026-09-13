"""Knowledge base compilation and vector index builder script."""
import os
import sys
import json
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zenova.rag.chunker import DocumentChunker
from zenova.rag.embeddings.dense_tfidf import TfidfDenseEmbedding
from zenova.rag.vectorstore.numpy_store import InMemoryVectorStore
from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.build_knowledge_base")


def main():
    base_dir = Path("knowledge")
    sources_dir = base_dir / "sources"
    processed_dir = base_dir / "processed"
    embeddings_dir = base_dir / "embeddings"
    metadata_dir = base_dir / "metadata"

    sources_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_files = list(sources_dir.glob("*.json"))
    if not source_files:
        logger.error(f"No source JSON files found in {sources_dir}")
        return

    logger.info(f"Discovered {len(source_files)} curated source files in {sources_dir}")

    chunker = DocumentChunker(max_words=180, overlap_words=25)
    all_chunks = []
    sources_manifest = []

    for src_file in source_files:
        try:
            with open(src_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            source_obj = chunker.validate_source(data)
            chunks = chunker.chunk_source(data)
            all_chunks.extend(chunks)

            sources_manifest.append({
                "source_id": source_obj.source_id,
                "title": source_obj.title,
                "publisher": source_obj.publisher,
                "url": source_obj.url,
                "publication_date": source_obj.publication_date,
                "source_type": source_obj.source_type.value,
                "evidence_tier": source_obj.reliability.evidence_tier.value,
                "citation": source_obj.citation,
                "scope": source_obj.scope,
                "review_status": source_obj.reliability.review_status.value,
                "chunks_count": len(chunks)
            })
        except Exception as e:
            logger.error(f"Failed to process source file {src_file.name}: {e}")
            raise e

    # 1. Save processed chunks
    chunks_path = processed_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_chunks], f, indent=2)
    logger.info(f"Saved {len(all_chunks)} passages to {chunks_path}")

    # 2. Fit embedding model
    embedding_model = TfidfDenseEmbedding(dimension=48)
    passages = [f"{c.title} {c.section} {c.content}" for c in all_chunks]
    embedding_model.fit(passages)
    embed_model_path = embeddings_dir / "embedding_model.joblib"
    embedding_model.save(str(embed_model_path))

    # 3. Embed all chunks and index in vector store
    logger.info("Computing dense vector embeddings for all passages...")
    embeddings = embedding_model.embed_batch(passages)

    vector_store = InMemoryVectorStore()
    vector_store.add_chunks(all_chunks, embeddings)
    index_path = embeddings_dir / "index.json"
    vector_store.save(str(index_path))

    # 4. Save metadata manifests
    manifest_path = metadata_dir / "sources_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_sources": len(sources_manifest),
            "total_chunks": len(all_chunks),
            "sources": sources_manifest
        }, f, indent=2)
    logger.info(f"Saved source manifest to {manifest_path}")

    provenance_path = metadata_dir / "knowledge_provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump({
            "system": "ZENOVA Curated Knowledge Base",
            "curation_policy": "Strict peer-reviewed and authoritative global health agencies only",
            "excluded_sources": "Arbitrary internet text, unverified user forums, social media posts",
            "embedding_model": embedding_model.model_name,
            "vector_store": "InMemoryVectorStore",
            "total_indexed_passages": len(all_chunks)
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("        ZENOVA KNOWLEDGE BASE COMPILATION COMPLETE")
    print("=" * 70)
    print(f"Total Sources Ingested: {len(sources_manifest)}")
    print(f"Total Passages Indexed: {len(all_chunks)}")
    print(f"Index File:             {index_path}")
    print(f"Embedding Checkpoint:   {embed_model_path}")
    print("\nIndexed Authoritative Sources:")
    for s in sources_manifest:
        print(f"  • [{s['source_id']}] {s['publisher']}: {s['title']} ({s['chunks_count']} passages)")
    print("=" * 70)


if __name__ == "__main__":
    main()
