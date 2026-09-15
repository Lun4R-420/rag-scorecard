from pathlib import Path

import chromadb

from rag_scorecard.ingestion.chunker import Chunk
from rag_scorecard.retrieval.embed import embed_texts

_PERSIST_DIR = str(Path(__file__).parents[3] / "data" / "chroma")
_COLLECTION_NAME = "squad_chunks"


def get_collection():
    chromadb_client = chromadb.PersistentClient(path=_PERSIST_DIR)
    collection = chromadb_client.get_or_create_collection(name=_COLLECTION_NAME)
    return collection


def add_chunks(chunks: list[Chunk]) -> None:
    collection = get_collection()
    ids = [f"{c.doc_title}_{c.chunk_index}" for c in chunks]
    embeddings = embed_texts([c.text for c in chunks])
    documents = [c.text for c in chunks]
    metadatas = [{"doc_title": c.doc_title, "chunk_index": c.chunk_index, "start_offset": c.start_offset} for c in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query(question: str, k: int = 5):
    collection = get_collection()
    question_embedding = embed_texts([question])[0]
    results = collection.query(query_embeddings=[question_embedding], n_results=k)
    return results


if __name__ == "__main__":
    from rag_scorecard.ingestion.loader import load_squad
    from rag_scorecard.ingestion.chunker import chunk_document

    docs = load_squad("data/raw/dev-v2.0.json")
    all_chunks = [c for d in docs for c in chunk_document(d)]
    print(f"embedding + storing {len(all_chunks)} chunks...")
    add_chunks(all_chunks)

    results = query("Who did the Normans conquer in England?", k=3)
    print(results)
