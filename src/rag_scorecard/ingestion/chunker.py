import tiktoken
from pydantic import BaseModel

from rag_scorecard.ingestion.loader import Document

_encoding = tiktoken.get_encoding("cl100k_base")


class Chunk(BaseModel):
    doc_title: str
    chunk_index: int
    text: str
    start_offset: int


def chunk_document(doc: Document, chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    token_ids = _encoding.encode(doc.text)
    chunks = []
    for i in range(0, len(token_ids), chunk_size - overlap):
        window = token_ids[i:i + chunk_size]
        chunk_text = _encoding.decode(window)
        start_offset = doc.text.find(chunk_text)
        chunks.append(Chunk(doc_title=doc.title, chunk_index=len(chunks), text=chunk_text, start_offset=start_offset))
    return chunks


if __name__ == "__main__":
    from rag_scorecard.ingestion.loader import load_squad

    docs = load_squad("data/raw/dev-v2.0.json")
    chunks = chunk_document(docs[0])
    print(f"doc '{docs[0].title}' -> {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"chunk {c.chunk_index} (offset {c.start_offset}): {c.text[:80]!r}")
