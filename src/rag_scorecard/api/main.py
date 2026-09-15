import time

from fastapi import FastAPI
from pydantic import BaseModel

from rag_scorecard.generation.llm_client import generate_answer
from rag_scorecard.retrieval.reranker import rerank
from rag_scorecard.retrieval.vector_store import query

app = FastAPI()

_K_FINAL = 5
_K_CANDIDATES = 20


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    start = time.perf_counter()
    results = query(request.question, k=_K_CANDIDATES)
    candidates = [{"text": doc, **meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]
    top = rerank(request.question, candidates, top_k=_K_FINAL)
    chunks = [{"text": c["text"], "chunk_index": c["chunk_index"]} for c in top]
    answer = generate_answer(request.question, chunks)
    sources = [f"{c['doc_title']}_{c['chunk_index']}" for c in top]
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"ask latency: {latency_ms:.2f} ms")
    return AskResponse(answer=answer, sources=sources, latency_ms=latency_ms)
