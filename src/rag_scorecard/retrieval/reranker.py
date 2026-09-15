from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


def rerank(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    pairs = [[question, c["text"]] for c in candidates]
    scores = _get_reranker().predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = score
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


if __name__ == "__main__":
    from rag_scorecard.retrieval.vector_store import query

    question = "Who did the Normans conquer in England?"
    results = query(question, k=10)
    candidates = [
        {"text": doc, **meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    top = rerank(question, candidates, top_k=3)
    for c in top:
        print(f"score={c['rerank_score']:.3f}  {c['doc_title']}_{c['chunk_index']}  {c['text'][:80]!r}")
