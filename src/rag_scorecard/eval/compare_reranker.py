from rag_scorecard.eval.retrieval_eval import load_questions, score_example
from rag_scorecard.retrieval.reranker import rerank
from rag_scorecard.retrieval.vector_store import query

_QUESTIONS_PATH = "eval_sets/questions.jsonl"
_K_FINAL = 5      
_K_CANDIDATES = 20  


def baseline_ids(question: str) -> list[str]:
    results = query(question, k=_K_FINAL)
    ids = []
    for metadata in results["metadatas"][0]:
        doc_title = metadata["doc_title"]
        chunk_index = metadata["chunk_index"]
        ids.append(f"{doc_title}_{chunk_index}")
    return ids


def reranked_ids(question: str) -> list[str]:
    results = query(question, k=_K_CANDIDATES)
    candidates = [{"text": doc, **meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]
    top = rerank(question, candidates, top_k=_K_FINAL)
    return [f"{c['doc_title']}_{c['chunk_index']}" for c in top]


def average_scores(questions: list[dict], id_fn) -> dict:
    scores = []
    for question in questions:
        retrieved = id_fn(question["question"])
        scores.append(score_example(question["expected_chunk_ids"], retrieved))

    recall = sum(s["recall"] for s in scores) / len(scores)
    precision = sum(s["precision"] for s in scores) / len(scores)
    mrr = sum(s["reciprocal_rank"] for s in scores) / len(scores)
    return {"recall": recall, "precision": precision, "reciprocal_rank": mrr}


def main() -> None:
    questions = load_questions(_QUESTIONS_PATH)

    print("scoring baseline (vector search only)...")
    baseline = average_scores(questions, baseline_ids)

    print("scoring reranked (vector search -> cross-encoder)...")
    reranked = average_scores(questions, reranked_ids)

    print(f"\n{'metric':<12}{'baseline':<12}{'reranked':<12}")
    for key in ("recall", "precision", "reciprocal_rank"):
        print(f"{key:<12}{baseline[key]:<12.2f}{reranked[key]:<12.2f}")


if __name__ == "__main__":
    main()
