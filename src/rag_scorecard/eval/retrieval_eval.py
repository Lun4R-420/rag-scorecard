import json

from rag_scorecard.retrieval.vector_store import query

_QUESTIONS_PATH = "eval_sets/questions.jsonl"
_K = 5


def load_questions(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def retrieved_chunk_ids(question: str, k: int) -> list[str]:
    results = query(question, k=k)
    ids = []
    for metadata in results["metadatas"][0]:
        doc_title = metadata["doc_title"]
        chunk_index = metadata["chunk_index"]
        ids.append(f"{doc_title}_{chunk_index}")
    return ids


def score_example(expected_ids: list[str], retrieved_ids: list[str]) -> dict:
    hit = any(id in expected_ids for id in retrieved_ids)
    recall = 1.0 if hit else 0.0
    precision = sum(1 for id in retrieved_ids if id in expected_ids) / len(retrieved_ids) if retrieved_ids else 0.0
    reciprocal_rank = 0
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            reciprocal_rank = 1 / (i + 1)
            break
    return {"recall": recall, "precision": precision, "reciprocal_rank": reciprocal_rank}


def main() -> None:
    questions = load_questions(_QUESTIONS_PATH)
    print(f"scoring {len(questions)} questions at k={_K}...")

    scores = []
    for ex in questions:
        retrieved = retrieved_chunk_ids(ex["question"], _K)
        scores.append(score_example(ex["expected_chunk_ids"], retrieved))

    recall = sum(s["recall"] for s in scores) / len(scores)
    precision = sum(s["precision"] for s in scores) / len(scores)
    mrr = sum(s["reciprocal_rank"] for s in scores) / len(scores)
    print(f"recall@{_K}:    {recall:.2f}")
    print(f"precision@{_K}: {precision:.2f}")
    print(f"MRR:         {mrr:.2f}")


if __name__ == "__main__":
    main()
