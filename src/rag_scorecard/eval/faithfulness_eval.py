"""LLM-as-judge faithfulness check: does generate_answer()'s output only
claim things the retrieved chunks actually support?

This catches a different failure mode than retrieval_eval.py — retrieval
can find the right chunks and the LLM can still hallucinate on top of
them, or ignore the chunks entirely. This is the other half of "does the
system actually work," not just "did retrieval work."
"""

from pathlib import Path

from rag_scorecard.eval.retrieval_eval import load_questions
from rag_scorecard.generation.llm_client import _build_prompt, _get_client, generate_answer, _MODEL
from rag_scorecard.retrieval.vector_store import query

_JUDGE_PROMPT_PATH = Path(__file__).parent.parent / "generation" / "prompt_templates" / "judge_v1.txt"
_QUESTIONS_PATH = "eval_sets/questions.jsonl"
_SAMPLE_SIZE = 15  
_K = 5


def judge_faithfulness(question: str, chunks: list[dict], answer: str) -> dict:
    context_block = "\n\n".join(f"[chunk {i}] {chunk['text']}" for i, chunk in enumerate(chunks))
    with open(_JUDGE_PROMPT_PATH, "r") as f:
        template = f.read()

    prompt = template.format(context_block=context_block, question=question, answer=answer)
    response = _get_client().messages.create(model=_MODEL, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
    response_text = response.content[0].text
    lines = response_text.splitlines()

    verdict_line = next((line for line in lines if line.startswith("VERDICT:")), None)
    reasoning_line = next((line for line in lines if line.startswith("REASONING:")), None)
    if verdict_line is None or reasoning_line is None:
        raise ValueError("Response missing VERDICT or REASONING lines")
    
    faithful = "UNFAITHFUL" not in verdict_line
    return {"faithful": faithful, "reasoning": reasoning_line, "raw": response_text}


def main() -> None:
    questions = load_questions(_QUESTIONS_PATH)[:_SAMPLE_SIZE]
    print(f"checking faithfulness on {len(questions)} questions...")

    results = []
    for ex in questions:
        question = ex["question"]

        retrieved = query(question, k=_K)
        chunks = [{"text": doc, "chunk_index": i} for i, doc in enumerate(retrieved["documents"][0])]

        answer = generate_answer(question, chunks)
        verdict = judge_faithfulness(question, chunks, answer)

        results.append({"question": question, "answer": answer, **verdict})

    faithful_count = sum(1 for r in results if r["faithful"])
    print(f"\nfaithful: {faithful_count}/{len(results)} ({faithful_count / len(results):.0%})\n")

    unfaithful = [r for r in results if not r["faithful"]]
    if unfaithful:
        print("UNFAITHFUL examples:")
        for r in unfaithful:
            print(f"  Q: {r['question']}")
            print(f"  A: {r['answer']}")
            print(f"  {r['reasoning']}")
            print()


if __name__ == "__main__":
    main()
