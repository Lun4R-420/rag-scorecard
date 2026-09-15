from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

_MODEL = "claude-haiku-4-5-20251001"
_PROMPT_PATH = Path(__file__).parent / "prompt_templates" / "answer_v1.txt"
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _build_prompt(question: str, chunks: list[dict]) -> str:
    with open(_PROMPT_PATH, "r") as f:
        template = f.read()
    context_block = "\n\n".join(f"[chunk {i}] {chunk['text']}" for i, chunk in enumerate(chunks))
    return template.format(context_block=context_block, question=question)


def generate_answer(question: str, chunks: list[dict]) -> str:
    client = _get_client()
    prompt = _build_prompt(question, chunks)
    response = client.messages.create(model=_MODEL, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
    return response.content[0].text


if __name__ == "__main__":
    from rag_scorecard.retrieval.vector_store import query

    question = "Who did the Normans conquer in England?"
    results = query(question, k=3)
    chunks = [{"text": doc, "chunk_index": i} for i, doc in enumerate(results["documents"][0])]
    print(chunks)
    answer = generate_answer(question, chunks)
    print(answer)
