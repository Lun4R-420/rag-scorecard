# RAG Scorecard

A retrieval-augmented generation system built around evaluation, not just generation. Most RAG demos stop at "it answers questions from my docs." This one measures whether it's actually working, and shows the numbers moving when you improve it.

## The problem this solves

A RAG system can look fine in a demo and still be wrong in ways you can't see without an eval harness. Maybe retrieval pulls the wrong chunks but the LLM still generates a plausible-sounding answer. Maybe retrieval finds the right chunk but the LLM ignores it and hallucinates anyway. Without retrieval metrics and a faithfulness check, you can't tell which failure mode you're looking at, and you can't fix what you can't see.

## What it does

Given a question, it retrieves relevant chunks from a corpus, reranks them for relevance, generates an answer with a forced citation, and checks whether that answer is actually faithful to the retrieved context. Every stage is measured, not just eyeballed.

Corpus: SQuAD v2.0 (dev set), 35 Wikipedia-derived topics. The eval set is auto-labeled using SQuAD's own question/answer pairs and their character offsets, mapped back to whichever chunk contains the answer, so there was no manual labeling step.

## Architecture

```
data/raw/dev-v2.0.json
        |
        v
  loader.py (parses into per-topic documents, tracks paragraph offsets)
        |
        v
  chunker.py (sliding window, 600 tokens, 100 token overlap)
        |
        v
  embed.py (sentence-transformers, all-MiniLM-L6-v2)
        |
        v
  vector_store.py (Chroma, persisted locally)
        |
        v
  reranker.py (cross-encoder, ms-marco-MiniLM-L-6-v2)
        |
        v
  llm_client.py (Claude, forced citation prompt)
        |
        v
       answer
```

Eval sits alongside this, not inside it:

- `build_eval_set.py` auto-generates `eval_sets/questions.jsonl` from SQuAD's answer offsets.
- `retrieval_eval.py` scores recall@k, precision@k, and MRR against that set.
- `faithfulness_eval.py` uses an LLM judge to check whether generated answers only claim things the retrieved chunks actually support.
- `compare_reranker.py` runs the same eval logic with and without reranking, side by side.

## Results

Baseline vector search vs. vector search plus cross-encoder reranking, on 50 auto-labeled questions:

| metric | baseline | reranked |
|---|---|---|
| recall@5 | 0.76 | 0.86 |
| precision@5 | 0.17 | 0.20 |
| MRR | 0.57 | 0.76 |

Faithfulness (LLM-as-judge, sample of 15): 100% faithful. The generation prompt forces the model to say "I don't know" when the retrieved context doesn't answer the question, which is most of why this number is high. It's a good sign the prompt is doing its job, though it also means faithfulness has little room to move until it's compared against a looser prompt as a baseline.

Latency on the `/ask` endpoint: first request after server start is slow (around 14 seconds, all of it spent loading the embedding and reranker models into memory). Once warm, a full request (retrieve, rerank, generate) takes about 3.8 seconds.

## Running it

```bash
git clone https://github.com/Lun4R-420/rag-scorecard.git
cd rag-scorecard
uv sync
```

Set `ANTHROPIC_API_KEY` in a `.env` file at the project root (see `.env.example`).

Download the corpus:

```bash
curl -o data/raw/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

Build the vector store and eval set:

```bash
python -m rag_scorecard.retrieval.vector_store
python -m rag_scorecard.eval.build_eval_set
```

Run the evals:

```bash
python -m rag_scorecard.eval.retrieval_eval
python -m rag_scorecard.eval.compare_reranker
python -m rag_scorecard.eval.faithfulness_eval
```

Run the API:

```bash
python -m uvicorn rag_scorecard.api.main:app --reload
```

Then `POST /ask` with `{"question": "..."}`. Docs at `/docs`.

## Tech stack

Python, `uv` for dependency management, sentence-transformers for embeddings and reranking, Chroma for the vector store, the Anthropic API for generation and judging, FastAPI for the endpoint, tiktoken for token-aware chunking.

## A few things worth knowing if you read the code

- Chunk offsets are tracked back to the original document text, which is what makes the auto-labeled eval set possible. SQuAD gives you an `answer_start` per question; the loader tracks where each paragraph starts inside the joined document text, so an answer's position can be mapped straight to the chunk that contains it.
- The reranker retrieves a wider candidate set (20) from the vector store before narrowing down to the final 5, since vector search is fast but approximate and the cross-encoder is slower but more accurate on a small set.
- Two real bugs came from the same root cause during development: a couple of file paths (`.env` and the Chroma persist directory) were written as relative paths, which worked fine when running scripts directly but broke once the API server ran with a different working directory. Both are now resolved relative to the file's own location instead of assuming a particular working directory.

## What's not done

- No cost/token tracking per request yet, only latency.
- Prompt versioning exists (`answer_v1.txt`, `judge_v1.txt`) but hasn't been iterated on yet.
- Faithfulness eval hasn't been tested against a deliberately looser prompt for comparison.
- No chunk size/overlap A/B test yet, only the reranker before/after.
