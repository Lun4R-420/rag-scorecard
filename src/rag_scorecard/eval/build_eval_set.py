import json
import random

from rag_scorecard.ingestion.chunker import Chunk, chunk_document
from rag_scorecard.ingestion.loader import Document, load_squad

_OUTPUT_PATH = "eval_sets/questions.jsonl"
_SAMPLE_SIZE = 50
_SEED = 42


def find_chunks_containing_offset(chunks: list[Chunk], offset: int) -> list[str]:
    matching_ids = []
    for chunk in chunks:
        if chunk.start_offset <= offset < chunk.start_offset + len(chunk.text):
            matching_ids.append(f"{chunk.doc_title}_{chunk.chunk_index}")
    return matching_ids


def build_examples_for_document(doc: Document) -> list[dict]:
    chunks = chunk_document(doc)
    examples = []
    for paragraph in doc.paragraphs:
        for qa in paragraph.qas:
            if qa.get("is_impossible", False):
                continue 
            answer = qa["answers"][0]  
            global_offset = paragraph.start_offset + answer["answer_start"]
            expected_ids = find_chunks_containing_offset(chunks, global_offset)
            if not expected_ids:
                continue  
            examples.append({
                "question": qa["question"],
                "expected_chunk_ids": expected_ids,
                "answer_text": answer["text"]
            })
    return examples


def main() -> None:
    docs = load_squad("data/raw/dev-v2.0.json")

    all_examples = []
    for doc in docs:
        all_examples.extend(build_examples_for_document(doc))

    print(f"built {len(all_examples)} candidate examples from {len(docs)} documents")

    random.seed(_SEED)
    sample = random.sample(all_examples, min(_SAMPLE_SIZE, len(all_examples)))

    with open(_OUTPUT_PATH, "w") as f:
        for ex in sample:
            f.write(json.dumps(ex) + "\n")

    print(f"wrote {len(sample)} examples to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
