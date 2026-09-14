import json
from pydantic import BaseModel

class ParagraphSpan(BaseModel):
    start_offset: int
    context: str
    qas: list[dict]

class Document(BaseModel):
    title: str
    text: str
    paragraphs: list[ParagraphSpan]


def load_squad(path: str) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    documents: list[Document] = []

    for topic in raw["data"]:
        title = topic["title"]
        contexts = []
        paragraphs_spans = []
        offset = 0
        for paragraph in topic["paragraphs"]:
            ctx = paragraph["context"]
            paragraphs_spans.append(
                ParagraphSpan(
                    start_offset=offset,
                    context=ctx,
                    qas=paragraph["qas"]
                )
            )
            contexts.append(ctx)
            offset += len(ctx) + 2  # +2 for the two newlines added

        full_text = "\n\n".join(contexts)
        documents.append(Document(title=title, text=full_text, paragraphs=paragraphs_spans))

    return documents


if __name__ == "__main__":
    docs = load_squad("data/raw/dev-v2.0.json")
    print(f"loaded {len(docs)} documents")
    print(f"first doc title: {docs[0].title}")
    print(f"first doc text: {docs[0].text}")
    print(f"first doc text length: {len(docs[0].text)}")
