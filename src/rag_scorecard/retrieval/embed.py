from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(texts)
    return vectors.tolist()


if __name__ == "__main__":
    vectors = embed_texts(["hello world", "the normans invaded england"])
    print(f"num vectors: {len(vectors)}")
    print(f"vector dim: {len(vectors[0])}")
