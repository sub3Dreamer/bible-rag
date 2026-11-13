from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class LocalE5Embeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-small",
            device="cpu",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서들을 로컬에서 임베딩한다."""

        prefixed_texts = [
            f"passage: {text}" for text in texts
        ]

        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """사용자 질문을 로컬에서 임베딩한다."""

        embedding = self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
        )

        return embedding.tolist()