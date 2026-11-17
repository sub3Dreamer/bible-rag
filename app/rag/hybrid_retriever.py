from app.rag.retriever import BibleRetriever
from app.rag.bm25_retriever import BM25Retriever
from app.rag.fusion import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(
        self,
        dense_retriever=None,
        bm25_retriever=None,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")

        self.dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else BibleRetriever(use_local=True)
        )

        self.bm25_retriever = (
            bm25_retriever
            if bm25_retriever is not None
            else BM25Retriever()
        )

        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(
        self,
        question: str,
        translation_id: str = "krv",
        k: int = 5,
    ) -> list[dict]:
        if k <= 0:
            return []

        candidate_k = max(self.candidate_k, k)

        dense_results = self.dense_retriever.search(
            question=question,
            translation_id=translation_id,
            k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            question=question,
            translation_id=translation_id,
            k=candidate_k,
        )

        return reciprocal_rank_fusion(
            result_lists=[dense_results, bm25_results],
            k=k,
            rrf_k=self.rrf_k,
        )