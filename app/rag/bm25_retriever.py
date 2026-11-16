import re

from rank_bm25 import BM25Okapi

from app.ingestion.loader import load_jsonl


def tokenize(text: str) -> list[str]:
    """한국어 성경 검색을 위한 기본 토큰화."""
    return re.findall(r"[가-힣]+|[a-zA-Z]+|\d+", text.lower())


class BM25Retriever:
    def __init__(self, chunks_path: str = "data/processed/krv_chunks.jsonl"):
        self.chunks = load_jsonl(chunks_path)

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        question: str,
        translation_id: str = "krv",
        k: int = 5,
    ) -> list[dict]:
        if k <= 0:
            return []

        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            if chunk["translation_id"] != translation_id:
                continue

            if scores[index] <= 0:
                continue

            results.append({
                "chunk_id": chunk["chunk_id"],
                "translation_id": chunk["translation_id"],
                "book_id": chunk["book_id"],
                "book_name": chunk["book_name"],
                "chapter": chunk["chapter"],
                "start_verse": chunk["start_verse"],
                "end_verse": chunk["end_verse"],
                "text": chunk["text"],
                "score": float(scores[index]),
            })

            if len(results) >= k:
                break

        return results