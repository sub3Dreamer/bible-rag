from langchain_chroma import Chroma

from app.ingestion.indexer import (
    create_vector_store,
    create_local_vector_store,
)


class BibleRetriever:
    def __init__(
        self,
        vector_store: Chroma | None = None,
        use_local: bool = True,
    ):
        if vector_store is not None:
            self.vector_store = vector_store
        elif use_local:
            # 로컬 임베딩 기반 검색
            self.vector_store = create_local_vector_store()
        else:
            # OpenAI 임베딩 기반 검색
            self.vector_store = create_vector_store()

    def search(
        self,
        question: str,
        translation_id: str = "krv",
        k: int = 5,
    ) -> list[dict]:
        """질문과 유사한 성경 청크를 검색한다."""

        if not question.strip():
            raise ValueError("질문은 비어 있을 수 없습니다.")

        if k < 1:
            raise ValueError("k는 1 이상이어야 합니다.")

        results = self.vector_store.similarity_search_with_score(
            query=question,
            k=k,
            filter={"translation_id": translation_id},
        )

        passages = []

        for document, distance in results:
            metadata = document.metadata

            passages.append({
                "chunk_id": metadata["chunk_id"],
                "translation_id": metadata["translation_id"],
                "book_id": metadata["book_id"],
                "book_name": metadata["book_name"],
                "chapter": metadata["chapter"],
                "start_verse": metadata["start_verse"],
                "end_verse": metadata["end_verse"],
                "text": document.page_content,
                "distance": float(distance),
            })

        return passages