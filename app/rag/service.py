from app.rag.hybrid_retriever import HybridRetriever
from app.rag.generator import BibleGenerator


class BibleRAG:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = BibleGenerator()

    def ask(
        self,
        question: str,
        translation_id: str = "krv",
        k: int = 5,
    ) -> dict:
        # 1. 사용자 질문 검증
        if not question.strip():
            raise ValueError("질문이 비어 있습니다.")

        if k < 1:
            raise ValueError("k는 1 이상이어야 합니다.")

        # 2. Hybrid Search
        chunks = self.retriever.search(
            question=question,
            translation_id=translation_id,
            k=k,
        )

        # print("\n=== DEBUG: Retrieved Chunks ===")
        # for index, chunk in enumerate(chunks, start=1):
        #     print(f"{index}. {chunk['chunk_id']}")

        # 3. 검색 결과가 없는 경우
        if not chunks:
            return {
                "question": question,
                "answer": "관련 성경 구절을 찾지 못했습니다.",
                "sources": [],
                "retrieved_chunks": [],
            }

        # 4. LLM에 전달할 컨텍스트 구성
        context_parts = []

        for index, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[자료 {index}]\n"
                f"출처: {chunk['chunk_id']}\n"
                f"내용: {chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        # 5. 로컬 LLM 답변 생성
        generation = self.generator.generate(
            question=question,
            context=context,
        )

        # print("\n=== DEBUG: Generation ===")
        # print(generation)

        answer = generation["answer"]
        source_ids = generation["source_ids"]

        # print("\n=== DEBUG: Source IDs ===")
        # print(source_ids)
        # print("Source IDs Type:", type(source_ids))

        # 6. 검색 결과 기반 출처 구성
        sources = []
        seen = set()

        for source_id in source_ids:
            if type(source_id) is not int:
                continue

            if source_id < 1 or source_id > len(chunks):
                continue

            if source_id in seen:
                continue

            seen.add(source_id)

            chunk = chunks[source_id - 1]

            sources.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            })

        if not sources:
            answer = "제공된 자료에서 답을 확인할 수 없습니다."

        # 7. 구조화된 결과 반환
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
        }