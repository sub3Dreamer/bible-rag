from app.rag.hybrid_retriever import HybridRetriever
from app.rag.generator import BibleGenerator


def main():
    question = "성령의 열매에는 어떤 것들이 있나요?"

    retriever = HybridRetriever()
    generator = BibleGenerator()

    results = retriever.search(
        question=question,
        translation_id="krv",
        k=5,
    )

    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"[자료 {index}]\n"
            f"출처: {result['chunk_id']}\n"
            f"내용: {result['text']}"
        )

    context = "\n\n".join(context_parts)

    print("=== Retrieved Context ===")
    print(context)

    answer = generator.generate(
        question=question,
        context=context,
    )

    print("\n=== Generated Answer ===")
    print(answer)


if __name__ == "__main__":
    main()