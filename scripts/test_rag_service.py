from app.rag.service import BibleRAG


def main():
    rag = BibleRAG()

    question = "성령의 열매에는 어떤 것들이 있나요?"

    result = rag.ask(
        question=question,
        translation_id="krv",
        k=5,
    )

    print("=== Question ===")
    print(result["question"])

    print("\n=== Answer ===")
    print(result["answer"])

    print("\n=== Retrieved Sources ===")

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(f"{index}. {source['chunk_id']}")

    print("\n=== Retrieved Chunks ===")
    print(len(result["retrieved_chunks"]))


if __name__ == "__main__":
    main()