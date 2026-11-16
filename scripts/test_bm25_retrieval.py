from app.rag.bm25_retriever import BM25Retriever


def main():
    retriever = BM25Retriever()

    question = "여호와는 나의 목자시니"
    results = retriever.search(question, k=5)

    print(f"질문: {question}\n")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{result['book_name']} "
            f"{result['chapter']}장 "
            f"{result['start_verse']}-"
            f"{result['end_verse']}절 "
            f"(BM25={result['score']:.3f})"
        )


if __name__ == "__main__":
    main()