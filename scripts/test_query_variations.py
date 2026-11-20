from app.rag.hybrid_retriever import HybridRetriever


questions = [
    "탕자의 비유는 어디에 나오는가?",
    "아들이 아버지의 재산을 가지고 먼 나라로 떠났다가 돌아오는 이야기",
    "아버지가 돌아온 아들을 맞이하고 잔치를 베푸는 비유",
]


def is_relevant(chunk_id: str) -> bool:
    # 정답: 누가복음 15:11-32
    translation_id, book_id, chapter, verse_range = chunk_id.split(":")
    start_verse, end_verse = map(int, verse_range.split("-"))

    return (
        translation_id == "krv"
        and book_id == "LUK"
        and int(chapter) == 15
        and start_verse <= 32
        and end_verse >= 11
    )


def main():
    retriever = HybridRetriever()

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")

        results = retriever.search(
            question=question,
            translation_id="krv",
            k=10,
        )

        found = False

        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            relevant = is_relevant(chunk_id)

            if relevant:
                found = True

            marker = " <-- EXPECTED" if relevant else ""
            print(f"{rank:2}. {chunk_id}{marker}")

        print(f"\nHit@10: {int(found)}")


if __name__ == "__main__":
    main()