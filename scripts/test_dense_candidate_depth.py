from app.rag.retriever import BibleRetriever


QUESTIONS = [
    {
        "id": "q005",
        "question": "여호와는 나의 목자라는 말씀은?",
        "book_id": "PSA",
        "chapter": 23,
        "start_verse": 1,
        "end_verse": 1,
    },
    {
        "id": "q008",
        "question": "탕자의 비유는 어디에 나오는가?",
        "book_id": "LUK",
        "chapter": 15,
        "start_verse": 11,
        "end_verse": 32,
    },
]


def is_relevant(chunk_id: str, item: dict) -> bool:
    translation_id, book_id, chapter, verse_range = chunk_id.split(":")
    start_verse, end_verse = map(int, verse_range.split("-"))

    return (
        translation_id == "krv"
        and book_id == item["book_id"]
        and int(chapter) == item["chapter"]
        and start_verse <= item["end_verse"]
        and end_verse >= item["start_verse"]
    )


def main():
    retriever = BibleRetriever(use_local=True)

    for item in QUESTIONS:
        print(f"\n{'=' * 60}")
        print(f"Question: {item['question']}")

        results = retriever.search(
            question=item["question"],
            translation_id="krv",
            k=100,
        )

        matches = []

        for rank, result in enumerate(results, start=1):
            if is_relevant(result["chunk_id"], item):
                matches.append((rank, result["chunk_id"]))

        if matches:
            print("\nRelevant Chunks:")
            for rank, chunk_id in matches:
                print(f"Rank {rank}: {chunk_id}")
        else:
            print("\nNo relevant chunks found in Dense Top-100.")


if __name__ == "__main__":
    main()