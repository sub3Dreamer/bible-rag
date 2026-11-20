from app.rag.retriever import BibleRetriever
from app.rag.bm25_retriever import BM25Retriever
from app.rag.fusion import reciprocal_rank_fusion


QUESTIONS = [
    ("여호와는 나의 목자라는 말씀은?", "PSA", 23, 1, 1),
    ("탕자의 비유는 어디에 나오는가?", "LUK", 15, 11, 32),
]


def is_relevant(chunk_id, book, chapter, start, end):
    _, chunk_book, chunk_chapter, verse_range = chunk_id.split(":")
    chunk_start, chunk_end = map(int, verse_range.split("-"))

    return (
        chunk_book == book
        and int(chunk_chapter) == chapter
        and chunk_start <= end
        and chunk_end >= start
    )


def main():
    dense = BibleRetriever(use_local=True)
    bm25 = BM25Retriever()

    for question, book, chapter, start, end in QUESTIONS:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")

        dense_results = dense.search(
            question=question,
            translation_id="krv",
            k=100,
        )
        bm25_results = bm25.search(
            question=question,
            translation_id="krv",
            k=100,
        )

        for candidate_k in [10, 20, 50, 100]:
            fused = reciprocal_rank_fusion(
                result_lists=[
                    dense_results[:candidate_k],
                    bm25_results[:candidate_k],
                ],
                k=100,
                rrf_k=60,
            )

            matches = [
                (rank, result["chunk_id"])
                for rank, result in enumerate(fused, start=1)
                if is_relevant(
                    result["chunk_id"], book, chapter, start, end
                )
            ]

            print(f"\nCandidate K: {candidate_k}")

            if matches:
                for rank, chunk_id in matches[:5]:
                    print(f"Rank {rank}: {chunk_id}")

                hit_at_5 = any(rank <= 5 for rank, _ in matches)
                print(f"Hit@5: {hit_at_5}")
            else:
                print("No relevant chunks in fused results")


if __name__ == "__main__":
    main()