
import json
from pathlib import Path

from app.rag.hybrid_retriever import HybridRetriever
from app.rag.fusion import reciprocal_rank_fusion


QUESTIONS_PATH = Path("data/evaluation/questions.jsonl")
TARGET_IDS = {"q005", "q008"}


def parse_chunk_id(chunk_id: str) -> dict:
    # 예: krv:PSA:23:1-5
    translation_id, book_id, chapter, verse_range = chunk_id.split(":")
    start_verse, end_verse = map(int, verse_range.split("-"))

    return {
        "translation_id": translation_id,
        "book_id": book_id,
        "chapter": int(chapter),
        "start_verse": start_verse,
        "end_verse": end_verse,
    }


def is_relevant(chunk_id: str, item: dict) -> bool:
    chunk = parse_chunk_id(chunk_id)

    if chunk["translation_id"] != item["translation_id"]:
        return False

    for expected in item["relevant_verses"]:
        if chunk["book_id"] != expected["book_id"]:
            continue

        if chunk["chapter"] != expected["chapter"]:
            continue

        # 구절 범위가 하나라도 겹치면 정답으로 인정
        if (
            chunk["start_verse"] <= expected["end_verse"]
            and chunk["end_verse"] >= expected["start_verse"]
        ):
            return True

    return False


def print_results(name: str, results: list[dict], item: dict):
    print(f"\n[{name}]")

    found = False

    for rank, result in enumerate(results, start=1):
        chunk_id = result["chunk_id"]
        relevant = is_relevant(chunk_id, item)

        if relevant:
            found = True

        marker = " <-- EXPECTED" if relevant else ""
        print(f"{rank:2}. {chunk_id}{marker}")

    if not found:
        print("정답 구절을 찾지 못했습니다.")


def load_questions() -> list[dict]:
    questions = []

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)

            if item["id"] in TARGET_IDS:
                questions.append(item)

    return questions


def main():
    retriever = HybridRetriever(candidate_k=20, rrf_k=60)
    questions = load_questions()

    for item in questions:
        question = item["question"]
        translation_id = item["translation_id"]

        print(f"\n{'=' * 60}")
        print(f"Question ID: {item['id']}")
        print(f"Question: {question}")

        dense_results = retriever.dense_retriever.search(
            question=question,
            translation_id=translation_id,
            k=20,
        )

        bm25_results = retriever.bm25_retriever.search(
            question=question,
            translation_id=translation_id,
            k=20,
        )

        hybrid_results = reciprocal_rank_fusion(
            result_lists=[dense_results, bm25_results],
            k=20,
            rrf_k=retriever.rrf_k,
        )

        print_results("Dense Top-20", dense_results, item)
        print_results("BM25 Top-20", bm25_results, item)
        print_results("Hybrid Top-20", hybrid_results, item)


if __name__ == "__main__":
    main()
