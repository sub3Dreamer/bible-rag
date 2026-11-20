import json
from pathlib import Path

from app.rag.service import BibleRAG


QUESTIONS_PATH = Path("data/evaluation/questions.jsonl")


def parse_chunk_id(chunk_id: str) -> dict:
    # 예: krv:GEN:1:1-5
    translation_id, book_id, chapter, verse_range = chunk_id.split(":")
    start_verse, end_verse = map(int, verse_range.split("-"))

    return {
        "translation_id": translation_id,
        "book_id": book_id,
        "chapter": int(chapter),
        "start_verse": start_verse,
        "end_verse": end_verse,
    }


def verse_keys(verse_range: dict) -> set:
    return {
        (
            verse_range["book_id"],
            verse_range["chapter"],
            verse,
        )
        for verse in range(
            verse_range["start_verse"],
            verse_range["end_verse"] + 1,
        )
    }


def evaluate_sources(sources: list, relevant_verses: list) -> dict:
    expected = set()

    for verse_range in relevant_verses:
        expected.update(verse_keys(verse_range))

    covered = set()
    correct_sources = 0

    for source in sources:
        parsed = parse_chunk_id(source["chunk_id"])
        source_verses = verse_keys(parsed)

        overlap = source_verses & expected

        if overlap:
            correct_sources += 1
            covered.update(overlap)

    precision = (
        correct_sources / len(sources)
        if sources else 0.0
    )

    recall = (
        len(covered) / len(expected)
        if expected else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_questions(path: Path) -> list:
    questions = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def main():
    questions = load_questions(QUESTIONS_PATH)
    rag = BibleRAG()

    results = []

    for item in questions:
        if item["id"] not in ("q004", "q005", "q008"):
            continue

        print(f"\n{'=' * 60}")
        print(f"Question ID: {item['id']}")
        print(f"Question: {item['question']}")

        result = rag.ask(
            question=item["question"],
            translation_id=item["translation_id"],
            k=5,
        )

        metrics = evaluate_sources(
            sources=result["sources"],
            relevant_verses=item["relevant_verses"],
        )

        print(f"\nAnswer:\n{result['answer']}")

        print("\nSelected Sources:")
        for source in result["sources"]:
            print(f"- {source['chunk_id']}")

        print("\nRetrieved Chunks:")
        for index, chunk in enumerate(result["retrieved_chunks"], start=1):
            print(f"{index}. {chunk['chunk_id']}")

        print(
            f"\nPrecision: {metrics['precision']:.3f}"
            f" | Recall: {metrics['recall']:.3f}"
            f" | F1: {metrics['f1']:.3f}"
        )

        results.append(metrics)

    if not results:
        print("평가할 질문이 없습니다.")
        return

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print("=== Final Evaluation ===")
    print(f"Questions: {len(results)}")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1: {avg_f1:.3f}")


if __name__ == "__main__":
    main()