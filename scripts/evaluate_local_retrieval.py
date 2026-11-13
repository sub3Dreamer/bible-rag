from dotenv import load_dotenv

load_dotenv()

from app.ingestion.loader import load_jsonl
from app.rag.retriever import BibleRetriever
from app.rag.evaluator import evaluate_question


def main():
    questions = load_jsonl("data/evaluation/questions.jsonl")
    retriever = BibleRetriever(use_local=True)

    results = []

    for item in questions:
        passages = retriever.search(
            question=item["question"],
            translation_id=item["translation_id"],
            k=5,
        )

        metrics = evaluate_question(
            passages=passages,
            relevant_verses=item["relevant_verses"],
        )

        results.append(metrics)

        print(
            f"{item['id']} | "
            f"Hit={metrics['hit']} | "
            f"RR={metrics['reciprocal_rank']:.3f} | "
            f"Recall={metrics['recall']:.3f}"
        )

    count = len(results)

    if count == 0:
        raise ValueError("평가 질문이 없습니다.")

    hit_at_5 = sum(r["hit"] for r in results) / count
    mrr_at_5 = sum(r["reciprocal_rank"] for r in results) / count
    recall_at_5 = sum(r["recall"] for r in results) / count

    print("\n=== Local E5 Retrieval Evaluation ===")
    print(f"Questions: {count}")
    print(f"Hit@5: {hit_at_5:.3f}")
    print(f"MRR@5: {mrr_at_5:.3f}")
    print(f"Range Recall@5: {recall_at_5:.3f}")


if __name__ == "__main__":
    main()