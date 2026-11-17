from dotenv import load_dotenv

load_dotenv()

from app.ingestion.loader import load_jsonl
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.evaluator import evaluate_question


def main():
    questions = load_jsonl("data/evaluation/questions.jsonl")

    retriever = HybridRetriever(
        candidate_k=10,
        rrf_k=60,
    )

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

        # if item["id"] == "q009":
        #     print("\n=== q009 Hybrid Top-10 ===")

        #     debug_passages = retriever.search(
        #         question=item["question"],
        #         translation_id=item["translation_id"],
        #         k=10,
        #     )

        #     for rank, passage in enumerate(debug_passages, start=1):
        #         print(
        #             f"{rank}. {passage['chunk_id']} "
        #             f"RRF={passage['rrf_score']:.6f}"
        #         )

        print(
            f"{item['id']} | "
            f"Hit={metrics['hit']} | "
            f"RR={metrics['reciprocal_rank']:.3f} | "
            f"Recall={metrics['recall']:.3f}"
        )

    if not results:
        raise ValueError("평가 질문이 없습니다.")

    count = len(results)

    hit_at_5 = sum(r["hit"] for r in results) / count
    mrr_at_5 = sum(r["reciprocal_rank"] for r in results) / count
    recall_at_5 = sum(r["recall"] for r in results) / count

    print("\n=== Hybrid Retrieval Evaluation ===")
    print(f"Questions: {count}")
    print(f"Candidate K: {retriever.candidate_k}")
    print(f"RRF K: {retriever.rrf_k}")
    print(f"Hit@5: {hit_at_5:.3f}")
    print(f"MRR@5: {mrr_at_5:.3f}")
    print(f"Range Recall@5: {recall_at_5:.3f}")


if __name__ == "__main__":
    main()