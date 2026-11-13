from dotenv import load_dotenv

from app.ingestion.loader import load_jsonl
from app.rag.retriever import BibleRetriever
from app.rag.evaluator import evaluate_question


def main():
    load_dotenv()

    questions = load_jsonl("data/evaluation/questions.jsonl")
    retriever = BibleRetriever()

    top_k_values = [5, 10, 20]

    for k in top_k_values:
        results = []

        for item in questions:
            passages = retriever.search(
                question=item["question"],
                translation_id=item["translation_id"],
                k=k,
            )

            metrics = evaluate_question(
                passages=passages,
                relevant_verses=item["relevant_verses"],
            )

            results.append(metrics)

        count = len(results)

        hit = sum(r["hit"] for r in results) / count
        mrr = sum(r["reciprocal_rank"] for r in results) / count
        recall = sum(r["recall"] for r in results) / count

        print(f"\n=== Top-{k} Evaluation ===")
        print(f"Hit@{k}: {hit:.3f}")
        print(f"MRR@{k}: {mrr:.3f}")
        print(f"Range Recall@{k}: {recall:.3f}")


if __name__ == "__main__":
    main()