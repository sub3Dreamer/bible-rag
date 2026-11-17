from dotenv import load_dotenv

load_dotenv()

from app.rag.retriever import BibleRetriever
from app.rag.bm25_retriever import BM25Retriever
from app.rag.fusion import reciprocal_rank_fusion


def print_results(name, results):
    print(f"\n=== {name} ===")

    for rank, item in enumerate(results, start=1):
        print(f"{rank}. {item['chunk_id']}")


def main():
    question = "성령의 열매는 무엇인가?"

    dense = BibleRetriever(use_local=True)
    bm25 = BM25Retriever()

    dense_results = dense.search(
        question=question,
        translation_id="krv",
        k=10,
    )

    bm25_results = bm25.search(
        question=question,
        translation_id="krv",
        k=10,
    )

    hybrid_results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=10,
        rrf_k=60,
    )

    print_results("Dense", dense_results)
    print_results("BM25", bm25_results)
    print_results("Hybrid", hybrid_results)


if __name__ == "__main__":
    main()