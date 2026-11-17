import numpy as np
import chromadb

from app.rag.local_embeddings import LocalE5Embeddings


def main():
    question = "성령의 열매는 무엇인가?"

    embeddings = LocalE5Embeddings()

    query_vector = np.asarray(
        embeddings.embed_query(question),
        dtype=np.float32,
    )

    client = chromadb.PersistentClient(
        path="data/chroma_local"
    )

    collection = client.get_collection(
        name="bible_krv_e5_small"
    )

    # ChromaDB에 저장된 전체 벡터를 가져온다.
    data = collection.get(
        include=["embeddings"]
    )

    ids = data["ids"]
    vectors = np.asarray(
        data["embeddings"],
        dtype=np.float32,
    )

    print(f"Stored vectors: {len(ids)}")
    print(f"Vector dimensions: {vectors.shape[1]}")

    # 저장된 벡터와 질문 벡터 사이의 코사인 거리
    query_norm = np.linalg.norm(query_vector)
    vector_norms = np.linalg.norm(vectors, axis=1)

    similarities = (
        vectors @ query_vector
    ) / (vector_norms * query_norm)

    distances = 1.0 - similarities

    ranked_indices = np.argsort(distances)[:10]

    print("\n=== Exact Search Top-10 ===")

    for rank, index in enumerate(ranked_indices, start=1):
        print(
            f"{rank}. {ids[index]} "
            f"cosine_distance={distances[index]:.8f}"
        )

    print("\n=== ChromaDB ANN Top-10 ===")

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=10,
        include=["distances"],
    )

    for rank, (chunk_id, distance) in enumerate(
        zip(results["ids"][0], results["distances"][0]),
        start=1,
    ):
        print(
            f"{rank}. {chunk_id} "
            f"distance={distance:.8f}"
        )

    print("Collection metadata:", collection.metadata)
    print("Collection configuration:", collection.configuration)


if __name__ == "__main__":
    main()