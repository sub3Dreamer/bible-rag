import hashlib
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

    print(
        "Embedding SHA256:",
        hashlib.sha256(query_vector.tobytes()).hexdigest(),
    )

    client = chromadb.PersistentClient(
        path="data/chroma_local"
    )

    # collection = client.get_collection(
    #     name="bible_krv_e5_small"
    # )

    collection = client.get_collection(
        name="bible_krv_e5_small_ef500"
    )

    print("Collection count:", collection.count())

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=10,
        include=["distances"],
    )

    print("\n=== ChromaDB Top-10 ===")

    for rank, (chunk_id, distance) in enumerate(
        zip(results["ids"][0], results["distances"][0]),
        start=1,
    ):
        print(
            f"{rank}. {chunk_id} "
            f"distance={distance:.8f}"
        )


if __name__ == "__main__":
    main()