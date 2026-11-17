import hashlib
import numpy as np

from app.rag.local_embeddings import LocalE5Embeddings


def main():
    question = "성령의 열매는 무엇인가?"

    embeddings = LocalE5Embeddings()

    vectors = []

    for i in range(3):
        vector = np.asarray(
            embeddings.embed_query(question),
            dtype=np.float32,
        )

        vectors.append(vector)

        digest = hashlib.sha256(vector.tobytes()).hexdigest()

        print(f"RUN {i + 1}")
        print(f"Dimension: {len(vector)}")
        print(f"SHA256: {digest}")
        print(f"First 5: {vector[:5]}")
        print()

    for i in range(1, len(vectors)):
        max_diff = np.max(
            np.abs(vectors[0] - vectors[i])
        )

        print(
            f"RUN 1 vs RUN {i + 1}: "
            f"Max difference = {max_diff:.10f}"
        )


if __name__ == "__main__":
    main()