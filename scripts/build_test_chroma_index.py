
import chromadb


def main():
    client = chromadb.PersistentClient(
        path="data/chroma_local"
    )

    source = client.get_collection(
        name="bible_krv_e5_small"
    )

    target_name = "bible_krv_e5_small_ef500"

    # 기존 실험 컬렉션이 있다면 덮어쓰지 않는다.
    existing_names = [
        collection.name
        for collection in client.list_collections()
    ]

    if target_name in existing_names:
        print(f"이미 존재하는 컬렉션: {target_name}")
        return

    target = client.create_collection(
        name=target_name,
        configuration={
            "hnsw": {
                "space": "l2",
                "ef_construction": 100,
                "ef_search": 500,
                "max_neighbors": 16,
            }
        },
    )

    batch_size = 200
    total = source.count()

    for offset in range(0, total, batch_size):
        data = source.get(
            limit=batch_size,
            offset=offset,
            include=[
                "embeddings",
                "metadatas",
                "documents",
            ],
        )

        if not data["ids"]:
            break

        kwargs = {
            "ids": data["ids"],
            "embeddings": data["embeddings"].tolist(),
        }

        if data["metadatas"] is not None:
            kwargs["metadatas"] = data["metadatas"]

        if data["documents"] is not None:
            kwargs["documents"] = data["documents"]

        target.add(**kwargs)

        print(
            f"Copied {min(offset + batch_size, total)}/{total}"
        )

    print("\n=== Test Index ===")
    print("Source:", source.count())
    print("Target:", target.count())
    print("Configuration:", target.configuration)


if __name__ == "__main__":
    main()
