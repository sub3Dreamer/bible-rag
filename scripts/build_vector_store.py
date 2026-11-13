from dotenv import load_dotenv

from app.ingestion.loader import load_jsonl
from app.ingestion.indexer import (
    create_documents,
    create_vector_store,
    index_chunks,
)


def main():
    load_dotenv()

    chunks = load_jsonl("data/processed/krv_chunks.jsonl")

    documents = create_documents(chunks)
    vector_store = create_vector_store()

    added_count = index_chunks(
        vector_store,
        documents,
        batch_size=100,
    )

    print(f"새로 인덱싱한 청크 수: {added_count}")
    print(f"전체 저장 청크 수: {vector_store._collection.count()}")


if __name__ == "__main__":
    main()