from dotenv import load_dotenv

load_dotenv()

from app.ingestion.loader import load_jsonl
from app.ingestion.indexer import (
    create_documents,
    create_local_vector_store,
    index_chunks,
)


def main():
    chunks = load_jsonl("data/processed/krv_chunks.jsonl")
    documents = create_documents(chunks)

    vector_store = create_local_vector_store()

    added_count = index_chunks(
        vector_store=vector_store,
        documents=documents,
        batch_size=50,
    )

    print(f"로컬 인덱싱 요청 문서 수: {len(documents)}")
    print(f"새로 저장한 문서 수: {added_count}")
    print(
        "ChromaDB 저장 문서 수:",
        vector_store._collection.count(),
    )


if __name__ == "__main__":
    main()