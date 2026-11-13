from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from app.rag.local_embeddings import LocalE5Embeddings


EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "bible_krv_test"


def create_documents(chunks: list[dict]) -> list[Document]:
    """청크 딕셔너리를 LangChain Document로 변환한다."""
    documents = []

    for chunk in chunks:
        content = (
            f"{chunk['book_name']} {chunk['chapter']}장 "
            f"{chunk['start_verse']}-{chunk['end_verse']}절\n"
            f"{chunk['text']}"
        )

        document = Document(
            id=chunk["chunk_id"],
            page_content=content,
            metadata={
                "translation_id": chunk["translation_id"],
                "book_id": chunk["book_id"],
                "book_name": chunk["book_name"],
                "chapter": chunk["chapter"],
                "start_verse": chunk["start_verse"],
                "end_verse": chunk["end_verse"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
            },
        )

        documents.append(document)

    return documents


# OpenAI 모델 임베딩
def create_vector_store(
    persist_directory: str = "data/chroma",
) -> Chroma:
    """영속 저장되는 ChromaDB 컬렉션을 생성하거나 연다."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(Path(persist_directory)),
    )


# 로컬 모델 임베딩
def create_local_vector_store() -> Chroma:
    embeddings = LocalE5Embeddings()

    return Chroma(
        collection_name="bible_krv_e5_small",
        embedding_function=embeddings,
        persist_directory="data/chroma_local",
    )


def index_chunks(
    vector_store: Chroma,
    documents: list[Document],
    batch_size: int = 100,
) -> int:
    """저장되지 않은 문서만 배치 단위로 인덱싱한다."""
    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    existing = vector_store.get(
        ids=[doc.id for doc in documents],
        include=[],
    )
    existing_ids = set(existing["ids"])

    new_documents = [
        doc for doc in documents
        if doc.id not in existing_ids
    ]

    for i in range(0, len(new_documents), batch_size):
        batch = new_documents[i:i + batch_size]

        vector_store.add_documents(
            documents=batch,
            ids=[doc.id for doc in batch],
        )

        print(
            f"인덱싱 진행: "
            f"{min(i + batch_size, len(new_documents))}"
            f"/{len(new_documents)}"
        )

    return len(new_documents)
