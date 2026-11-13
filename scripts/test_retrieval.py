from dotenv import load_dotenv

from app.ingestion.indexer import create_vector_store


def main():
    load_dotenv()

    vector_store = create_vector_store()

    question = "하나님이 세상을 창조하신 과정은?"

    results = vector_store.similarity_search(
        query=question,
        k=5,
        filter={"translation_id": "krv"},
    )

    print(f"질문: {question}")
    print(f"검색 결과: {len(results)}개")

    for index, document in enumerate(results, start=1):
        metadata = document.metadata

        print(f"\n=== 검색 결과 {index} ===")
        print(
            f"{metadata['book_name']} "
            f"{metadata['chapter']}장 "
            f"{metadata['start_verse']}-"
            f"{metadata['end_verse']}절"
        )
        print(document.page_content)


if __name__ == "__main__":
    main()