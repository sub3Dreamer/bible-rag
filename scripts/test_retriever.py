from dotenv import load_dotenv

from app.rag.retriever import BibleRetriever


def main():
    load_dotenv()

    retriever = BibleRetriever()

    questions = [
        "하나님이 세상을 창조하신 과정은?",
        "믿음과 행함의 관계는?",
        "야곱이 에서를 두려워한 이유는?",
        "사랑은 오래 참고 온유하다는 말씀은?",
    ]

    for question in questions:
        print(f"\n질문: {question}")

        results = retriever.search(
            question=question,
            translation_id="krv",
            k=5,
        )

        for index, passage in enumerate(results, start=1):
            print(
                f"{index}. "
                f"{passage['book_name']} "
                f"{passage['chapter']}장 "
                f"{passage['start_verse']}-"
                f"{passage['end_verse']}절 "
                f"(distance={passage['distance']:.4f})"
            )


if __name__ == "__main__":
    main()