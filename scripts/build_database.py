
from app.ingestion.loader import load_jsonl
from app.ingestion.validator import validate_verses
from app.translations.verse_repository import VerseRepository


def main():
    records = load_jsonl("data/processed/krv.jsonl")

    errors = validate_verses(records)

    if errors:
        print(f"검증 실패: {len(errors)}개 오류")
        for error in errors[:10]:
            print(error)
        return

    repository = VerseRepository()

    repository.create_table()
    repository.insert_verses(records)

    print(f"SQLite 저장 완료: {len(records)}개 구절")


if __name__ == "__main__":
    main()
