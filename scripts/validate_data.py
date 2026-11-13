
from app.ingestion.loader import load_jsonl
from app.ingestion.validator import (
    validate_verses,
    get_statistics,
)


def main():
    file_path = "data/processed/krv.jsonl"

    records = load_jsonl(file_path)

    errors = validate_verses(records)
    stats = get_statistics(records)

    print("=== 성경 데이터 검증 결과 ===")
    print(f"전체 구절 수: {stats['total_verses']}")
    print(f"전체 책 개수: {stats['total_books']}")
    print(f"발견된 오류 수: {len(errors)}")

    if errors:
        print("\n=== 오류 목록 (최대 20개) ===")
        for error in errors[:20]:
            print(error)
    else:
        print("기본 데이터 검증 통과!")

    # KRV 데이터셋에 대한 추가 확인
    if stats["total_verses"] != 31101:
        print("경고: 예상 구절 수(31,101개)와 다릅니다.")


if __name__ == "__main__":
    main()
