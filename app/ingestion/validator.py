
from collections import Counter


REQUIRED_FIELDS = {
    "translation_id",
    "verse_id",
    "book_id",
    "book_order",
    "book_name",
    "chapter",
    "verse",
    "text",
    "source",
}


def validate_verses(records: list[dict]) -> list[str]:
    """
    정규화된 성경 데이터를 검증한다.

    반환값:
        오류 메시지 목록
        오류가 없으면 빈 리스트 반환
    """
    errors = []
    verse_ids = set()

    for index, record in enumerate(records, start=1):

        # 1. 필수 필드 검사
        missing = REQUIRED_FIELDS - record.keys()

        if missing:
            errors.append(
                f"{index}번째 레코드: 필수 필드 누락 {missing}"
            )
            continue

        # 2. 장절 번호 검사
        chapter = record["chapter"]
        verse = record["verse"]

        if type(chapter) is not int or chapter < 1:
            errors.append(
                f"{index}번째 레코드: 잘못된 chapter {chapter}"
            )

        if type(verse) is not int or verse < 1:
            errors.append(
                f"{index}번째 레코드: 잘못된 verse {verse}"
            )

        # 3. 본문 검사
        text = record["text"]

        if not isinstance(text, str) or not text.strip():
            errors.append(
                f"{index}번째 레코드: 본문이 비어 있거나 잘못된 타입"
            )

        # 4. 중복 구절 검사
        key = (
            record["translation_id"],
            record["verse_id"],
        )

        if key in verse_ids:
            errors.append(
                f"{index}번째 레코드: 중복 구절 {key}"
            )
        else:
            verse_ids.add(key)

    # 5. 책 개수 검사
    book_ids = {
        record["book_id"]
        for record in records
        if isinstance(record.get("book_id"), str)
    }

    if len(book_ids) != 66:
        errors.append(
            f"성경 책 개수 오류: {len(book_ids)}권 (기대값 66권)"
        )

    return errors


def get_statistics(records: list[dict]) -> dict:
    """성경 데이터의 기본 통계를 반환한다."""

    books = Counter(record["book_id"] for record in records)

    return {
        "total_verses": len(records),
        "total_books": len(books),
        "verses_by_book": dict(books),
    }
