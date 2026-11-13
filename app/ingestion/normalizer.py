def normalize_verse(
    record: dict,
    translation_id: str
) -> dict:
    """
    원본 성경 레코드를 공통 구조로 변환한다.
    """
    return {
        "translation_id": translation_id,
        "verse_id": f"{record['code']}.{record['chapter']}.{record['verse']}",
        "book_id": record["code"],
        "book_order": record["book"],
        "book_name": record["name_kr"],
        "chapter": record["chapter"],
        "verse": record["verse"],
        "text": record["text"].strip(),
        "source": record["source"],
    }