from collections import defaultdict


def chunk_verses(
    records: list[dict],
    chunk_size: int = 5,
) -> list[dict]:
    """
    성경 구절을 같은 역본, 책, 장 내에서
    chunk_size개씩 묶는다.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")

    # (역본, 책, 장) 기준 그룹화
    groups = defaultdict(list)

    for record in records:
        key = (
            record["translation_id"],
            record["book_id"],
            record["chapter"],
        )
        groups[key].append(record)

    chunks = []

    for group in groups.values():
        # 절 번호순 정렬
        group.sort(key=lambda r: r["verse"])

        # chunk_size 단위로 분할
        for i in range(0, len(group), chunk_size):
            batch = group[i:i + chunk_size]

            first = batch[0]
            last = batch[-1]

            chunk = {
                "chunk_id": (
                    f"{first['translation_id']}:"
                    f"{first['book_id']}:"
                    f"{first['chapter']}:"
                    f"{first['verse']}-{last['verse']}"
                ),
                "translation_id": first["translation_id"],
                "book_id": first["book_id"],
                "book_name": first["book_name"],
                "chapter": first["chapter"],
                "start_verse": first["verse"],
                "end_verse": last["verse"],
                "text": "\n".join(
                    f"{r['verse']}. {r['text']}"
                    for r in batch
                ),
                "source": first["source"],
            }

            chunks.append(chunk)

    return chunks