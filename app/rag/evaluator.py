def is_relevant(
    passage: dict,
    relevant_verses: list[dict],
) -> bool:
    """검색된 청크가 정답 구절 범위와 겹치는지 확인한다."""

    for relevant in relevant_verses:
        if passage["book_id"] != relevant["book_id"]:
            continue

        if passage["chapter"] != relevant["chapter"]:
            continue

        # 두 구절 범위가 겹치는지 확인
        overlaps = (
            passage["start_verse"] <= relevant["end_verse"]
            and passage["end_verse"] >= relevant["start_verse"]
        )

        if overlaps:
            return True

    return False


def evaluate_question(
    passages: list[dict],
    relevant_verses: list[dict],
) -> dict:
    """한 질문에 대한 검색 품질을 평가한다."""

    relevant_flags = [
        is_relevant(passage, relevant_verses)
        for passage in passages
    ]

    # Hit@K: 정답이 하나라도 포함됐는가?
    hit = int(any(relevant_flags))

    # Reciprocal Rank: 첫 번째 정답의 역순위
    reciprocal_rank = 0.0

    for rank, relevant in enumerate(relevant_flags, start=1):
        if relevant:
            reciprocal_rank = 1.0 / rank
            break

    # Recall@K: 정답 범위 중 몇 개를 검색했는가?
    matched = set()

    for passage in passages:
        for index, relevant in enumerate(relevant_verses):
            if is_relevant(passage, [relevant]):
                matched.add(index)

    recall = (
        len(matched) / len(relevant_verses)
        if relevant_verses else 0.0
    )

    return {
        "hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "recall": recall,
    }