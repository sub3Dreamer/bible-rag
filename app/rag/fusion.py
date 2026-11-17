from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """여러 검색 결과를 RRF 방식으로 결합한다."""

    if k <= 0:
        return []

    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")

    scores = defaultdict(float)
    documents = {}

    for results in result_lists:
        seen = set()

        for rank, document in enumerate(results, start=1):
            chunk_id = document["chunk_id"]

            # 같은 검색기에서 중복된 청크는 한 번만 반영
            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            scores[chunk_id] += 1.0 / (rrf_k + rank)

            if chunk_id not in documents:
                documents[chunk_id] = document

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    fused_results = []

    for chunk_id in ranked_ids[:k]:
        result = documents[chunk_id].copy()
        result["rrf_score"] = scores[chunk_id]
        fused_results.append(result)

    return fused_results