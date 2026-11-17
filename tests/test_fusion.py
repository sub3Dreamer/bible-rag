import pytest

from app.rag.fusion import reciprocal_rank_fusion


def test_rrf_merges_duplicate_chunks():
    dense_results = [
        {"chunk_id": "A"},
        {"chunk_id": "B"},
    ]

    bm25_results = [
        {"chunk_id": "B"},
        {"chunk_id": "C"},
    ]

    results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=3,
    )

    assert len(results) == 3
    assert results[0]["chunk_id"] == "B"

    expected_score = 1 / 62 + 1 / 61

    assert results[0]["rrf_score"] == pytest.approx(
        expected_score
    )


def test_rrf_limits_results():
    results = reciprocal_rank_fusion(
        [[
            {"chunk_id": "A"},
            {"chunk_id": "B"},
            {"chunk_id": "C"},
        ]],
        k=2,
    )

    assert len(results) == 2


def test_rrf_empty_results():
    results = reciprocal_rank_fusion(
        [[], []],
        k=5,
    )

    assert results == []