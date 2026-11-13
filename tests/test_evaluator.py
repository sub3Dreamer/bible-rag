from app.rag.evaluator import evaluate_question


def test_evaluate_question():
    relevant = [{
        "book_id": "JAS",
        "chapter": 2,
        "start_verse": 14,
        "end_verse": 26,
    }]

    passages = [
        {
            "book_id": "HEB",
            "chapter": 11,
            "start_verse": 6,
            "end_verse": 10,
        },
        {
            "book_id": "PSA",
            "chapter": 33,
            "start_verse": 21,
            "end_verse": 22,
        },
        {
            "book_id": "JAS",
            "chapter": 2,
            "start_verse": 16,
            "end_verse": 20,
        },
    ]

    result = evaluate_question(passages, relevant)

    assert result["hit"] == 1
    assert abs(result["reciprocal_rank"] - 1 / 3) < 1e-9
    assert result["recall"] == 1.0