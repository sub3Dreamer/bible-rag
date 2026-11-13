import json
from pathlib import Path

from app.ingestion.loader import load_jsonl
from app.ingestion.chunker import chunk_verses
from app.ingestion.validator import validate_verses

def main():
    input_path = Path("data/processed/krv.jsonl")
    output_path = Path("data/processed/krv_chunks.jsonl")

    records = load_jsonl(str(input_path))

    errors = validate_verses(records)
    if errors:
        raise ValueError(
            f"데이터 검증 실패: {errors[:5]}"
        )
    
    chunks = chunk_verses(records, chunk_size=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print(f"원본 구절 수: {len(records)}")
    print(f"생성된 청크 수: {len(chunks)}")
    print(f"첫 청크 ID: {chunks[0]['chunk_id']}")
    print(f"마지막 청크 ID: {chunks[-1]['chunk_id']}")

if __name__ == "__main__":
    main()