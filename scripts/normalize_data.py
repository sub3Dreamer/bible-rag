import json
from pathlib import Path

from app.ingestion.loader import load_jsonl
from app.ingestion.normalizer import normalize_verse

def main():
    input_path = Path("data/raw/krv/krv_holybible.jsonl")
    output_path = Path("data/processed/krv.jsonl")

    # 원본 JSONL 로딩
    records = load_jsonl(str(input_path))

    # 레코드 정규화
    normalized = [
        normalize_verse(record, "krv")
        for record in records
    ]

    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSONL 파일 저장
    with output_path.open("w", encoding="utf-8") as file:
        for record in normalized:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"원본 구절 수 : {len(records)}")
    print(f"정규화 구절 수 : {len(normalized)}")
    print(f"저장 경로 : {output_path}")

if __name__ == "__main__":
    main()