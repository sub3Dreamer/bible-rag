import json
from pathlib import Path

data_dir = Path("data/raw/krv")

files = list(data_dir.glob("*.jsonl"))

if not files:
    raise FileNotFoundError("KRV JSONL 파일을 찾을 수 없습니다.")

for file in files:
    print(f"파일명: {file.name}")

    with file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break

            record = json.loads(line)
            print(json.dumps(record, ensure_ascii=False, indent=2))