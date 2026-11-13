import json
from pathlib import Path


def load_jsonl(file_path: str) -> list[dict]:
    """
    JSONL 파일을 읽어 각 레코드를 dict로 반환한다.
    
    구현사항:
    1. 파일을 UTF-8로 읽는다.
    2. 한 줄씩 JSON으로 파싱한다.
    3. 빈 줄은 무시한다.
    4. 파싱된 dict를 리스트에 저장한다.
    5. 리스트를 반환한다.
    """
    documents = []
    
    path = Path(file_path)
    
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():    # 빈 줄은 건너뛴다.
                continue

            record = json.loads(line)   # JSON 형식의 문자열을 dict로 변환함.
            documents.append(record)

    return documents