from app.ingestion.loader import load_jsonl

file_path = "data/raw/krv/krv_holybible.jsonl"

documents = load_jsonl(file_path)

print(f"전체 구절 수 : {len(documents)}")
print(f"첫 번째 구절 : {documents[0]}")
print(f"마지막 구절 : {documents[-1]}")