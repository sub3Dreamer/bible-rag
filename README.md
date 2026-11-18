# Bible RAG

**로컬 임베딩과 RAG를 활용한 성경·신학 문서 검색 및 지능형 질의응답 시스템**

성경 본문을 기반으로 자연어 검색과 근거 중심 질의응답을 제공하기 위한 개인 개발 프로젝트입니다.

현재 한국어 개역한글판(KRV) 성경을 대상으로 데이터 전처리, Dense/BM25 Hybrid Search, 검색 품질 평가, Ollama 기반 로컬 LLM 답변 생성을 구현했습니다. 향후 다중 역본, 신학 문서, 서비스 API 및 Agent 기능으로 확장할 계획입니다.

## 1. 프로젝트 목표

- **의미·키워드 결합 검색:** Dense Retrieval과 BM25를 결합하여 관련 성경 구절 검색
- **검색 품질 검증:** 평가 데이터셋과 정량 지표를 통한 검색 성능 및 안정성 측정
- **근거 중심 답변:** 검색된 성경 구절을 컨텍스트로 활용한 로컬 LLM 답변 생성
- **로컬 중심 아키텍처:** 문서 원문과 검색 컨텍스트의 외부 LLM/API 전송을 피하도록 실행 경로 구성
- **확장 가능한 구조:** 다중 성경 역본 및 신학 문서 검색 지원(예정)

## 2. 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python 3.12 |
| RAG Framework | LangChain, LCEL |
| Embedding | `intfloat/multilingual-e5-small` (로컬, 384차원) |
| Vector Database | ChromaDB 1.5.9 (HNSW) |
| Keyword Retrieval | BM25 (`rank-bm25`) |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Local LLM | Ollama, `qwen2.5:3b` |
| Relational Database | SQLite |
| Data Processing | JSONL |
| Testing | pytest 및 검색 평가 스크립트 |
| Development Environment | Windows, VSCode, Conda |
| Version Control | Git, GitHub |

기본 검색 및 답변 생성 경로는 로컬 임베딩과 로컬 LLM을 사용합니다. OpenAI 임베딩 관련 코드는 비교 실험용으로 별도 유지하고 있습니다.

## 3. 시스템 아키텍처

### 3.1 데이터 인덱싱

```text
KRV 성경 원본 (JSONL)
         |
         v
   데이터 정규화
         |
         +------> SQLite (구절 단위 저장)
         |
         v
   청크 생성 (동일 장 내 5절 단위)
         |
         +------> BM25 검색용 청크 데이터
         |
         v
  Local E5 Embedding
         |
         v
      ChromaDB
   (벡터 인덱스 저장)
```

### 3.2 Hybrid Search 및 답변 생성

```text
사용자 자연어 질문
         |
         +-----------------------+
         |                       |
         v                       v
 Local E5 Embedding         BM25 키워드 검색
         |                       |
         v                       |
 ChromaDB Dense Search           |
         |                       |
         +-----------+-----------+
                     |
                     v
              RRF 순위 결합
                     |
                     v
               Top-K 청크
                     |
                     v
          Context + 질문 + Prompt
                     |
                     v
             Ollama 로컬 LLM
                (Qwen2.5 3B)
                     |
                     v
               자연어 답변
                     |
                     v
    BibleRAG 서비스 응답 (answer / sources /
               retrieved_chunks)
```

검색은 `HybridRetriever`, 생성은 `BibleGenerator`, 전체 실행 흐름은 `BibleRAG` 서비스가 담당합니다. 검색된 컨텍스트는 매 요청 시 LLM에 참고 자료로 전달되며, 모델 가중치를 변경하는 학습 과정은 아닙니다.

## 4. 주요 구현 기능

### 4.1 성경 데이터 전처리

- 한국어 개역한글판(KRV) 성경 66권 처리
- 총 31,101개 구절 정규화
- 책·장·절·역본 메타데이터 구성
- 중복 구절 및 데이터 유효성 검증
- SQLite 기반 구절 단위 저장

### 4.2 청크 생성 및 인덱싱

- 동일한 책과 장 범위 내에서 5절 단위 청크 생성
- 총 6,703개 청크 생성
- 청크별 고유 ID 및 메타데이터 관리
- ChromaDB를 활용한 로컬 벡터 인덱싱
- 기존 청크 ID 확인을 통한 중복 인덱싱 방지

### 4.3 Dense Retrieval

- LangChain 및 ChromaDB 기반 의미 검색
- `intfloat/multilingual-e5-small` 로컬 임베딩 적용
- 질문에 `query: `, 문서에 `passage: ` 접두어 적용
- 임베딩 벡터 정규화 및 검색 결과 거리 정보 제공
- 역본별 검색을 위한 메타데이터 필터링

### 4.4 BM25 및 Hybrid Search

- `rank-bm25` 기반 키워드 검색
- Dense Search와 BM25의 후보 결과를 RRF로 결합
- 검색 결과를 `chunk_id` 기준으로 병합하고 순위 기반 점수 산정
- 최종 Top-K 청크 반환

### 4.5 로컬 LLM 답변 생성

- Ollama에서 실행하는 `qwen2.5:3b` 모델 사용
- LangChain `ChatPromptTemplate` 및 `ChatOllama` 연동
- LCEL(`prompt | llm`) 기반 생성 파이프라인
- 검색된 성경 구절을 컨텍스트로 전달하여 한국어 답변 생성
- 근거가 부족할 때 임의 답변을 피하도록 프롬프트 규칙 구성

### 4.6 RAG 서비스 통합

- `BibleRAG.ask(question, translation_id="krv", k=5)` 인터페이스
- 검색·컨텍스트 구성·답변 생성을 하나의 서비스 흐름으로 통합
- `question`, `answer`, `sources`, `retrieved_chunks`를 포함한 구조화된 결과 반환
- **현재 `sources`는 실제 답변에 사용된 근거로 검증된 목록이 아니라 검색된 후보 목록**이며, 정밀한 인용 검증은 향후 개선 예정

### 4.7 검색 품질 평가 및 안정화

- 정답 구절 범위가 지정된 평가 질문 10개
- Hit@K, MRR@K, Range Recall@K 측정
- OpenAI 임베딩, Local E5, BM25, Hybrid Search 비교
- 임베딩 출력 및 Exact Search와 ChromaDB ANN 검색 결과 비교
- HNSW 설정을 조정한 별도 실험 컬렉션 구축 및 반복 평가

## 5. 검색 성능 평가

동일한 10개 평가 질문과 Top-5 조건에서 측정한 결과입니다.

| 평가 지표 | OpenAI Dense | Local E5 Dense | BM25 | Hybrid (E5 + BM25 + RRF) |
|---|---:|---:|---:|---:|
| Hit@5 | 0.400 | 0.700 | 0.600 | **0.800** |
| MRR@5 | 0.187 | 0.508 | 0.433 | **0.700** |
| Range Recall@5 | 0.400 | 0.700 | 0.600 | **0.750** |

Hybrid 수치는 HNSW `ef_search=500` 실험 컬렉션에서 측정했으며, 동일한 평가를 5회 반복했을 때 모두 같은 수치를 기록했습니다.

### 5.1 ChromaDB ANN 검색 안정화 실험

기존 컬렉션(`ef_search=100`)에서는 동일한 질문·임베딩으로 독립 프로세스에서 검색할 때 일부 Top-K 결과가 달라지는 현상이 관찰됐습니다. 반면 전체 6,703개 벡터에 대한 Exact Search 결과는 일정했습니다.

- 사용 버전: ChromaDB 1.5.9
- 기존 컬렉션: `bible_krv_e5_small` (`space=l2`, `ef_search=100`)
- 실험 컬렉션: `bible_krv_e5_small_ef500` (`space=l2`, `ef_search=500`)
- 실험 컬렉션의 대표 질문 검색: 5회 모두 기대 구절이 1위
- Hybrid 평가: 5회 모두 동일한 지표 기록

**주의:** 실험 컬렉션은 HNSW 인덱스를 새로 구축했으므로, 개선 원인이 `ef_search` 증가인지 인덱스 재구축인지 단정할 수 없습니다. 더 긴 반복 실험 및 검색 지연 시간 측정이 필요합니다.

### 5.2 평가 시 유의사항

- 질문 10개로 구성된 초기 평가이며 일반적인 모델 우열을 증명하지 않습니다.
- Range Recall은 정답 구절 범위와 검색 청크의 중첩을 기준으로 측정하며 절 단위 Recall과 다릅니다.
- 생성 답변의 충실성·정확성에 대한 정량 평가는 아직 별도로 수행하지 않았습니다.
- 향후 질문 수를 늘리고 다양한 질문 유형에서 성능을 검증할 예정입니다.

## 6. 프로젝트 실행 방법

### 6.1 개발 환경 구성

```bash
conda create -n bible-rag python=3.12
conda activate bible-rag
python -m pip install -r requirements.txt
```

프로젝트 루트의 `.env.example`을 참고하여 `.env`를 구성합니다. 로컬 경로로 실행하더라도 외부 추적 설정과 API 사용 여부를 확인해야 합니다.

### 6.2 데이터 준비 및 인덱싱

성경 원본 데이터와 생성된 SQLite·ChromaDB 파일은 저장소에 포함하지 않습니다. 아래 데이터 출처와 이용 조건을 확인한 후 원본 데이터를 별도로 준비해야 합니다.

```bash
# 데이터 정규화 및 검증
python -m scripts.normalize_data
python -m scripts.validate_data

# 청크 생성 및 로컬 벡터 인덱스 구축
python -m scripts.build_chunks
python -m scripts.build_local_vector_store

# 로컬 Dense 검색 평가
python -m scripts.evaluate_local_retrieval
```

각 스크립트 실행 전에 필요한 입력 데이터가 해당 경로에 준비되어 있어야 합니다.

### 6.3 Ollama 모델 준비

[Ollama 공식 사이트](https://ollama.com/download)에서 Ollama를 설치한 후 다음을 실행합니다.

```bash
ollama pull qwen2.5:3b
ollama list
```

기본 Ollama 서비스 주소는 `http://127.0.0.1:11434`입니다.

### 6.4 RAG 답변 생성 테스트

Hybrid 검색기에서 사용하는 ChromaDB 컬렉션이 실제로 준비되어 있는지 확인합니다. 현재 안정화 실험에 사용한 컬렉션 이름은 `bible_krv_e5_small_ef500`입니다. 이 컬렉션은 원본 데이터만 내려받는다고 자동 생성되는 것은 아닙니다.

```bash
# 검색 결과를 컨텍스트로 전달해 로컬 LLM 답변 생성
python -m scripts.test_rag_generation

# BibleRAG 서비스 통합 테스트
python -m scripts.test_rag_service
```

예시 질문: `성령의 열매에는 어떤 것들이 있나요?`

예시 답변: `성령의 열매에는 사랑, 희락, 화평, 오래 참음, 자비, 양선, 충성, 온유, 절제가 있습니다.`

서비스 호출 예시:

```python
from app.rag.service import BibleRAG

rag = BibleRAG()
result = rag.ask("성령의 열매에는 어떤 것들이 있나요?")

print(result["answer"])
print(result["sources"])
```

※ 위 예시는 `app/rag/service.py`가 작성되어 있고, 로컬 인덱스 및 Ollama 모델이 준비된 환경을 전제로 합니다.

## 7. 보안 설계

본 프로젝트는 기본 RAG 경로에서 문서 원문과 검색 컨텍스트를 외부 LLM/API로 전송하지 않는 로컬 중심 구조를 지향합니다.

- 로컬 E5 모델을 통한 문서·질문 임베딩
- 로컬 ChromaDB 및 SQLite 저장
- Ollama 로컬 LLM을 통한 답변 생성
- 환경변수를 통한 API 키 관리 및 LangSmith 추적 비활성화
- `.gitignore`를 통한 원본 데이터·로컬 DB·인증 정보의 Git 추적 제외

다만 모델 최초 다운로드에는 네트워크 연결이 필요할 수 있으며, 비교 실험용 OpenAI 코드가 저장소에 남아 있습니다. **모든 실행 경로에서 외부 통신이 기술적으로 차단되었다는 뜻은 아닙니다.** 실제 배포 시에는 외부 송신 제한, 의존성·추적 설정 점검 및 네트워크 수준 검증이 필요합니다.

## 8. 향후 개발 계획

- [x] 성경 데이터 정규화 및 검증
- [x] SQLite 기반 구절 저장
- [x] 청크 생성 및 ChromaDB 인덱싱
- [x] Dense Retrieval 구현
- [x] 로컬 E5 임베딩 적용
- [x] 검색 품질 평가 및 임베딩 모델 비교
- [x] BM25 기반 키워드 검색
- [x] RRF 기반 Hybrid Search
- [x] ChromaDB ANN 검색 안정화 실험
- [x] Ollama 기반 로컬 LLM 답변 생성
- [x] 검색·생성 서비스 통합 코드 구성
- [ ] 검색 품질 평가 데이터셋 확장
- [ ] 답변 근거 구절 식별 및 인용 검증
- [ ] 생성 답변의 충실성·정확성 평가
- [ ] FastAPI 기반 서비스 API
- [ ] LangGraph 기반 Agent 기능
- [ ] 다중 성경 역본 및 신학 문서 지원

## 9. 데이터 출처 및 라이선스

본 프로젝트는 성경 본문 기반 RAG(Retrieval-Augmented Generation) 시스템을 구현하기 위해 공개된 성경 데이터셋을 활용합니다.

### 9.1 한국어 성경 (KRV)

- **역본:** 개역한글판 (1961)
- **원본 제공 기관:** 대한성서공회
- **데이터셋 제공:** [crizin/bible-db](https://github.com/crizin/bible-db)
- **사용 파일:** `data/raw/krv/krv_holybible.jsonl`
- **저작권 및 이용 조건:** 원문 저작권 및 재배포 가능 여부는 권리자의 조건과 관련 법령을 확인하여 적용
- **사용 목적:** 성경 본문 검색, 임베딩 생성 및 RAG 기반 질의응답

본 프로젝트는 성경 본문을 AI로 임의 수정하지 않으며, 데이터 정규화 과정에서도 원문 텍스트를 보존하는 것을 원칙으로 합니다.

### 9.2 영어 성경 (KJV) — 추가 예정

- **역본:** King James Version (1769)
- **데이터 출처:** [crizin/bible-db](https://github.com/crizin/bible-db)
- **저작권 및 이용 조건:** 국가별 이용 조건 확인 후 적용
- **사용 목적:** 다국어 검색 및 역본 비교 기능

### 9.3 데이터셋 및 소스코드 라이선스

`crizin/bible-db` 저장소의 수집·변환 스크립트와 성경 본문 데이터는 각각 별도의 이용 조건이 적용될 수 있습니다.

자세한 내용은 원본 저장소의 [LICENSE](https://github.com/crizin/bible-db/blob/main/LICENSE) 및 [NOTICE](https://github.com/crizin/bible-db/blob/main/NOTICE)를 참고하세요. 실제 이용 및 재배포 전에 해당 파일의 존재 여부와 적용 범위를 직접 확인해야 합니다.

본 프로젝트 자체의 소스코드 라이선스는 별도로 관리합니다.
