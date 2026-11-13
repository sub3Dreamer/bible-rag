# Bible RAG

**로컬 임베딩과 RAG를 활용한 성경·신학 문서 검색 및 지능형 질의응답 시스템**

성경 본문을 기반으로 자연어 검색과 근거 중심의 질의응답을 제공하기 위한 개인 개발 프로젝트입니다.

현재는 한국어 개역한글판(KRV) 성경을 대상으로 데이터 전처리, 벡터 검색 및 검색 품질 평가 기능을 구현했습니다.

향후 로컬 LLM을 활용한 답변 생성, Hybrid Search, 다중 역본 및 신학 문서 검색 기능을 추가할 예정입니다.

## 1. 프로젝트 목표

- **의미 기반 검색:** 사용자의 질문과 의미적으로 관련된 성경 구절 검색
- **검색 품질 개선:** 임베딩 모델별 검색 성능을 정량적으로 평가하고 개선
- **근거 중심 답변:** 검색된 성경 구절을 근거로 LLM 답변 생성 (예정)
- **로컬 중심 아키텍처:** 문서 원문과 검색 컨텍스트의 외부 API 전송 최소화 및 통제
- **확장 가능한 구조:** 다중 성경 역본과 신학 문서 검색 지원 (예정)

## 2. 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python 3.12 |
| RAG Framework | LangChain |
| Embedding | multilingual-e5-small |
| Vector Database | ChromaDB |
| Relational Database | SQLite |
| Data Processing | JSONL |
| Testing | pytest |
| Development Environment | VSCode, Conda |
| Version Control | Git, GitHub |

현재 검색 파이프라인은 로컬 임베딩 모델을 사용하며, 기존 OpenAI 임베딩은 성능 비교를 위한 실험 코드로 유지하고 있습니다.

## 3. 시스템 아키텍처

### 데이터 인덱싱

```text
KRV 성경 원본 (JSONL)
         |
         v
   데이터 정규화
         |
         +------> SQLite
         |        (구절 단위 저장)
         |
         v
     청크 생성
   (5절 단위 분할)
         |
         v
  Local E5 Embedding
         |
         v
      ChromaDB
   (벡터 인덱스 저장)
```

### 검색 파이프라인

```text
사용자 자연어 질문
         |
         v
  Local E5 Embedding
         |
         v
  ChromaDB 유사도 검색
         |
         v
   관련 성경 청크 검색
         |
         v
  검색 결과 및 구절 정보
```

현재는 Dense Retrieval 기반의 의미 검색을 구현했습니다.

BM25 키워드 검색, RRF 기반 Hybrid Search 및 로컬 LLM 답변 생성은 향후 추가할 예정입니다.

## 4. 주요 구현 기능

### 4.1 성경 데이터 전처리

- 한국어 개역한글판(KRV) 성경 66권 처리
- 총 31,101개 구절 정규화
- 책, 장, 절, 역본 등의 메타데이터 구성
- 중복 구절 및 데이터 유효성 검증
- SQLite 기반 구절 단위 저장

### 4.2 청크 생성 및 인덱싱

- 동일한 책과 장 범위 내에서 5절 단위로 청크 생성
- 총 6,703개 청크 생성
- 청크별 고유 ID 및 메타데이터 관리
- ChromaDB를 활용한 벡터 인덱싱
- 기존 청크 ID 확인을 통한 중복 인덱싱 방지

### 4.3 의미 기반 검색

- LangChain과 ChromaDB를 활용한 Dense Retrieval
- 다국어 임베딩 모델 `intfloat/multilingual-e5-small` 적용
- 질문과 문서에 각각 E5 전용 접두어 적용
- 검색 결과의 성경 책·장·절 범위 및 거리 정보 제공
- 역본별 검색을 위한 메타데이터 필터링

### 4.4 검색 품질 평가

- 정답 구절 범위가 지정된 평가 질문 10개 구성
- Hit@K, MRR@K, Range Recall@K 측정
- OpenAI 임베딩과 로컬 E5 임베딩 성능 비교
- 검색 결과 순위와 정답 구절 범위의 중첩 여부 평가

## 5. 검색 성능 평가

동일한 10개 평가 질문과 Top-5 검색 조건에서 두 임베딩 모델의 성능을 비교했습니다.

| 평가 지표 | OpenAI Embedding | Local E5 |
|---|---:|---:|
| Hit@5 | 0.400 | **0.700** |
| MRR@5 | 0.187 | **0.508** |
| Range Recall@5 | 0.400 | **0.700** |

### 평가 결과

- Hit@5: **40% → 70%**
- MRR@5: **0.187 → 0.508**
- Range Recall@5: **40% → 70%**

로컬 E5 임베딩 적용 후, 평가 질문에서 정답 구절을 검색하는 비율과 정답 구절의 검색 순위가 개선되었습니다.

또한 임베딩 생성에 외부 OpenAI API를 사용하지 않는 로컬 검색 환경을 구축했습니다.

**평가 시 유의사항**

- 평가 질문이 10개인 초기 실험 결과로, 모델의 일반적인 우열을 의미하지 않습니다.
- Range Recall은 정답 구절 범위와 검색 청크의 중첩 여부를 기준으로 측정한 지표이며, 절 단위 Recall과는 구분됩니다.
- 향후 평가 질문을 확대하고 Hybrid Search 도입 전후의 성능을 비교할 예정입니다.

## 6. 프로젝트 실행 방법

### 개발 환경

- Python 3.12
- Conda
- Windows / VSCode

### 환경 구성

```bash
conda create -n bible-rag python=3.12
conda activate bible-rag
pip install -r requirements.txt
```

프로젝트 루트의 `.env.example`을 참고하여 `.env` 파일을 구성합니다.

### 데이터 준비

성경 원본 데이터와 생성된 SQLite·ChromaDB 파일은 저장소에 포함하지 않습니다.

원본 데이터는 아래에 명시된 데이터 출처에서 이용 조건을 확인한 후 별도로 준비해야 합니다.

데이터 전처리, 청크 생성, 인덱싱 및 평가에는 `scripts/` 디렉터리의 스크립트를 사용합니다.

```bash
# 데이터 정규화
python -m scripts.normalize_data

# 데이터 검증
python -m scripts.validate_data

# 청크 생성
python -m scripts.build_chunks

# 로컬 벡터 인덱스 생성
python -m scripts.build_local_vector_store

# 검색 성능 평가
python -m scripts.evaluate_local_retrieval
```

각 스크립트 실행 전에 필요한 입력 데이터가 해당 경로에 준비되어 있어야 합니다.

## 7. 보안 설계

본 프로젝트는 외부 LLM/API에 의존하지 않고 문서를 처리할 수 있는 구조를 목표로 합니다.

현재 적용한 설계 원칙은 다음과 같습니다.

- 로컬 임베딩 모델을 통한 문서 및 질문 임베딩
- 로컬 ChromaDB와 SQLite를 통한 데이터 저장
- 환경변수를 이용한 API 키 분리
- LangSmith 추적 비활성화
- `.gitignore`를 통한 원본 데이터, 로컬 DB 및 인증 정보의 Git 추적 제외

향후 로컬 LLM을 도입하고 외부 네트워크 송신 통제를 강화하여 문서 원문과 검색 컨텍스트가 외부 서비스로 전송되지 않도록 구성할 계획입니다.

단, 현재 저장소에는 기존 OpenAI 임베딩 실험 코드가 포함되어 있으므로 모든 실행 경로의 외부 통신이 차단된 상태를 의미하지는 않습니다.

## 8. 향후 개발 계획

- [x] 성경 데이터 정규화 및 검증
- [x] SQLite 기반 구절 저장
- [x] 청크 생성 및 ChromaDB 인덱싱
- [x] Dense Retrieval 구현
- [x] 로컬 E5 임베딩 적용
- [x] 검색 품질 평가 및 임베딩 모델 비교
- [ ] BM25 기반 키워드 검색
- [ ] RRF 기반 Hybrid Search
- [ ] 검색 품질 평가 데이터셋 확장
- [ ] 로컬 LLM 기반 답변 생성
- [ ] LangGraph 기반 Agent 기능
- [ ] FastAPI 기반 서비스 API
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

자세한 내용은 원본 저장소의 [LICENSE](https://github.com/crizin/bible-db/blob/main/LICENSE) 및 [NOTICE](https://github.com/crizin/bible-db/blob/main/NOTICE)를 참고하세요.

본 프로젝트 자체의 소스코드 라이선스는 별도로 관리합니다.