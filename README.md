## 데이터 출처 및 라이선스

본 프로젝트는 성경 본문 기반 RAG(Retrieval-Augmented Generation) 시스템을 구현하기 위해 공개된 성경 데이터셋을 활용합니다.

### 1. 한국어 성경 (KRV)

- **역본:** 개역한글판 (1961)
- **원본 제공 기관:** 대한성서공회
- **데이터셋 제공:** [crizin/bible-db](https://github.com/crizin/bible-db)
- **사용 파일:** `data/krv/krv_holybible.jsonl`
- **라이선스:** 대한민국에서 저작재산권 보호기간이 만료된 역본으로, 원문과 출처를 존중하여 사용
- **사용 목적:** 성경 본문 검색, 임베딩 생성 및 RAG 기반 질의응답

본 프로젝트는 성경 본문을 AI로 임의 수정하지 않으며, 데이터 정규화 과정에서도 원문 텍스트를 보존하는 것을 원칙으로 합니다.

### 2. 영어 성경 (KJV) — 추가 예정

- **역본:** King James Version (1769)
- **데이터 출처:** crizin/bible-db
- **라이선스:** 국가별 이용 조건 확인 후 적용
- **사용 목적:** 다국어 검색 및 역본 비교 기능

### 3. 데이터셋 및 소스코드 라이선스

`crizin/bible-db` 저장소의 수집·변환 스크립트에는 MIT 라이선스가 적용되며, 성경 본문과 기타 데이터에는 각 원본의 별도 이용 조건이 적용됩니다.

자세한 내용은 원본 저장소의 [LICENSE](https://github.com/crizin/bible-db/blob/main/LICENSE) 및 [NOTICE](https://github.com/crizin/bible-db/blob/main/NOTICE)를 참고하세요.

본 프로젝트 자체의 소스코드 라이선스는 별도로 관리합니다.