# 서울대학교 빅데이터 핀테크 전문가 과정 12기 — 자연어처리 실습 커리큘럼

현업 데이터(DART 감사보고서 HTML)를 직접 다루며 **HTML 파싱 → 재무 데이터 정제 → LLM 파인튜닝 → RAG·에이전트 구축**까지 E2E 파이프라인을 완성하는 실습 과정입니다.

---

## 교육 목표

- PyTorch 텐서 연산·자동미분·최적화 기본기 확립
- HTML DOM 파싱과 비정형 금융 문서 처리 역량 습득
- 감사보고서 재무제표 데이터 추출·정제 및 구조화 CSV/JSON 생성
- Qwen3 기반 QLoRA SFT로 재무 Q&A 모델 파인튜닝
- RAG·스트리밍·Tool Calling을 결합한 재무 분석 에이전트 구현

---

## 대상 및 요구사항

- 파이썬 기본 문법 및 머신러닝·딥러닝 기초 개념 보유자
- 개발 환경: Google Colab (GPU 권장), Python 3.10+, PyTorch 2.x
- MPS(Apple Silicon)는 일부 실습(`torchtext` 사용 구간)에서 미지원

---

## 사용 도구

| 구분 | 도구 |
|:---|:---|
| 딥러닝 | PyTorch, TorchText |
| 데이터 처리 | BeautifulSoup4, lxml, pandas |
| LLM 파인튜닝 | Hugging Face Transformers, PEFT (QLoRA), trl (SFT) |
| 검색·에이전트 | sentence-transformers, FAISS, TextIteratorStreamer |
| 모델 | Qwen/Qwen3-0.6B |

---

## 전체 일정

| 회차 | 일시 | 시간 | 주제 |
|:---:|---|:---:|---|
| 1 | 3월 17일 (화) | 14:00 – 18:00 | PyTorch 기초·텐서 연산·선형/로지스틱 회귀 |
| 2 | 3월 19일 (목) | 14:00 – 18:00 | HTML 파싱과 감사보고서 구조 분석 |
| 3 | 3월 20일 (금) | 09:00 – 13:00 | 재무제표 테이블 추출 및 데이터 정제 |
| 4 | 3월 24일 (화) | 14:00 – 18:00 | LLM 파인튜닝 — Qwen3 QLoRA SFT |
| 5 | 3월 26일 (수) | 14:00 – 18:00 | RAG · 스트리밍 생성 · 에이전트 설계 |
| 6 | 4월 7일 (월) | 14:00 – 18:00 | 최종 발표 및 프로젝트 고도화 |

---

## 세부 커리큘럼

### 1회차 — PyTorch 기초 및 텐서 연산
**파일**: `1회/실습_1회_파이토치기초.ipynb` · `1회/자연어처리와딥러닝/NLP_딥러닝_통합커리큘럼.ipynb`

- 텐서 생성·조작·브로드캐스팅·인덱싱
- 자동미분(`autograd`)과 역전파 원리
- 옵티마이저(SGD)·손실함수(MSE, BCE) 설계
- 선형 회귀·로지스틱 회귀 구현
- GPU/CUDA vs MPS(Apple Silicon) 환경 비교
- *(부록)* RNN·CNN 감성분석 → Seq2Seq → Attention → Transformer 통합 실습

---

### 2회차 — HTML 파싱과 감사보고서 구조 분석
**파일**: `2회/실습_2회_HTML파싱_기초_감사보고서.ipynb`

- HTML 태그·속성·DOM 트리 구조 이해
- BeautifulSoup 파서 3종 비교 (`lxml` / `html5lib` / `html.parser`)
- EUC-KR 인코딩 처리와 `\xa0` 정규화
- CSS 클래스 기반 섹션 레벨 판별 (`SECTION-N` 패턴)
- 말형된 `<h2><p>제목</p></h2>` 구조 우회 함수 구현
- 섹션 텍스트 수집 → `sections.json` 저장
- *(심화)* 주석 섹션 계층 구조 미리보기

> **산출물**: `output/processed/jsons/2024/sections.json`

---

### 3회차 — 재무제표 테이블 추출 및 데이터 정제
**파일**: `3회/실습_3회_HTML파싱_재무제표테이블_정제.ipynb`

- "제목 테이블 + 데이터 테이블" 패턴 식별 전략
- `pd.read_html()` 기반 테이블 파싱과 `MultiIndex` 처리
- 괄호형 음수·쉼표·하이픈 처리, 단위(백만원) 추출
- 5개 재무제표 개별 추출 및 CSV 저장
  - 재무상태표(BS) · 손익계산서(IS) · 포괄손익계산서(OCI)
  - 자본변동표(CSE) · 현금흐름표(CF)

> **산출물**: `output/processed/tables/2024/*.csv`

---

### 4회차 — LLM 파인튜닝 (Qwen3 QLoRA SFT)
**파일**: `4회/실습_4회_LLM파인튜닝_재무QA모델.ipynb`

- 토크나이저 원리 (BPE, 특별 토큰, Chat Template)
- Qwen3 Thinking / Non-thinking 모드와 샘플링 파라미터
- 감사보고서 CSV→ 재무 Q&A 데이터셋 자동 생성 (JSONL)
  - 직접 조회형·비율 계산형·YoY 변동형·텍스트 서술형
- 4-bit 양자화(`BitsAndBytesConfig`) + LoRA 어댑터 구성
- `SFTTrainer`(trl) 학습 루프·체크포인트 저장
- 파인튜닝 전/후 응답 품질 비교 추론
- 환경 자동 감지: Colab CUDA / Mac MPS / CPU 분기 설정

---

### 5회차 — RAG · 스트리밍 생성 · 에이전트 설계
**파일**: `5회/실습_5회_RAG_스트리밍_에이전트.ipynb`

- **RAG**: `paraphrase-multilingual-MiniLM-L12-v2` 임베딩 + FAISS 벡터 DB
- 문서 청킹 전략 (섹션 기반 + 단락 재분할·오버랩)
- **스트리밍 생성**: `TextIteratorStreamer` + 멀티스레딩 실시간 출력
- **Agentic 설계 (ReAct 패턴)**:
  - `get_financial_value` · `calculate_ratio` · `get_yoy_change` · `search_audit_report`
  - Thought → Action → Observation → Answer 루프
- 통합 재무 분석 챗봇 (`FinancialChatbot` 클래스)

---

### 6회차 — 최종 발표 및 프로젝트 고도화
**일시**: 4월 7일 (월) 14:00 – 18:00

- 팀별 최종 발표 및 데모
- E2E 파이프라인 (HTML 파싱 → 데이터 정제 → LLM·에이전트) 재현성 점검


---

## 데이터 파이프라인 전체 흐름

```
실습데이터셋/감사보고서_2024.htm  (DART HTML, EUC-KR)
        │
        ▼  [2회] BeautifulSoup HTML 파싱
output/processed/jsons/2024/
├── sections.json   ← 섹션별 텍스트 (5개 섹션)
└── notes.json      ← 주석 계층 트리
        │
        ▼  [3회] pandas 테이블 추출·정제
output/processed/tables/2024/
├── 2024_BS_재무상태표.csv
├── 2024_IS_손익계산서.csv
├── 2024_OCI_포괄손익계산서.csv
├── 2024_CSE_자본변동표.csv
└── 2024_CF_현금흐름표.csv
        │
        ▼  [4회] 재무 Q&A 데이터셋 생성 → Qwen3 QLoRA SFT
Qwen3-0.6B 파인튜닝 모델 (LoRA 어댑터)
        │
        ▼  [5회] RAG + 스트리밍 + Tool Calling
재무 분석 에이전트 챗봇
```

---

## 디렉토리 구조

```
12기핀테크실습/
├── README.md
├── 실습데이터셋/
│   └── 감사보고서_2024.htm
├── 1회/
│   ├── 실습_1회_파이토치기초.ipynb
│   └── 자연어처리와딥러닝/
│       └── NLP_딥러닝_통합커리큘럼.ipynb   ← RNN·CNN·Seq2Seq·Attention·Transformer
├── 2회/
│   └── 실습_2회_HTML파싱_기초_감사보고서.ipynb
├── 3회/
│   └── 실습_3회_HTML파싱_재무제표테이블_정제.ipynb
├── 4회/
│   └── 실습_4회_LLM파인튜닝_재무QA모델.ipynb
├── 5회/
│   └── 실습_5회_RAG_스트리밍_에이전트.ipynb
└── output/
    └── processed/
        ├── jsons/2024/
        │   ├── sections.json
        │   └── notes.json
        └── tables/2024/
            ├── 2024_BS_재무상태표.csv
            ├── 2024_IS_손익계산서.csv
            ├── 2024_OCI_포괄손익계산서.csv
            ├── 2024_CSE_자본변동표.csv
            └── 2024_CF_현금흐름표.csv
```

---

## 환경 설정

### Google Colab (권장)

각 실습 노트북 첫 번째 셀에서 필요한 패키지를 자동으로 설치합니다.  
런타임 유형을 **GPU(T4 이상)** 로 설정 후 실행하세요.

### 로컬 환경

```bash
pip install torch==2.2.2 torchtext==0.17.2 portalocker==3.1.1
pip install beautifulsoup4 lxml html5lib pandas
pip install transformers peft trl datasets accelerate bitsandbytes
pip install sentence-transformers faiss-cpu evaluate
pip install spacy && python -m spacy download en_core_web_sm de_core_news_sm
```

> **주의**: `bitsandbytes`(4-bit 양자화)는 CUDA GPU 전용입니다.  
> Mac MPS 환경에서는 `float16` 전체 정밀도 모드로 자동 전환됩니다.  
> `torchtext`는 MPS를 지원하지 않으므로 1회 NLP 통합 커리큘럼은 CUDA 또는 CPU에서 실행하세요.

---

## 문의

- 조교 장동준 (qwer4107@snu.ac.kr)
