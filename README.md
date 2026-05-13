![우량애](assets/wooryangae_wordmark.png)

# spec — 3GPP LTE/NR Radio Specifications Q&A

3GPP **TS 36 (LTE)** 및 **TS 38 (NR)** Radio Access Network 핵심 사양을
로컬에 다운로드하고, Claude API 기반 **RAG**로 자연어 질의응답하는 도구입니다.

> ⚠️ **이 저장소는 public입니다.** API 키, 사내 자료, 개인정보 등을 절대 commit하지 마세요.
> 시크릿은 모두 `.env`(gitignored)에 두고, `.env.example`만 공유합니다.

## 주요 기능
- 3GPP 공식 FTP에서 RAN 핵심 spec 최신 버전 자동 다운로드 (TS 36/38 시리즈, ~28개)
- `.docx` → 섹션 단위 청킹 (3GPP 번호체계 메타데이터 보존)
- Voyage AI `voyage-3-large` 임베딩 + Chroma 로컬 벡터DB
- Claude Sonnet 4.6/Opus 4.7 기반 답변 + 출처 인용 + prompt caching

## 빠른 시작

### 1. 환경 준비
```bash
git clone https://github.com/GeneBaik9/spec.git
cd spec
uv sync
cp .env.example .env
# .env 파일을 열어 ANTHROPIC_API_KEY, VOYAGE_API_KEY 입력
```

### 2. Spec 다운로드 (시간 소요 — 수 GB)
```bash
uv run scripts/download_specs.py            # config/specs.yaml의 모든 spec
uv run scripts/download_specs.py --only 38.331   # 특정 spec만
```

### 3. 벡터DB 구축
```bash
uv run scripts/ingest.py                    # 다운로드된 모든 spec ingest
```

### 4. 질의응답
```bash
uv run spec-qa "NR PDCCH의 monitoring occasion 결정 방식은?"
uv run spec-qa --interactive                # 대화형 모드
```

## 디렉터리 구조
```
spec/
├── config/specs.yaml          # 다운로드 대상 spec 목록 (수정 가능)
├── scripts/
│   ├── download_specs.py      # 3GPP FTP → specs/
│   └── ingest.py              # docx → 청크 → 임베딩 → Chroma
├── src/spec_qa/
│   ├── parser.py              # docx 섹션 청킹
│   ├── embeddings.py          # Voyage AI 클라이언트
│   ├── vectorstore.py         # Chroma wrapper
│   ├── rag.py                 # Retrieval + Claude 답변 생성
│   └── cli.py                 # `spec-qa` 명령
├── specs/                     # ⛔ .gitignore (대용량 docx)
└── chroma_db/                 # ⛔ .gitignore (벡터 인덱스)
```

## 다운로드 대상 (Rel-18 latest, RAN 핵심 ~28개)

| WG | TS 36 (LTE) | TS 38 (NR) |
|---|---|---|
| RAN1 물리계층 | 36.211/212/213/214 | 38.211/212/213/214/215 |
| RAN2 L2/L3 | 36.300/304/306/321/322/323/331 | 38.300/304/306/321/322/323/331 |
| RAN3 아키텍처 | — | 38.401 |
| RAN4 무선성능 | 36.101/104/133 | 38.101-1~5/104/133 |

추가/제거는 `config/specs.yaml` 편집.

## 비용 안내
- **Voyage AI**: 신규 가입 시 200M tokens 무료. 28개 spec 전체 임베딩은 약 5~10M tokens 예상.
- **Claude API**: 질의당 평균 5~30K input tokens (prompt caching으로 절감). 일반 사용 시 질의당 약 $0.02~0.10.

## 라이선스
MIT. 3GPP 사양 자체는 ETSI/3GPP 저작권이며 본 도구는 다운로드/검색만 수행합니다.
