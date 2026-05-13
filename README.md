![우량애](assets/wooryangae_wordmark.png)

# spec — 3GPP LTE/NR Radio Specifications Q&A

3GPP **TS 36 (LTE)** 및 **TS 38 (NR)** Radio Access Network 핵심 사양 (Release 19) 을
로컬에 다운로드해 두고 **Claude Code 세션에서 직접 자연어로 질의응답**하는
환경입니다.

> ⚠️ **이 저장소는 public입니다.** API 키·사내 자료·개인정보를 절대 commit하지 마세요.
> 운영 방식상 외부 API 키는 필요하지 않습니다.

## 사용 흐름 — 1줄 요약

```bash
git clone https://github.com/GeneBaik9/spec.git  &&  cd spec
uv sync
uv run python scripts/download_specs.py      # 3GPP archive에서 .docx 34개 (5~10분)
uv run python scripts/download_pdfs.py       # ETSI에서 PDF 29개 (5~10분)
uv run python scripts/build_index.py         # INDEX.md 생성
claude                                       # Claude Code 세션 열고 그냥 자연어로 질문
```

세션을 열면 `CLAUDE.md`의 지침에 따라 Claude가
INDEX → PDF 목차 → 본문 페이지 범위를 순서대로 읽고 한국어로 답합니다.
출처(spec 번호·버전·페이지)는 답변마다 명시.

## 다운로드 대상 (34개, Rel-19 latest)

| WG | TS 36 (LTE) | TS 38 (NR) |
|---|---|---|
| RAN1 — 물리계층 | 36.211 / 212 / 213 / 214 | 38.211 / 212 / 213 / 214 / 215 |
| RAN2 — L2/L3 | 36.300 / 304 / 306 / 321 / 322 / 323 / 331 | 38.300 / 304 / 306 / 321 / 322 / 323 / 331 |
| RAN3 — 인터페이스 | — | 38.401 |
| RAN4 — 무선성능 | 36.101 / 104 / 133 | 38.101-1~5 / 104 / 133 |

추가/제거는 `config/specs.yaml`.

## 디렉터리 구조

```
spec/
├── CLAUDE.md                  # Claude Code 작업 지침 (Q&A 절차)
├── INDEX.md                   # 모든 spec의 파일 경로/페이지 수/제목 (자동 생성)
├── config/specs.yaml          # 다운로드 대상 spec 목록
├── scripts/
│   ├── download_specs.py      # 3GPP archive → specs/{raw,docx}/
│   ├── download_pdfs.py       # ETSI → specs/pdf/ (multipart은 libreoffice fallback)
│   ├── build_index.py         # INDEX.md 빌더
│   └── ingest.py              # (선택) Voyage+Chroma RAG 인덱싱
├── src/spec_qa/               # (선택) RAG 모듈 + spec-qa CLI
├── specs/                     # ⛔ .gitignore (대용량 PDF/docx)
└── tests/                     # pytest 102 pass / 2 skip
```

`specs/` 내부:
- `specs/pdf/*.pdf` — 주 검색 대상 (Read tool로 직접 읽힘)
- `specs/docx/*.docx` — 백업 + multipart 5개 (38.101-1~5; PDF 변환 못 한 것)
- `specs/raw/*.zip` — 3GPP archive 원본

## 시스템 요구사항

| 항목 | 버전 |
|---|---|
| Python | 3.12+ |
| uv | 최신 |
| poppler-utils (`pdfinfo`) | INDEX 빌더용 |
| libreoffice-core + **libreoffice-writer** | multipart spec docx → PDF 변환용 (선택) |

`libreoffice-writer`가 없으면 38.101-1~5 PDF 변환이 안 됩니다 (docx로 폴백).
설치: `sudo apt install libreoffice-writer`.

## 선택 — 외부 API 기반 RAG (옵션 B)

처음 설계엔 Voyage AI + Anthropic API 기반 RAG도 포함되어 있어
`uv run spec-qa ask "..."` 식 CLI를 쓸 수 있습니다. 사용하려면:
1. `.env`에 `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` 입력
2. `uv run python scripts/ingest.py`로 Chroma DB 구축
3. `uv run spec-qa ask "..."` 또는 `interactive`

본 PC만 사용한다면 이 옵션은 필요 없습니다.

## 비용

- 메인 워크플로 (Claude Code 직접) — **외부 API 호출 0**
- 옵션 B RAG — Voyage 임베딩 ~$5 1회, Claude 질의당 ~$0.02~0.10

## 라이선스

코드: MIT. 3GPP/ETSI 사양은 ETSI/3GPP 저작권이며 본 도구는 다운로드/검색만 합니다.
