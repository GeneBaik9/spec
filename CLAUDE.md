# CLAUDE.md — 3GPP RAN spec Q&A 작업 지침

이 디렉터리는 사용자 본인 PC에서 **Claude Code 세션 안에서 직접** 3GPP TS 36 (LTE) /
TS 38 (NR) RAN 핵심 사양에 대한 질의응답을 받기 위한 환경이다.
별도 RAG 인프라(Voyage, Anthropic API CLI)는 **사용하지 않는다** — 사용자가
이 디렉터리에서 Claude Code를 열고 자연어로 물으면 Claude(나)가 직접 답한다.

---

## 1. 디렉터리 레이아웃

| 경로 | 내용 |
|---|---|
| `specs/pdf/*.pdf` | **주 검색 대상**. ETSI 발행 PDF 29개 (TS 36/38, Rel-19). Read tool로 직접 읽힘. |
| `specs/docx/*.docx` | multipart 5개 (38.101-1~5) 백업. PDF 변환 안 됨 (`libreoffice-writer` 미설치). |
| `specs/raw/*.zip` | 3GPP archive 원본 zip. 일반적으론 안 본다. |
| `INDEX.md` | 전체 spec 목록 + 한 줄 설명 + 파일 경로. **사용자 질문 들어오면 여기부터 본다.** |
| `config/specs.yaml` | 다운로드 대상 목록 (참고용). |
| `src/spec_qa/parser.py` | docx → 섹션 청크 (multipart 5개 처리할 때만 필요). |

기존 RAG 인프라 (`embeddings.py`, `vectorstore.py`, `rag.py`, `cli.py`, `ingest.py`)는
남겨두긴 했지만 본 PC 사용에선 호출하지 않는다. 사용자가 향후 외부 API를 쓸
의향이 생기면 활성화 가능.

---

## 2. 사용자가 spec 질문을 했을 때 따라야 할 절차

### Step 1. 어떤 spec을 봐야 할지 판단
- `INDEX.md`를 먼저 Read 한다. 28~34개 spec 중 질문에 직접 관련된 것 1~3개 추림.
- 키워드 매핑 (참고):
  - **물리채널·변조** → 38.211 / 36.211
  - **채널 코딩** → 38.212 / 36.212
  - **물리계층 절차 (PDCCH/PDSCH/PUSCH/PUCCH/CSI/HARQ)** → 38.213, 38.214 / 36.213
  - **물리계층 측정 (RSRP/RSRQ/SINR)** → 38.215 / 36.214
  - **MAC** → 38.321 / 36.321
  - **RLC** → 38.322 / 36.322
  - **PDCP** → 38.323 / 36.323
  - **RRC (메시지·연결관리·재구성)** → 38.331 / 36.331
  - **NR Stage-2 / 전체 동작** → 38.300 / 36.300
  - **Idle/Inactive 절차·셀 선택·재선택** → 38.304 / 36.304
  - **UE radio capability** → 38.306 / 36.306
  - **NG-RAN 아키텍처·인터페이스 (Xn/F1/E1)** → 38.401
  - **UE radio 성능 (RF tx/rx, FR1/FR2)** → 38.101-1~5 / 36.101 / 36.104
  - **BS radio 성능** → 38.104 / 36.104
  - **RRM 요구사항 (측정 정확도·셀 선택 기준)** → 38.133 / 36.133

### Step 2. PDF 내 섹션 위치 찾기
- 첫 30~40 페이지에 보통 **목차**가 있다. `Read({pdf_path, pages: "1-30"})` 으로 목차 확보.
- 키워드로 매칭된 섹션 번호(예: "5.3.5.4")와 페이지 번호 메모.
- 큰 spec(38.331, 36.331, 38.300, 38.133 등 수백~천 페이지)은 한 번에 못 읽으니
  **목차 → 해당 페이지 범위만 다시 Read**. Read tool은 한 호출에 최대 20 페이지.

### Step 3. 본문 읽고 답변 작성
- 관련 페이지를 Read 하고 본문 인용해 한국어로 설명.
- 기술 용어(MIB/SIB, BWP, CORESET, RNTI, RACH preamble 등)는 **원문 그대로** 유지.
- 답변 끝에 **반드시 출처 명시**:
  ```
  📚 출처
  - TS 38.331 v19.2.0 §5.3.5 RRC Reconfiguration (pages 312–328)
  - TS 38.300 v19.2.0 §9.2.4 PDCP / RLC interaction (page 87)
  ```
- 컨텍스트에서 답을 못 찾으면 **"제공된 spec 범위에선 명시적 정의를 찾지 못했다"고 명시**.
  추측 금지. 가능한 인접 섹션 1~2개 안내해서 사용자가 직접 확인하도록.

### Step 4. multipart spec(38.101-1~5) 질문일 때
- PDF가 없으므로 `specs/docx/38.101-N-v...docx`를 직접 Read 시도 → docx는 Read 불가.
- 대신 `src/spec_qa/parser.py`의 `parse_spec_file()` 를 일회성 호출:
  ```bash
  uv run python -c "from spec_qa.parser import parse_spec_file; \
    chunks = parse_spec_file(Path('specs/docx/38.101-1-v19.5.0.docx')); \
    for c in chunks: ... "
  ```
  또는 임시로 텍스트 덤프 후 grep. 빈번히 묻는 spec이면 사용자에게
  `sudo apt install libreoffice-writer` 권유 + `uv run python scripts/download_pdfs.py --only 38.101-1`.

---

## 3. 응답 스타일 가이드

- **한국어 답변**, **기술 용어 원어**.
- 정량적 수치(timer 값, 임계값, MCS 번호 등)는 spec에서 본 그대로 인용. 외부 지식으로
  보강하지 않는다.
- 약어는 첫 등장 시 풀어쓴다: "BWP (Bandwidth Part)".
- 본문이 길면 마크다운 헤딩·표·번호 목록 활용. 단답이 자연스러우면 단답으로.
- spec 번호와 버전을 **항상** 표시 (예: "TS 38.331 v19.2.0 기준").

---

## 4. 메타 작업 (질의응답 외)

사용자가 인프라 작업을 요청하면 (예: "spec 추가해줘", "재다운로드해줘"):
- `config/specs.yaml` 수정
- `uv run python scripts/download_specs.py [--only ...]`
- `uv run python scripts/download_pdfs.py [--only ...]`
- 변경 후 `INDEX.md` 재생성 (`scripts/build_index.py` 있음)

테스트/lint:
- `uv run pytest -q` (현재 102 pass + 2 skip)
- `uv run ruff check .`

---

## 5. 절대 하지 말 것

- `.env` 파일에 실 API 키 commit. 어차피 본 PC 사용에선 키 자체가 불필요하지만
  실수로 push 가능. (.gitignore로 막혀있긴 함)
- public repo이므로 사용자 개인정보·사내 자료 commit 금지.
- PDF/docx 자체를 commit (`.gitignore`로 막혀있음 — 확인 후 push).
- 추측으로 spec 내용 만들어내기. 원문 인용이 어려우면 솔직히 "찾지 못함"이라고 말한다.
