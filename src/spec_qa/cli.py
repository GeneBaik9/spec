"""Typer CLI for 3GPP TS 36/38 radio specs Q&A.

Entry point: ``spec-qa`` (registered in pyproject.toml).

Commands:
  ask         Single question answer
  interactive Interactive REPL (multi-turn)
  info        Vector DB status summary
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from spec_qa.rag import Citation, RagAnswer, SpecRAG, estimate_cost
from spec_qa.vectorstore import ChromaSpecStore

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    help="3GPP TS 36/38 radio specs Q&A",
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True, style="bold red")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _print_answer(result: RagAnswer, show_chunks: bool = False) -> None:
    """Render a RagAnswer to the terminal using rich formatting."""
    console.print()
    console.print(Panel(f"[bold cyan]{result.question}[/bold cyan]", title="질문", expand=False))

    # Answer body
    console.print()
    console.print(Rule("답변", style="green"))
    console.print(result.answer)
    console.print()

    # Citations
    if result.citations:
        console.print(Rule("📚 출처", style="yellow"))
        for idx, cit in enumerate(result.citations, 1):
            _print_citation(idx, cit)
        console.print()

    # Retrieved chunks (debug mode)
    if show_chunks:
        console.print(Rule("🔍 검색된 청크", style="dim"))
        for idx, chunk in enumerate(result.retrieved_chunks, 1):
            console.print(
                f"  [{idx}] {chunk.metadata.get('spec_no')} §{chunk.metadata.get('section_no')} "
                f"(dist={chunk.distance:.4f}): {chunk.text[:120]}..."
            )
        console.print()

    # Token usage + cost
    _print_usage(result.usage)


def _print_citation(idx: int, cit: Citation) -> None:
    """Print a single citation entry."""
    header = Text()
    header.append(f"  [{idx}] ", style="bold yellow")
    header.append(f"TS {cit.spec_no} v{cit.version} §{cit.section_no}  ", style="bold")
    header.append(cit.section_title, style="italic")
    console.print(header)

    # Show up to 200 chars of the cited text, indented
    preview = cit.cited_text[:200].replace("\n", " ")
    if len(cit.cited_text) > 200:
        preview += "…"
    console.print(f'      "{preview}"', style="dim")


def _print_usage(usage: dict, prefix: str = "") -> None:
    """Print token counts and approximate cost."""
    cost = estimate_cost(usage)
    console.print(
        f"{prefix}[dim]📊 토큰: "
        f"input={usage.get('input_tokens', 0):,}  "
        f"output={usage.get('output_tokens', 0):,}  "
        f"cached_read={usage.get('cache_read_input_tokens', 0):,}  "
        f"cost≈${cost:.4f}[/dim]"
    )


def _make_rag() -> SpecRAG:
    """Construct SpecRAG; exits with a helpful message on missing API key."""
    try:
        return SpecRAG()
    except ValueError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="질문 (큰따옴표로 묶어서 입력)")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="검색 chunk 수")] = 8,
    spec: Annotated[
        list[str],
        typer.Option("--spec", help="검색 대상 spec 제한 (반복 가능). e.g. --spec 38.331"),
    ] = [],
    show_chunks: Annotated[
        bool, typer.Option("--show-chunks", help="검색된 chunk 미리보기 출력")
    ] = False,
) -> None:
    """단일 질문 답변."""
    rag = _make_rag()
    spec_filter = list(spec) if spec else None

    with console.status("[bold green]검색 및 답변 생성 중…[/bold green]"):
        result = rag.answer(question, top_k=top_k, spec_filter=spec_filter)

    _print_answer(result, show_chunks=show_chunks)


@app.command()
def interactive() -> None:
    """대화형 REPL. 빈 입력 또는 'exit'으로 종료."""
    console.print(Panel("[bold]3GPP Spec Q&A — 대화형 모드[/bold]\n빈 입력 또는 'exit' 입력 시 종료."))

    rag = _make_rag()

    total_usage: dict = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    turn = 0

    while True:
        try:
            question = typer.prompt("\n질문")
        except (KeyboardInterrupt, EOFError):
            break

        question = question.strip()
        if not question or question.lower() in {"exit", "quit", "q"}:
            break

        turn += 1
        with console.status("[bold green]검색 및 답변 생성 중…[/bold green]"):
            result = rag.answer(question)

        _print_answer(result)

        # Accumulate usage
        for key in total_usage:
            total_usage[key] += result.usage.get(key, 0)

    if turn > 0:
        console.print()
        console.print(Rule("세션 요약", style="bold blue"))
        console.print(f"  총 질문 수: {turn}")
        _print_usage(total_usage, prefix="  누적 ")
    console.print("[dim]종료[/dim]")


@app.command()
def info() -> None:
    """벡터DB 상태 확인 — 저장된 청크 수와 spec 분포."""
    try:
        store = ChromaSpecStore()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"ChromaDB 초기화 실패: {exc}")
        raise typer.Exit(code=1) from exc

    total = store.count()
    console.print(Panel(f"[bold]벡터 DB 정보[/bold]\n총 청크 수: [cyan]{total:,}[/cyan]"))

    if total == 0:
        console.print("[yellow]저장된 데이터가 없습니다.[/yellow]")
        return

    # Aggregate spec distribution by fetching all metadata
    # We use a dummy query with a tiny random vector just for metadata collection;
    # for a pure metadata scan we instead call the underlying collection directly.
    try:
        raw = store._collection.get(include=["metadatas"])  # type: ignore[attr-defined]
        metadatas = raw.get("metadatas", []) or []
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]메타데이터 읽기 실패: {exc}[/yellow]")
        return

    # Tally by spec_no
    spec_counts: dict[str, int] = {}
    version_map: dict[str, set[str]] = {}
    for meta in metadatas:
        spec_no = str(meta.get("spec_no", "unknown"))
        version = str(meta.get("version", ""))
        spec_counts[spec_no] = spec_counts.get(spec_no, 0) + 1
        version_map.setdefault(spec_no, set()).add(version)

    table = Table(title="Spec 분포", show_lines=False)
    table.add_column("Spec No", style="cyan", no_wrap=True)
    table.add_column("버전(s)", style="magenta")
    table.add_column("청크 수", style="green", justify="right")
    table.add_column("비율", justify="right")

    for spec_no in sorted(spec_counts):
        count = spec_counts[spec_no]
        versions = ", ".join(sorted(version_map.get(spec_no, set())))
        pct = count / total * 100
        table.add_row(spec_no, versions, f"{count:,}", f"{pct:.1f}%")

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
