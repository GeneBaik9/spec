"""Ingest 3GPP spec docx files into the Chroma vector store.

Scans a directory for ``{spec_no}-v{ver}.docx`` files, parses them into
chunks, embeds with Voyage AI, and upserts into ChromaDB.

Usage examples::

    python scripts/ingest.py                            # all docx in specs/docx/
    python scripts/ingest.py --only 38.331,38.211       # specific specs
    python scripts/ingest.py --reset                    # clear collection first
    python scripts/ingest.py --dry-run                  # parse only, no embed/upsert
    python scripts/ingest.py --db-path ./chroma_db
    python scripts/ingest.py --docx-dir specs/docx
    python scripts/ingest.py --batch-size 64
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Ensure the project src is importable when run as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from spec_qa.embeddings import VoyageEmbedder  # noqa: E402
from spec_qa.parser import Chunk, parse_spec_file  # noqa: E402
from spec_qa.vectorstore import ChromaSpecStore  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOYAGE_COST_PER_1M = 0.18  # USD per 1M tokens — voyage-3-large

console = Console()

app = typer.Typer(
    name="ingest",
    help="Ingest 3GPP docx files into Chroma vector store.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def scan_docx_dir(docx_dir: Path) -> list[Path]:
    """Return all .docx files in *docx_dir* matching the expected filename pattern.

    Pattern: ``{spec_no}-v{major}.{minor}.{patch}.docx``
    e.g. ``38.331-v18.5.0.docx``, ``38.101-1-v19.5.0.docx``
    """
    _RE_DOCX = re.compile(r"^[\d.]+-(?:\d+-)?v\d+\.\d+\.\d+\.docx$")
    files = sorted(
        p for p in docx_dir.glob("*.docx") if _RE_DOCX.match(p.name)
    )
    return files


def _spec_no_from_path(docx_path: Path) -> str:
    """Extract spec number from a filename, e.g. '38.331-v18.5.0.docx' → '38.331'."""
    stem = docx_path.stem  # "38.331-v18.5.0"
    m = re.search(r"-v\d+\.\d+\.\d+$", stem)
    if m:
        return stem[: m.start()]
    return stem


def parse_all(
    docx_paths: list[Path],
    only: list[str] | None = None,
) -> tuple[list[Chunk], list[str]]:
    """Parse all docx files and return (all_chunks, failed_spec_nos).

    Parameters
    ----------
    docx_paths:
        List of .docx file paths to parse.
    only:
        If provided, skip files whose spec_no is not in this list.

    Returns
    -------
    (chunks, failed)
        ``chunks`` is the flat list of all successfully parsed Chunk objects.
        ``failed`` is a list of filenames that raised an exception.
    """
    all_chunks: list[Chunk] = []
    failed: list[str] = []

    for path in docx_paths:
        spec_no = _spec_no_from_path(path)

        # --only filter
        if only and spec_no not in only:
            continue

        try:
            chunks = parse_spec_file(path)
        except Exception as exc:  # noqa: BLE001 — intentional broad catch for resilience
            console.print(f"  [yellow]⚠ Skipping {path.name}: {exc}[/yellow]")
            failed.append(path.name)
            continue

        total_tokens = sum(c.token_count for c in chunks)
        console.print(
            f"   {path.name:<40}  →  "
            f"[cyan]{len(chunks):>5,}[/cyan] chunks "
            f"({total_tokens / 1_000:.1f}K tokens)"
        )
        all_chunks.extend(chunks)

    return all_chunks, failed


def embed_all(
    chunks: list[Chunk],
    embedder: VoyageEmbedder,
    batch_size: int = 128,
) -> list[list[float]]:
    """Embed all chunk texts using *embedder*.

    Parameters
    ----------
    chunks:
        Chunks whose ``.text`` fields will be embedded.
    embedder:
        Initialised VoyageEmbedder instance.
    batch_size:
        Number of texts per Voyage API call.

    Returns
    -------
    list[list[float]]
        One embedding vector per chunk, in the same order.
    """
    texts = [c.text for c in chunks]
    return embedder.embed_documents(texts, batch_size=batch_size)


def upsert_all(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    store: ChromaSpecStore,
) -> None:
    """Upsert *chunks* and *embeddings* into *store*."""
    store.upsert_chunks(chunks, embeddings)


def print_summary(
    chunks: list[Chunk],
    failed: list[str],
    store: ChromaSpecStore,
    dry_run: bool = False,
) -> None:
    """Render a rich table summarising ingestion results."""
    # Aggregate per spec
    spec_stats: dict[str, dict] = {}
    for c in chunks:
        key = c.spec_no
        if key not in spec_stats:
            spec_stats[key] = {"chunks": 0, "tokens": 0}
        spec_stats[key]["chunks"] += 1
        spec_stats[key]["tokens"] += c.token_count

    total_chunks = sum(s["chunks"] for s in spec_stats.values())
    total_tokens = sum(s["tokens"] for s in spec_stats.values())
    estimated_cost = total_tokens * VOYAGE_COST_PER_1M / 1_000_000

    table = Table(title="Ingestion Summary", show_lines=False)
    table.add_column("Spec No.", style="cyan", no_wrap=True)
    table.add_column("Chunks", justify="right")
    table.add_column("Tokens", justify="right")

    for spec_no, stats in sorted(spec_stats.items()):
        table.add_row(
            spec_no,
            f"{stats['chunks']:,}",
            f"{stats['tokens']:,}",
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_chunks:,}[/bold]",
        f"[bold]{total_tokens:,}[/bold]",
    )
    console.print(table)

    # Cost / collection info
    if dry_run:
        console.print(
            f"\n[yellow]dry-run[/yellow] — estimated embed cost: "
            f"[green]${estimated_cost:.4f}[/green] "
            f"(voyage-3-large @ $0.18/1M tokens)"
        )
    else:
        console.print(
            f"\nEstimated embed cost: [green]${estimated_cost:.4f}[/green]"
        )
        count = store.count()
        console.print(f"[green]✅ Done. Collection has {count:,} chunks.[/green]")

    if failed:
        console.print(
            f"\n[red]Failed ({len(failed)}):[/red] {', '.join(failed)}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    docx_dir: Annotated[
        Path,
        typer.Option("--docx-dir", help="Directory containing .docx files."),
    ] = Path("specs/docx"),
    db_path: Annotated[
        Path,
        typer.Option("--db-path", help="ChromaDB persistence directory."),
    ] = Path("./chroma_db"),
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help="Comma-separated spec numbers to ingest (e.g. '38.331,38.211').",
        ),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option("--reset/--no-reset", help="Clear collection before ingesting."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip --reset confirmation prompt."),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Embedding batch size.", min=1),
    ] = 128,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Parse only; skip embedding and upserting."),
    ] = False,
) -> None:
    """Ingest 3GPP spec docx files into the Chroma vector store."""

    # ── 1. Scan ──────────────────────────────────────────────────────────────
    console.print(f"\n[bold]🔍 Scanning {docx_dir}/[/bold]")
    docx_paths = scan_docx_dir(docx_dir)

    if not docx_paths:
        console.print(f"[red]No matching .docx files found in {docx_dir}[/red]")
        raise typer.Exit(code=1)

    only_list: list[str] | None = (
        [s.strip() for s in only.split(",") if s.strip()] if only else None
    )

    # Count files after filter (for display)
    visible_paths = (
        [p for p in docx_paths if _spec_no_from_path(p) in only_list]
        if only_list
        else docx_paths
    )
    console.print(f"→  [cyan]{len(visible_paths)}[/cyan] docx file(s) found")

    # ── 2. Optional reset ────────────────────────────────────────────────────
    store = ChromaSpecStore(db_path=db_path)

    if reset:
        current_count = store.count()
        if not yes:
            confirmed = typer.confirm(
                f"기존 컬렉션 {current_count:,}개 chunk 삭제. 계속?"
            )
            if not confirmed:
                console.print("[yellow]Aborted.[/yellow]")
                raise typer.Exit()
        store.reset()
        console.print(
            f"[yellow]Collection cleared ({current_count:,} chunks removed).[/yellow]"
        )

    # ── 3. Parse ─────────────────────────────────────────────────────────────
    console.print("\n[bold]📦 Parsing chunks...[/bold]")
    all_chunks, failed = parse_all(docx_paths, only=only_list)

    if not all_chunks and not failed:
        console.print("[yellow]No chunks produced (--only filter may be too narrow).[/yellow]")
        raise typer.Exit()

    total_tokens = sum(c.token_count for c in all_chunks)
    estimated_cost = total_tokens * VOYAGE_COST_PER_1M / 1_000_000
    console.print(
        f"\n[bold]📐 Total:[/bold] {len(all_chunks):,} chunks, "
        f"{total_tokens / 1_000_000:.2f}M tokens"
    )

    if dry_run:
        console.print("\n[yellow]--dry-run: skipping embedding and upsert.[/yellow]")
        print_summary(all_chunks, failed, store, dry_run=True)
        return

    # ── 4. Embed ─────────────────────────────────────────────────────────────
    console.print(
        f"\n[bold]🧠 Embedding via voyage-3-large[/bold] "
        f"(estimated cost: [green]${estimated_cost:.4f}[/green])"
    )
    try:
        embedder = VoyageEmbedder()
        embeddings = embed_all(all_chunks, embedder, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Embedding failed: {exc}[/red]")
        console.print(
            f"[yellow]Partial result: {len(all_chunks):,} chunks parsed but not stored."
            f"\nFailed specs: {', '.join(failed) if failed else 'none'}[/yellow]"
        )
        raise typer.Exit(code=1) from exc

    # ── 5. Upsert ────────────────────────────────────────────────────────────
    console.print(f"\n[bold]💾 Upserting to Chroma ({db_path})[/bold]")
    upsert_all(all_chunks, embeddings, store)

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print_summary(all_chunks, failed, store, dry_run=False)


if __name__ == "__main__":
    app()
