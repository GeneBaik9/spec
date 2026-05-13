"""
Download PDF versions of 3GPP specifications.

Two strategies depending on spec type:
  - Single-part specs (e.g. 38.331): fetch the official ETSI PDF reissue.
    ETSI adds 100 to the 3GPP series prefix: 38.xxx → 138.xxx, 36.xxx → 136.xxx.
  - Multi-part specs (e.g. 38.101-1): ETSI does not publish these.
    Convert the already-downloaded .docx using LibreOffice.

Usage:
    uv run python scripts/download_pdfs.py
    uv run python scripts/download_pdfs.py --only 38.331,38.211
    uv run python scripts/download_pdfs.py --force
    uv run python scripts/download_pdfs.py --dry-run
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ETSI rejects the default httpx UA with 403; a browser UA is required.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
ETSI_BASE = "https://www.etsi.org/deliver/etsi_ts"

# _60 suffix denotes "published" stage in ETSI versioning.
ETSI_STAGE = "_60"

TIMEOUT = 60.0
DOCX_DIR = Path("specs/docx")
PDF_DIR = Path("specs/pdf")

console = Console()
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Version(NamedTuple):
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class Result:
    spec_no: str
    version: str
    method: str        # "etsi" | "libreoffice" | "dry-run" | "skipped"
    status: str        # "ok" | "skipped" | "error"
    size: str          # human-readable file size or "-"
    note: str = field(default="")


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

_DOCX_RE = re.compile(
    r"^(?P<spec>[\d.]+-[\d]+|[\d.]+)-v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)\.docx$"
)


def parse_docx_filename(name: str) -> tuple[str, Version] | None:
    """Parse '{spec_no}-v{major}.{minor}.{patch}.docx' → (spec_no, Version).

    Handles both single-part (38.331) and multi-part (38.101-1) spec names.
    Returns None if the name doesn't match.
    """
    m = _DOCX_RE.match(name)
    if not m:
        return None
    spec_no = m.group("spec")
    ver = Version(int(m.group("major")), int(m.group("minor")), int(m.group("patch")))
    return spec_no, ver


def is_multipart(spec_no: str) -> bool:
    """Return True for multi-part specs like '38.101-1', '38.101-2', etc."""
    return "-" in spec_no


# ---------------------------------------------------------------------------
# ETSI URL construction
# ---------------------------------------------------------------------------


def etsi_url(spec_no: str, version: Version) -> str:
    """Build the canonical ETSI PDF URL for a single-part 3GPP spec.

    Mapping rules:
      - ETSI number = 3GPP number + 100 on the series prefix
        36.xxx → 136.xxx,  38.xxx → 138.xxx
      - Directory group: floor(etsi_num / 100) * 100  to  that + 99
        138331 → 138300_138399
      - ver_dir: "{major:02d}.{minor:02d}.{patch:02d}_60"
      - ver_str: "{major}{minor:02d}{patch:02d}"

    Example: 38.331 v19.2.0
      etsi_full = 138331
      group     = 138300_138399
      ver_dir   = 19.02.00_60
      ver_str   = 190200
      → .../138300_138399/138331/19.02.00_60/ts_138331v190200p.pdf
    """
    # e.g. "38.331" → series=38, sub=331; etsi_num = 138331
    parts = spec_no.split(".")
    series = int(parts[0]) + 100
    sub = int(parts[1])
    etsi_num = series * 1000 + sub          # 138 * 1000 + 331 = 138331
    etsi_full = str(etsi_num)               # "138331"

    # Group directory: e.g. 138331 → 138300_138399
    dir_low = (etsi_num // 100) * 100
    dir_high = dir_low + 99
    group = f"{dir_low}_{dir_high}"

    ver_dir = (
        f"{version.major:02d}.{version.minor:02d}.{version.patch:02d}{ETSI_STAGE}"
    )
    ver_str = f"{version.major}{version.minor:02d}{version.patch:02d}"

    filename = f"ts_{etsi_full}v{ver_str}p.pdf"
    return f"{ETSI_BASE}/{group}/{etsi_full}/{ver_dir}/{filename}"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        follow_redirects=True,
    )


def _head_ok(client: httpx.Client, url: str) -> bool:
    """Return True if HEAD request returns 200."""
    try:
        r = client.head(url)
        return r.status_code == 200
    except httpx.RequestError:
        return False


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# ETSI download (with fallback version probing)
# ---------------------------------------------------------------------------


def _candidate_versions(version: Version) -> list[Version]:
    """Generate nearby version candidates to try when the exact version 404s.

    Strategy: try same major with nearby minor/patch combos.
    Keeps total attempts ≤ 10 (not counting the original).
    """
    candidates: list[Version] = []
    major = version.major

    # Same minor, patch 0..3 (excluding exact match)
    for p in range(4):
        v = Version(major, version.minor, p)
        if v != version:
            candidates.append(v)

    # Adjacent minors (±1), patch 0
    for delta in (-1, 1, 2, 3, 4):
        m = version.minor + delta
        if m >= 0:
            candidates.append(Version(major, m, 0))

    # Deduplicate while preserving order, cap at 10
    seen: set[Version] = set()
    result: list[Version] = []
    for v in candidates:
        if v not in seen:
            seen.add(v)
            result.append(v)
        if len(result) >= 10:
            break
    return result


def download_etsi_pdf(
    client: httpx.Client,
    spec_no: str,
    version: Version,
    dest: Path,
    force: bool,
    dry_run: bool = False,
) -> tuple[bool, str, Version | None]:
    """Try to fetch the ETSI PDF for *spec_no* at *version* (with fallback).

    Returns (success, note, actual_version_used).
    On dry_run, just returns (True, url, version) without downloading.
    """
    # Try exact version first, then candidates
    to_try = [version] + _candidate_versions(version)

    for candidate in to_try:
        url = etsi_url(spec_no, candidate)

        if dry_run:
            console.print(f"  [cyan]DRY-RUN ETSI[/cyan]  {spec_no} → {url}")
            return True, url, candidate

        if dest.exists() and not force and candidate == version:
            return True, "skipped (already exists)", version

        # Probe before streaming to avoid partial writes on 404
        if not _head_ok(client, url):
            if candidate == version:
                console.print(
                    f"  [dim]{spec_no} v{candidate} → 404, trying fallbacks…[/dim]"
                )
            continue

        # HEAD OK — stream download
        note = "" if candidate == version else f"version adjusted to v{candidate}"
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    continue
                total = int(resp.headers.get("content-length", 0)) or None
                with open(dest, "wb") as fh:
                    from tqdm import tqdm
                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"{spec_no} (ETSI)",
                        leave=False,
                    ) as bar:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            fh.write(chunk)
                            bar.update(len(chunk))
        except httpx.RequestError as exc:
            return False, f"Network error: {exc}", None

        return True, note, candidate

    return False, "ETSI 404 on all candidates", None


# ---------------------------------------------------------------------------
# LibreOffice conversion
# ---------------------------------------------------------------------------

_LIBREOFFICE_BIN: str | None = shutil.which("libreoffice") or shutil.which("soffice")


def _libreoffice_writer_available() -> bool:
    """Return True if LibreOffice Writer module is available (can load .docx).

    The libreoffice binary may exist without libreoffice-writer installed.
    In that case, any attempt to convert a docx will silently fail with
    "Error: source file could not be loaded" despite exit code 0.
    We detect this by probing the LO module path.
    """
    import glob
    if _LIBREOFFICE_BIN is None:
        return False
    # Look for the writer shared library in standard LO lib dirs
    lo_lib_dirs = [
        "/usr/lib/libreoffice/program",
        "/opt/libreoffice*/program",
        "/usr/lib/x86_64-linux-gnu/libreoffice/program",
    ]
    for pattern in lo_lib_dirs:
        for lo_dir in glob.glob(pattern):
            # swlo = StarWriter LO module
            if any(Path(lo_dir).glob("libswlo.so*")):
                return True
    return False


def convert_with_libreoffice(
    docx_path: Path,
    pdf_dir: Path,
    dest: Path,
    force: bool,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Convert *docx_path* to PDF in *pdf_dir* using LibreOffice.

    LibreOffice names the output file after the input stem, so we rename
    the result to *dest* to follow our naming convention.

    Returns (success, note).
    """
    if dry_run:
        console.print(
            f"  [cyan]DRY-RUN LIBREOFFICE[/cyan]  {docx_path.name} → {dest.name}"
        )
        return True, str(dest)

    if dest.exists() and not force:
        return True, "skipped (already exists)"

    if _LIBREOFFICE_BIN is None:
        return False, "LibreOffice not found (install libreoffice)"

    if not _libreoffice_writer_available():
        return False, (
            "libreoffice-writer not installed — "
            "run: sudo apt install libreoffice-writer"
        )

    pdf_dir.mkdir(parents=True, exist_ok=True)

    # LibreOffice writes {stem}.pdf in outdir; stem comes from the input filename.
    intermediate = pdf_dir / (docx_path.stem + ".pdf")

    console.print(f"  [yellow]libreoffice[/yellow]  converting {docx_path.name} …", end="")
    t0 = time.monotonic()

    proc = subprocess.run(
        [
            _LIBREOFFICE_BIN,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(pdf_dir),
            str(docx_path),
        ],
        timeout=120,
        check=False,
        capture_output=True,
    )

    elapsed = time.monotonic() - t0
    console.print(f" done ({elapsed:.1f}s)")

    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        return False, f"libreoffice exit {proc.returncode}: {stderr[:200]}"

    if not intermediate.exists():
        stderr = proc.stderr.decode(errors="replace").strip()
        # "source file could not be loaded" indicates missing writer module
        if "source file could not be loaded" in stderr:
            return False, (
                "libreoffice-writer not installed — "
                "run: sudo apt install libreoffice-writer"
            )
        stdout = proc.stdout.decode(errors="replace").strip()
        return False, f"Output PDF not found after conversion. stdout: {stdout[:200]}"

    # Rename to canonical name if different
    if intermediate != dest:
        intermediate.rename(dest)

    return True, ""


# ---------------------------------------------------------------------------
# Per-spec orchestration
# ---------------------------------------------------------------------------


def process_spec(
    client: httpx.Client,
    spec_no: str,
    version: Version,
    docx_path: Path,
    pdf_dir: Path,
    force: bool,
    dry_run: bool,
) -> Result:
    """Process a single spec: download ETSI PDF or convert with LibreOffice."""
    ver_str = str(version)
    dest = pdf_dir / f"{spec_no}-v{ver_str}.pdf"

    if is_multipart(spec_no):
        # Multi-part specs are not published on ETSI; use LibreOffice
        ok, note = convert_with_libreoffice(docx_path, pdf_dir, dest, force, dry_run)
        method = "libreoffice"
        if dry_run:
            return Result(spec_no, ver_str, "dry-run", "dry-run", "-", note)
        if not ok:
            return Result(spec_no, ver_str, method, "error", "-", note)
        if note.startswith("skipped"):
            return Result(spec_no, ver_str, method, "skipped", _human_size(dest.stat().st_size), note)
        size = _human_size(dest.stat().st_size) if dest.exists() else "-"
        return Result(spec_no, ver_str, method, "ok", size, note)

    # Single-part spec → try ETSI first
    ok, note, actual_ver = download_etsi_pdf(client, spec_no, version, dest, force, dry_run)

    if dry_run:
        return Result(spec_no, ver_str, "dry-run", "dry-run", "-", note)

    if ok and note == "skipped (already exists)":
        size = _human_size(dest.stat().st_size) if dest.exists() else "-"
        return Result(spec_no, ver_str, "etsi", "skipped", size, note)

    if ok and actual_ver is not None:
        size = _human_size(dest.stat().st_size) if dest.exists() else "-"
        return Result(spec_no, str(actual_ver), "etsi", "ok", size, note)

    # ETSI failed entirely — fall back to LibreOffice conversion
    console.print(
        f"  [yellow]WARN[/yellow] {spec_no}: ETSI download failed ({note}), "
        f"falling back to LibreOffice…"
    )
    ok2, note2 = convert_with_libreoffice(docx_path, pdf_dir, dest, force, dry_run)
    method = "libreoffice"
    if not ok2:
        return Result(spec_no, ver_str, method, "error", "-", f"ETSI: {note}; LO: {note2}")
    if note2.startswith("skipped"):
        size = _human_size(dest.stat().st_size) if dest.exists() else "-"
        return Result(spec_no, ver_str, method, "skipped", size, note2)
    size = _human_size(dest.stat().st_size) if dest.exists() else "-"
    return Result(spec_no, ver_str, method, "ok", size, f"ETSI fallback; {note}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(results: list[Result]) -> None:
    table = Table(title="PDF Download / Conversion Summary", show_lines=False)
    table.add_column("Spec", style="bold")
    table.add_column("Version")
    table.add_column("Method")
    table.add_column("Status")
    table.add_column("Size", justify="right")
    table.add_column("Note")

    status_styles = {
        "ok": "green",
        "skipped": "dim",
        "error": "red",
        "dry-run": "cyan",
    }
    method_styles = {
        "etsi": "blue",
        "libreoffice": "yellow",
        "dry-run": "cyan",
        "skipped": "dim",
    }

    for r in results:
        s_style = status_styles.get(r.status, "white")
        m_style = method_styles.get(r.method, "white")
        table.add_row(
            r.spec_no,
            r.version,
            f"[{m_style}]{r.method}[/{m_style}]",
            f"[{s_style}]{r.status}[/{s_style}]",
            r.size,
            r.note,
        )

    console.print(table)

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    errors = sum(1 for r in results if r.status == "error")
    dry = sum(1 for r in results if r.status == "dry-run")
    console.print(
        f"\nTotal: {len(results)}  "
        f"[green]ok={ok}[/green]  "
        f"[dim]skipped={skipped}[/dim]  "
        f"[red]errors={errors}[/red]"
        + (f"  [cyan]dry-run={dry}[/cyan]" if dry else "")
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    only: str = typer.Option(
        "",
        help="Comma-separated spec numbers to process (e.g. 38.331,38.101-1)",
    ),
    docx_dir: Path = typer.Option(DOCX_DIR, help="Directory containing .docx files"),
    pdf_dir: Path = typer.Option(PDF_DIR, help="Output directory for PDFs"),
    force: bool = typer.Option(False, "--force", help="Re-download / re-convert even if PDF exists"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without downloading"),
) -> None:
    """Download or convert 3GPP specification PDFs.

    Single-part specs are fetched from ETSI; multi-part specs (38.101-x)
    are converted from .docx using LibreOffice.
    """
    if not dry_run and _LIBREOFFICE_BIN is None:
        # Warn early; we can still proceed for ETSI-only specs
        console.print(
            "[yellow]WARNING[/yellow]: LibreOffice not found. "
            "Multi-part specs (38.101-x) cannot be converted."
        )

    # Scan docx directory
    docx_files = sorted(docx_dir.glob("*.docx"))
    if not docx_files:
        console.print(f"[red]No .docx files found in {docx_dir}[/red]")
        raise typer.Exit(1)

    # Parse filenames
    parsed: list[tuple[str, Version, Path]] = []
    for p in docx_files:
        result = parse_docx_filename(p.name)
        if result is None:
            console.print(f"[yellow]SKIP[/yellow] Cannot parse filename: {p.name}")
            continue
        spec_no, ver = result
        parsed.append((spec_no, ver, p))

    # Filter by --only
    if only:
        requested = {s.strip() for s in only.split(",")}
        parsed = [(s, v, p) for s, v, p in parsed if s in requested]
        if not parsed:
            console.print(f"[red]No matching specs found for: {only}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]Processing {len(parsed)} spec(s)…[/bold]")

    results: list[Result] = []
    with _make_client() as client:
        for spec_no, version, docx_path in parsed:
            console.print(f"\n[dim]→ {spec_no} v{version}[/dim]")
            r = process_spec(
                client, spec_no, version, docx_path, pdf_dir, force, dry_run
            )
            results.append(r)

    console.print()
    print_summary(results)


if __name__ == "__main__":
    app()
