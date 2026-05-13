"""
Download the latest version of 3GPP specifications listed in specs.yaml.

Fetches the archive index from the 3GPP FTP site, picks the highest-versioned
zip, extracts the contained .docx (or .doc), and stores both the raw zip and
the extracted document under specs/.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import NamedTuple

import httpx
import typer
import yaml
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.3gpp.org/ftp/Specs"
USER_AGENT = "spec-qa/0.1 (+https://github.com/GeneBaik9/spec)"
TIMEOUT = 60.0
MAX_RETRIES = 1

console = Console()
app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class VersionCode(NamedTuple):
    """Parsed 3GPP version code (major, minor, patch) from zip filename suffix.

    3GPP encodes version components in base-36-like notation where a=10,
    b=11, ..., z=35.  In practice only a–z appear for major, and digits for
    minor/patch, but we handle the general case.
    """

    major: int
    minor: int
    patch: int


class DownloadResult(NamedTuple):
    spec_no: str
    version: str          # human-readable, e.g. "18.5.0"
    status: str           # "ok", "skipped", "error"
    file_size: str        # e.g. "12.3 MB" or "-"
    note: str             # extra detail for errors / warnings


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def parse_config(config_path: Path) -> list[str]:
    """Return a flat list of spec numbers from specs.yaml."""
    with open(config_path) as fh:
        data = yaml.safe_load(fh)

    spec_nos: list[str] = []
    specs_block = data.get("specs", {})
    for series_dict in specs_block.values():       # ts_36, ts_38, ...
        for wg_list in series_dict.values():       # ran1_physical_layer, ...
            for entry in wg_list:
                spec_nos.append(str(entry))
    return spec_nos


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------

_VER_CHAR_RE = re.compile(r"^([a-z0-9])([0-9]{1,2})([0-9]{1,2})$")


def _char_to_int(ch: str) -> int:
    """Convert a single version character to integer (digit or a-z → 10-35)."""
    if ch.isdigit():
        return int(ch)
    return ord(ch) - ord("a") + 10


def parse_version_suffix(suffix: str) -> VersionCode | None:
    """Parse a 3GPP version suffix such as 'i50' → VersionCode(18, 5, 0).

    Returns None if the suffix does not match the expected pattern.
    """
    m = _VER_CHAR_RE.match(suffix.lower())
    if not m:
        return None
    major = _char_to_int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(3))
    return VersionCode(major, minor, patch)


def version_to_str(v: VersionCode) -> str:
    return f"{v.major}.{v.minor}.{v.patch}"


# ---------------------------------------------------------------------------
# Archive index scraping
# ---------------------------------------------------------------------------


def _archive_url(spec_no: str) -> str:
    """Build the archive directory URL for a given spec number.

    38.101-1 → .../archive/38_series/38.101-1/
    """
    # series is everything before the first dot
    series = spec_no.split(".")[0]
    return f"{BASE_URL}/archive/{series}_series/{spec_no}/"


def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        follow_redirects=True,
    )


def _get_with_retry(client: httpx.Client, url: str) -> httpx.Response | None:
    """GET url with one retry on failure.  Returns None on persistent error."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.get(url)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None          # no point retrying
        except httpx.TimeoutException:
            if attempt == MAX_RETRIES:
                return None
        except httpx.RequestError:
            if attempt == MAX_RETRIES:
                return None
    return None


def list_archive_versions(
    client: httpx.Client, spec_no: str
) -> list[tuple[str, str]]:
    """Fetch the archive index and return [(filename, full_url), ...] for .zip files."""
    url = _archive_url(spec_no)
    resp = _get_with_retry(client, url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    results: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if href.lower().endswith(".zip"):
            filename = href.rstrip("/").split("/")[-1]
            full_url = href if href.startswith("http") else url + filename
            results.append((filename, full_url))
    return results


# ---------------------------------------------------------------------------
# Version selection
# ---------------------------------------------------------------------------


def _version_suffix_from_filename(filename: str, spec_no: str) -> str | None:
    """Extract the raw version suffix (e.g. 'i50') from a zip filename.

    Strategy: strip the spec-number prefix (stripping dots and dashes) and
    take the first alphanumeric token of the remainder.
    """
    base = filename.lower().removesuffix(".zip")
    # Normalise spec_no for prefix removal: 38.331 → 38331, 38.101-1 → 3810101
    norm_spec = re.sub(r"[.\-]", "", spec_no)
    # The filename might encode the multipart suffix differently, e.g.
    # 38.101-1 → "38101-01-...".  We strip any leading digits/dashes up to
    # the first version-looking token.
    # Remove the normalised prefix if present
    candidate = re.sub(r"^" + re.escape(norm_spec), "", base)
    # Also try stripping a dash-separated part number prefix (38101-01-i50)
    candidate = re.sub(r"^[-_0-9]+", "", candidate)
    candidate = candidate.lstrip("-_")
    # Take the first word token
    m = re.match(r"([a-z0-9]{3,})", candidate)
    if m:
        return m.group(1)
    return None


def pick_latest(
    entries: list[tuple[str, str]], spec_no: str
) -> tuple[str, str, VersionCode] | None:
    """From the archive listing, return (filename, url, version) of the newest zip.

    Returns None if no parseable version is found.
    """
    best: tuple[str, str, VersionCode] | None = None
    for filename, url in entries:
        suffix = _version_suffix_from_filename(filename, spec_no)
        if suffix is None:
            continue
        ver = parse_version_suffix(suffix)
        if ver is None:
            continue
        if best is None or ver > best[2]:
            best = (filename, url, ver)
    return best


# ---------------------------------------------------------------------------
# Download & extraction
# ---------------------------------------------------------------------------


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


def download_zip(
    client: httpx.Client,
    url: str,
    dest: Path,
    force: bool,
) -> bool:
    """Stream-download *url* to *dest*.  Returns True on success."""
    if dest.exists() and not force:
        return True   # already present, caller handles "skipped" logic

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = _get_with_retry(client, url)
    if resp is None:
        return False

    total = int(resp.headers.get("content-length", 0)) or None
    with (
        open(dest, "wb") as fh,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            leave=False,
        ) as bar,
    ):
        for chunk in resp.iter_bytes(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))
    return True


def extract_docx(
    zip_path: Path,
    spec_no: str,
    version: VersionCode,
    out_dir: Path,
) -> tuple[Path | None, str]:
    """Extract the .docx (or .doc) from a zip into out_dir/docx/.

    Returns (dest_path, warning_message).  warning_message is empty on success.
    """
    docx_dir = out_dir / "docx"
    docx_dir.mkdir(parents=True, exist_ok=True)

    ver_str = version_to_str(version)
    # Sanitise spec_no for use in filename (e.g. 38.101-1 stays as-is)
    dest_stem = f"{spec_no}-v{ver_str}"

    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            return None, "Bad zip file"

        tmp_path = Path(tmp)
        docx_files = sorted(tmp_path.rglob("*.docx"))
        doc_files = sorted(tmp_path.rglob("*.doc"))

        if docx_files:
            src = docx_files[0]     # typically only one per zip
            dest = docx_dir / f"{dest_stem}.docx"
            shutil.copy2(src, dest)
            return dest, ""

        if doc_files:
            src = doc_files[0]
            dest = docx_dir / f"{dest_stem}.doc"
            shutil.copy2(src, dest)
            return dest, "Legacy .doc — conversion needed"

    return None, "No .docx/.doc found in zip"


# ---------------------------------------------------------------------------
# Per-spec orchestration
# ---------------------------------------------------------------------------


def process_spec(
    client: httpx.Client,
    spec_no: str,
    out_dir: Path,
    force: bool,
    dry_run: bool,
) -> DownloadResult:
    """Download and extract a single spec.  Returns a DownloadResult."""
    entries = list_archive_versions(client, spec_no)
    if not entries:
        return DownloadResult(spec_no, "-", "error", "-", "Archive index unreachable or empty")

    best = pick_latest(entries, spec_no)
    if best is None:
        return DownloadResult(spec_no, "-", "error", "-", "Could not parse any version from archive")

    filename, url, version = best
    ver_str = version_to_str(version)

    if dry_run:
        console.print(f"[cyan]DRY-RUN[/cyan] {spec_no} {ver_str}  →  {url}")
        return DownloadResult(spec_no, ver_str, "dry-run", "-", url)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Preserve original filename so multiple specs don't collide
    zip_dest = raw_dir / f"{spec_no}-{filename}"

    already_exists = zip_dest.exists()
    ok = download_zip(client, url, zip_dest, force)
    if not ok:
        return DownloadResult(spec_no, ver_str, "error", "-", f"Download failed: {url}")

    if already_exists and not force:
        status = "skipped"
    else:
        status = "ok"

    docx_path, warning = extract_docx(zip_dest, spec_no, version, out_dir)
    if warning:
        console.print(f"[yellow]WARN[/yellow] {spec_no}: {warning}")

    if docx_path is None:
        size_str = _human_size(zip_dest.stat().st_size)
        return DownloadResult(spec_no, ver_str, "error", size_str, warning or "Extraction failed")

    size_str = _human_size(docx_path.stat().st_size)
    note = warning if warning else ""
    return DownloadResult(spec_no, ver_str, status, size_str, note)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(results: list[DownloadResult]) -> None:
    table = Table(title="Download Summary", show_lines=False)
    table.add_column("Spec", style="bold")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Size", justify="right")
    table.add_column("Note")

    status_styles = {"ok": "green", "skipped": "dim", "error": "red", "dry-run": "cyan"}

    for r in results:
        style = status_styles.get(r.status, "white")
        table.add_row(r.spec_no, r.version, f"[{style}]{r.status}[/{style}]", r.file_size, r.note)

    console.print(table)

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    errors = sum(1 for r in results if r.status == "error")
    console.print(
        f"\nTotal: {len(results)}  "
        f"[green]ok={ok}[/green]  [dim]skipped={skipped}[/dim]  [red]errors={errors}[/red]"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    only: str = typer.Option("", help="Comma-separated spec numbers to download (e.g. 38.331,36.331)"),
    config: Path = typer.Option(Path("config/specs.yaml"), help="Path to specs.yaml"),
    out_dir: Path = typer.Option(Path("specs"), help="Root output directory"),
    force: bool = typer.Option(False, "--force", help="Re-download even if already present"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print URLs without downloading"),
) -> None:
    """Download the latest 3GPP specification documents listed in specs.yaml."""
    all_specs = parse_config(config)

    if only:
        requested = {s.strip() for s in only.split(",")}
        unknown = requested - set(all_specs)
        if unknown:
            console.print(f"[yellow]Unknown specs (not in config): {', '.join(sorted(unknown))}[/yellow]")
        target_specs = [s for s in all_specs if s in requested]
    else:
        target_specs = all_specs

    console.print(f"[bold]Processing {len(target_specs)} spec(s)…[/bold]")

    results: list[DownloadResult] = []
    with _make_client() as client:
        for spec_no in target_specs:
            console.print(f"  [dim]{spec_no}[/dim]", end="  ")
            result = process_spec(client, spec_no, out_dir, force, dry_run)
            status_color = {"ok": "green", "skipped": "dim", "error": "red", "dry-run": "cyan"}.get(
                result.status, "white"
            )
            console.print(f"[{status_color}]{result.status}[/{status_color}] {result.version}")
            results.append(result)

    print_summary(results)


if __name__ == "__main__":
    app()
