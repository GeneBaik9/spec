"""Generate INDEX.md describing every downloaded 3GPP spec.

Reads PDF metadata via pdfinfo, falls back to docx filename for multipart
specs without PDF. Groups output by series (TS 36 / TS 38) and working
group, mirroring config/specs.yaml.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "specs" / "pdf"
DOCX_DIR = ROOT / "specs" / "docx"
CONFIG = ROOT / "config" / "specs.yaml"
OUTPUT = ROOT / "INDEX.md"

# Map config keys to display labels
SERIES_LABELS = {"ts_36": "TS 36 series — LTE", "ts_38": "TS 38 series — NR"}
WG_LABELS = {
    "ran1_physical_layer": "RAN1 — Physical Layer",
    "ran2_layer2_layer3": "RAN2 — L2/L3 Protocols",
    "ran3_architecture": "RAN3 — Architecture & Interfaces",
    "ran4_radio_performance": "RAN4 — Radio Performance",
}


def pdf_meta(pdf_path: Path) -> tuple[int, str]:
    """Return (page_count, title) from pdfinfo. Missing values default to (0, '')."""
    out = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    pages = 0
    title = ""
    for line in out.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
        elif line.startswith("Title:"):
            title = line.split(":", 1)[1].strip()
    return pages, title


def clean_title(raw: str) -> str:
    """Strip ETSI/version prefixes and the trailing 3GPP reference."""
    cleaned = re.sub(r"^(?:ETSI\s+)?TS\s+\d{3}[\s\d.-]*-\s+V[\d.]+\s+-\s+", "", raw)
    cleaned = re.sub(r"\s+\(3GPP TS [\d.-]+ version [\d.]+ Release \d+\)\s*$", "", cleaned)
    return cleaned.strip()


def find_artifact(spec_no: str) -> tuple[Path | None, str, int, str]:
    """Locate the best available artifact for *spec_no*.

    Returns (path, kind, pages, title). kind is 'pdf' or 'docx' or 'missing'.
    """
    for pdf in sorted(PDF_DIR.glob(f"{spec_no}-v*.pdf")):
        pages, raw_title = pdf_meta(pdf)
        return pdf, "pdf", pages, clean_title(raw_title) if raw_title else ""
    for docx in sorted(DOCX_DIR.glob(f"{spec_no}-v*.docx")):
        return docx, "docx", 0, ""
    return None, "missing", 0, ""


def build() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    specs_block: dict = cfg.get("specs", {})

    lines: list[str] = [
        "# 3GPP RAN Spec INDEX",
        "",
        "_Latest available Release 19 versions, downloaded via `scripts/download_specs.py`"
        " (docx) and `scripts/download_pdfs.py` (PDF via ETSI + libreoffice)._",
        "",
        "## How to use",
        "1. Pick the spec(s) from the tables below based on the topic.",
        "2. `Read({path, pages: \"1-30\"})` first → get the table of contents.",
        "3. `Read({path, pages: \"NNN-MMM\"})` for the section you need.",
        "",
    ]

    totals = defaultdict(int)
    for series_key, wg_dict in specs_block.items():
        lines.append(f"## {SERIES_LABELS.get(series_key, series_key)}")
        lines.append("")
        for wg_key, spec_list in wg_dict.items():
            lines.append(f"### {WG_LABELS.get(wg_key, wg_key)}")
            lines.append("")
            lines.append("| Spec | Title | Format | Pages | Path |")
            lines.append("|------|-------|--------|-------|------|")
            for spec_no in spec_list:
                path, kind, pages, title = find_artifact(spec_no)
                if path is None:
                    lines.append(f"| {spec_no} | _(missing)_ | — | — | — |")
                    continue
                rel = path.relative_to(ROOT)
                page_cell = f"{pages:,}" if pages else "—"
                lines.append(f"| {spec_no} | {title or '(no metadata)'} | {kind.upper()} | {page_cell} | `{rel}` |")
                totals[kind] += 1
            lines.append("")

    lines.append("---")
    lines.append("")
    summary = ", ".join(f"{k.upper()}: {v}" for k, v in totals.items())
    lines.append(f"_Total artifacts — {summary}_")
    lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({sum(totals.values())} specs)")


if __name__ == "__main__":
    build()
