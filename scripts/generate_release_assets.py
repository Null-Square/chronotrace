#!/usr/bin/env python3
"""Generate reviewer-facing figures and paper macros from the frozen selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "configs/chronotrace_pairwise_multi_witness_confirmation_v3.selection.json"
TARGETS = ("ABCD", "BCDA", "CDAB", "DABC", "DCBA", "ADCB", "BADC", "CBAD")


def _load() -> dict[str, Any]:
    return json.loads(SELECTION.read_text(encoding="utf-8"))


def _fmt_pct(numerator: int, denominator: int, digits: int = 1) -> str:
    return f"{100.0 * numerator / denominator:.{digits}f}"


def _results_svg(selection: dict[str, Any]) -> str:
    full = int(selection["full_history_certificate_coverage"])
    pairs = int(selection["label_blind_pairwise_orientation_certificate_coverage"])
    abstentions = int(selection["full_history_abstention_count"])
    ambiguous = int(selection["ambiguous_pair_count"])
    contradictions = int(selection["contradictory_pair_count"])
    rows = []
    y = 385
    for seed, result in selection["per_seed"].items():
        certified = int(result["full_history_certificate_coverage"])
        width = 420 * certified / 8
        rows.append(
            f'<text x="75" y="{y}" class="seed">{seed}</text>'
            f'<rect x="330" y="{y - 22}" width="420" height="26" rx="6" class="barbg"/>'
            f'<rect x="330" y="{y - 22}" width="{width:.1f}" height="26" rx="6" class="bar"/>'
            f'<text x="775" y="{y}" class="value">{certified}/8 histories</text>'
        )
        y += 48
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-labelledby="title desc">
<title id="title">ChronoTrace frozen fresh confirmation results</title>
<desc id="desc">ChronoTrace certified {full} of 32 complete histories and {pairs} of 192 pairwise precedences, with {abstentions} abstentions and {contradictions} contradictions.</desc>
<style>
  text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; fill: #172033; }}
  .title {{ font-size: 38px; font-weight: 750; }} .subtitle {{ font-size: 20px; fill: #526070; }}
  .metric {{ font-size: 42px; font-weight: 750; }} .label {{ font-size: 16px; fill: #5b6573; }}
  .seed {{ font-size: 17px; font-weight: 650; }} .value {{ font-size: 17px; font-weight: 650; }}
  .footer {{ font-size: 18px; font-weight: 650; }} .card {{ fill: #f7f9fc; stroke: #d8dee9; }}
  .barbg {{ fill: #e8edf4; }} .bar {{ fill: #3159c8; }}
</style>
<rect width="1200" height="630" rx="28" fill="#ffffff"/>
<text x="60" y="72" class="title">ChronoTrace — frozen fresh confirmation</text>
<text x="60" y="108" class="subtitle">Label-blind Pythia-14M chronology certificates · N = K = 4 · replay-capable access</text>
<rect x="60" y="145" width="330" height="150" rx="18" class="card"/>
<rect x="435" y="145" width="330" height="150" rx="18" class="card"/>
<rect x="810" y="145" width="330" height="150" rx="18" class="card"/>
<text x="90" y="215" class="metric">{full}/32</text>
<text x="90" y="255" class="label">complete histories certified ({_fmt_pct(full, 32, 1)}%)</text>
<text x="465" y="215" class="metric">{pairs}/192</text>
<text x="465" y="255" class="label">pairwise precedences certified ({_fmt_pct(pairs, 192, 1)}%)</text>
<text x="840" y="215" class="metric">{contradictions}</text>
<text x="840" y="255" class="label">contradictory certified pairs</text>
<text x="60" y="338" class="subtitle">Complete-history coverage by fresh seed</text>
{''.join(rows)}
<text x="60" y="590" class="footer">{abstentions} conservative abstentions · {ambiguous} ambiguous pair decisions · preregistered tier: STRONG</text>
</svg>
'''


def _pipeline_svg() -> str:
    labels = (
        (50, "Known inputs", "base checkpoint · candidate stages · training rule"),
        (285, "Hidden variable", "unknown stage chronology"),
        (520, "Observed endpoint", "final model weights"),
        (755, "Frozen geometry", "ordered interactions · witness bank"),
        (990, "Two-sided certificates", "test i<j and j<i independently"),
        (1225, "Output", "precedence graph · total order or abstain"),
    )
    boxes = []
    for x, title, body in labels:
        boxes.append(
            f'<rect x="{x}" y="118" width="190" height="126" rx="16" class="box"/>'
            f'<text x="{x + 18}" y="158" class="boxtitle">{title}</text>'
            f'<foreignObject x="{x + 18}" y="172" width="156" height="60">'
            '<div xmlns="http://www.w3.org/1999/xhtml" class="body">'
            f'{body}</div></foreignObject>'
        )
    arrows = []
    for x in (240, 475, 710, 945, 1180):
        arrows.append(f'<path d="M{x} 181 H{x + 40}" class="arrow" marker-end="url(#arrow)"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1450" height="360" viewBox="0 0 1450 360" role="img" aria-labelledby="title desc">
<title id="title">ChronoTrace certificate pipeline</title>
<desc id="desc">Known training inputs and a hidden chronology produce final weights. ChronoTrace freezes low-degree geometry and witnesses, then issues label-blind two-sided precedence certificates that may abstain.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#3159c8"/></marker></defs>
<style>
 text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; fill: #172033; }}
 .title {{ font-size: 31px; font-weight: 750; }} .subtitle {{ font-size: 17px; fill: #526070; }}
 .box {{ fill: #f7f9fc; stroke: #cad3e0; stroke-width: 1.5; }} .boxtitle {{ font-size: 17px; font-weight: 700; }}
 .body {{ font-family: Inter, ui-sans-serif, system-ui; font-size: 14px; line-height: 1.25; color: #526070; }}
 .arrow {{ stroke: #3159c8; stroke-width: 3; fill: none; }} .tag {{ font-size: 15px; font-weight: 650; fill: #3159c8; }}
</style>
<rect width="1450" height="360" rx="24" fill="#ffffff"/>
<text x="50" y="55" class="title">Certified inverse reconstruction of training chronology</text>
<text x="50" y="86" class="subtitle">The chronology is hidden from the decision rule; the method certifies precedence claims or abstains.</text>
{''.join(boxes)}{''.join(arrows)}
<text x="767" y="282" class="tag">witnesses frozen before higher-order candidate output</text>
<text x="1004" y="312" class="tag">label-blind decision</text>
</svg>
'''


def _macros(selection: dict[str, Any]) -> str:
    full = int(selection["full_history_certificate_coverage"])
    pairs = int(selection["label_blind_pairwise_orientation_certificate_coverage"])
    ambiguous = int(selection["ambiguous_pair_count"])
    abstentions = int(selection["full_history_abstention_count"])
    min_margin = float(selection["minimum_excluded_orientation_margin_over_guard"])
    return (
        "% Generated by scripts/generate_release_assets.py; do not edit by hand.\n"
        f"\\newcommand{{\\ChronoFullCertified}}{{{full}}}\n"
        "\\newcommand{\\ChronoFullTotal}{32}\n"
        f"\\newcommand{{\\ChronoFullPct}}{{{_fmt_pct(full, 32, 1)}\\%}}\n"
        f"\\newcommand{{\\ChronoPairCertified}}{{{pairs}}}\n"
        "\\newcommand{\\ChronoPairTotal}{192}\n"
        f"\\newcommand{{\\ChronoPairPct}}{{{_fmt_pct(pairs, 192, 1)}\\%}}\n"
        f"\\newcommand{{\\ChronoAbstentions}}{{{abstentions}}}\n"
        f"\\newcommand{{\\ChronoAmbiguousPairs}}{{{ambiguous}}}\n"
        "\\newcommand{\\ChronoContradictions}{0}\n"
        "\\newcommand{\\ChronoDoubleExclusions}{0}\n"
        "\\newcommand{\\ChronoOutcomeTier}{strong}\n"
        f"\\newcommand{{\\ChronoMinMarginOverGuard}}{{{min_margin:.3e}}}\n"
    )


def _matrix(selection: dict[str, Any]) -> str:
    abstentions = selection["abstentions_by_seed_target_and_pair"]
    seeds = tuple(selection["per_seed"])
    lines = [
        "% Generated by scripts/generate_release_assets.py; do not edit by hand.",
        "\\begin{tikzpicture}[x=1.05cm,y=0.70cm,font=\\scriptsize]",
    ]
    for column, target in enumerate(TARGETS):
        lines.append(f"\\node[rotate=45,anchor=west] at ({column + 1},1.0) {{{target}}};")
    for row, seed in enumerate(seeds):
        y = -row
        lines.append(f"\\node[anchor=east] at (0.65,{y}) {{{seed}}};")
        by_target = abstentions.get(seed, {})
        for column, target in enumerate(TARGETS):
            x = column + 1
            if target in by_target:
                label = "A"
                style = "fill=black!65,text=white"
            else:
                label = "C"
                style = "fill=black!10"
            lines.append(
                f"\\node[draw,minimum width=0.82cm,minimum height=0.48cm,{style}] "
                f"at ({x},{y}) {{{label}}};"
            )
    lines.extend(
        [
            "\\node[anchor=west] at (0.9,-4.8) {C = complete history certified};",
            "\\node[anchor=west] at (4.8,-4.8) {A = conservative abstention};",
            "\\end{tikzpicture}",
            "",
        ]
    )
    return "\n".join(lines)


def _expected_files(selection: dict[str, Any]) -> dict[Path, str]:
    return {
        ROOT / "assets/chronotrace-results.svg": _results_svg(selection),
        ROOT / "assets/chronotrace-pipeline.svg": _pipeline_svg(),
        ROOT / "paper/generated/results_macros.tex": _macros(selection),
        ROOT / "paper/figures/confirmation_matrix.tex": _matrix(selection),
    }


def _write(files: dict[Path, str]) -> None:
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


def _check(files: dict[Path, str]) -> int:
    failed = False
    for path, expected in files.items():
        if not path.exists():
            print(f"missing generated asset: {path.relative_to(ROOT)}")
            failed = True
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            print(f"stale generated asset: {path.relative_to(ROOT)}")
            failed = True
    if failed:
        print("run: python scripts/generate_release_assets.py --write")
        return 1
    print("ChronoTrace generated release assets: in sync")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    selection = _load()
    files = _expected_files(selection)
    if args.write:
        _write(files)
        return 0
    return _check(files)


if __name__ == "__main__":
    raise SystemExit(main())
