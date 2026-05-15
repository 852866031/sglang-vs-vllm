#!/usr/bin/env python3
"""
plot_results.py — compare engines on three stacked subplots sharing the
time axis:
    1) request arrival rate (req/s)
    2) per-request E2E latency
    3) per-request TTFT

Reads result CSVs produced by the per-engine run_benchmark.py scripts
(same columns as example_script/example_output.csv:
  idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s)

By default compares vLLM and SGLang. Pass --runs name=path ... to plot
arbitrary sets.

Default output: outputs/compare.png
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"

# Stable color assignment per engine family. Variant tags (e.g.
# "sglang-cap96", "sglang-csgmv-cap96") inherit the family color via
# prefix matching in `_color_for()` so the SGLang series renders the
# same color across every comparison plot.
COLORS = {
    "vllm": "tab:blue",
    "sglang": "tab:orange",
    "mini-sglang": "tab:green",
    "nano-vllm": "tab:red",
}
# Order matters: longest-prefix-wins so "mini-sglang-X" doesn't match "sglang".
_COLOR_PREFIXES = sorted(COLORS.keys(), key=len, reverse=True)


def _color_for(name: str) -> str:
    """Return the stable color for a run name, falling back to a prefix
    match against known engine families (so 'sglang-cap96' picks up the
    sglang color)."""
    if name in COLORS:
        return COLORS[name]
    for prefix in _COLOR_PREFIXES:
        if name == prefix or name.startswith(prefix + "-") or name.startswith(prefix + "_"):
            return COLORS[prefix]
    return None


def load_run(path: Path) -> Dict[str, List[float]]:
    """Read a results CSV, filter to status=='ok', return columns as lists.

    Plots key off `t_rel_s` (the column the plot uses for the x-axis).
    When the CSV has a `scheduled_t_rel_s` column — written by recent
    run_benchmark.py — that value is preferred, because the recorded
    actual-send-time `t_rel_s` can drift under event-loop saturation and
    bunch requests at artificial times. The actual-send-time is preserved
    separately as `actual_t_rel_s` for diagnostics.
    """
    t_rel: List[float] = []
    actual_t_rel: List[float] = []
    latency: List[float] = []
    ttft: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        has_sched = "scheduled_t_rel_s" in (reader.fieldnames or [])
        for r in reader:
            if (r.get("status") or "").strip() != "ok":
                continue
            try:
                actual = float(r["t_rel_s"])
                lat = float(r["latency_s"])
            except (TypeError, ValueError):
                continue
            if has_sched:
                try:
                    sched = float(r["scheduled_t_rel_s"])
                except (TypeError, ValueError):
                    sched = actual
                t_rel.append(sched)
            else:
                t_rel.append(actual)
            actual_t_rel.append(actual)
            latency.append(lat)
            ttft_raw = r.get("ttft_s")
            try:
                ttft.append(float(ttft_raw) if ttft_raw not in (None, "", "None") else float("nan"))
            except ValueError:
                ttft.append(float("nan"))
    return {
        "t_rel_s": t_rel,            # x-axis (scheduled when available)
        "actual_t_rel_s": actual_t_rel,
        "latency_s": latency,
        "ttft_s": ttft,
    }


def _scatter_metric(
    ax,
    runs: Dict[str, Dict[str, List[float]]],
    y_key: str,
    ylabel: str,
    title: str,
    ymax: float = None,
) -> None:
    """Scatter ``y_key`` vs ``t_rel_s`` for every engine on ``ax``, with a
    dashed line at the grand mean and per-engine averages in the legend."""
    all_vals: List[float] = []
    for data in runs.values():
        for v in data[y_key]:
            if v == v:  # filter NaN
                all_vals.append(v)
    overall_avg = sum(all_vals) / len(all_vals) if all_vals else float("nan")

    for name, data in runs.items():
        xs = data["t_rel_s"]
        ys = data[y_key]
        pairs = [(x, y) for x, y in zip(xs, ys) if y == y]
        if not pairs:
            continue
        xs2, ys2 = zip(*pairs)
        avg = sum(ys2) / len(ys2)
        ax.scatter(
            xs2, ys2,
            s=10, alpha=0.55,
            label=f"{name} (avg {avg:.2f}s)",
            color=_color_for(name),
            edgecolors="none",
        )

    if all_vals:
        ax.axhline(overall_avg, color="gray", linestyle="--", linewidth=1.0)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=9)


def _rate_subplot(
    ax,
    runs: Dict[str, Dict[str, List[float]]],
    bin_s: float = 1.0,
) -> None:
    """Per-second arrival rate per engine. The engines replay the same
    timeline.csv so curves should overlap closely; divergence here means
    one engine was holding requests in its queue."""
    # Common time range across all runs.
    all_t = [t for data in runs.values() for t in data["t_rel_s"]]
    if not all_t:
        return
    t_max = max(all_t)
    n_bins = max(1, int(t_max / bin_s) + 1)
    edges = [i * bin_s for i in range(n_bins + 1)]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

    for name, data in runs.items():
        counts = [0] * n_bins
        for t in data["t_rel_s"]:
            i = min(int(t / bin_s), n_bins - 1)
            counts[i] += 1
        rates = [c / bin_s for c in counts]
        ax.plot(
            centers, rates,
            label=f"{name} ({len(data['t_rel_s'])} req)",
            color=_color_for(name),
            linewidth=1.2, alpha=0.85,
        )

    ax.set_ylabel("req / s")
    ax.set_title("Request Arrival Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def make_combined_plot(
    runs: Dict[str, Dict[str, List[float]]],
    out_path: Path,
    ymax_latency: float = None,
    ymax_ttft: float = None,
    bin_s: float = 1.0,
) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 12), sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 2]},
    )
    _rate_subplot(axes[0], runs, bin_s=bin_s)
    _scatter_metric(
        axes[1], runs, y_key="latency_s",
        ylabel="latency (s)", title="Request E2E Latency vs Time",
        ymax=ymax_latency,
    )
    _scatter_metric(
        axes[2], runs, y_key="ttft_s",
        ylabel="TTFT (s)", title="Request TTFT vs Time",
        ymax=ymax_ttft,
    )
    axes[-1].set_xlabel("time since first request (s)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    total = sum(len(d["t_rel_s"]) for d in runs.values())
    print(f"wrote {out_path} ({total} points across {len(runs)} runs)")


def parse_runs(specs: List[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for s in specs:
        if "=" not in s:
            sys.exit(f"--runs entries must be NAME=PATH (got: {s!r})")
        name, path = s.split("=", 1)
        runs[name] = Path(path)
    return runs


def _resolve_variants(globs: List[str]) -> Dict[str, Path]:
    """Expand glob(s) into a NAME→PATH map. NAME is derived from filename:
    sglang_cap96.csv -> 'sglang-cap96', sglang_results.csv -> 'sglang'."""
    import glob as _glob
    paths: List[Path] = []
    for g in globs:
        matched = sorted(_glob.glob(g))
        if not matched:
            print(f"[warn] no files matched {g!r}", file=sys.stderr)
        for m in matched:
            paths.append(Path(m))
    out: Dict[str, Path] = {}
    for p in paths:
        stem = p.stem
        # Trim trailing "_results" so the legend isn't noisy.
        if stem.endswith("_results"):
            stem = stem[: -len("_results")]
        # Convert underscores to dashes so "sglang_cap96" becomes "sglang-cap96".
        name = stem.replace("_", "-")
        if name in out:
            # Disambiguate by path if needed.
            name = f"{name}-{len(out)}"
        out[name] = p
    return out


def _auto_discover_variants(baseline_path: Path) -> List[str]:
    """Default no-args behaviour: find every results CSV in the baseline's
    directory that isn't the baseline itself. Returns the paths as a list
    of glob-resolved strings suitable for `_resolve_variants()`."""
    out_dir = baseline_path.parent
    candidates: List[Path] = []
    for prefix in COLORS:  # vllm, sglang, mini-sglang, nano-vllm
        candidates += sorted(out_dir.glob(f"{prefix}_*.csv"))
        # Also support files where '-' was converted to '_' on disk
        # (mini-sglang -> mini_sglang_*.csv).
        alt = prefix.replace("-", "_")
        if alt != prefix:
            candidates += sorted(out_dir.glob(f"{alt}_*.csv"))
    seen: Dict[Path, None] = {}
    for p in candidates:
        if p.resolve() == baseline_path.resolve():
            continue
        seen.setdefault(p.resolve(), None)
    return [str(p) for p in seen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", default=None,
        help="Pairs of NAME=PATH. Default: "
             "vllm=outputs/vllm_results.csv sglang=outputs/sglang_results.csv. "
             "Used in single-plot mode (mutually exclusive with --sweep).",
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR / "compare.png"))
    ap.add_argument("--ymax_latency", type=float, default=None,
                    help="Clip y-axis on the latency subplot (e.g. 10).")
    ap.add_argument("--ymax_ttft", type=float, default=None,
                    help="Clip y-axis on the TTFT subplot.")
    ap.add_argument("--rate_bin_s", type=float, default=1.0,
                    help="Bin width (s) for the request-rate subplot.")

    # Sweep mode: one plot per variant against a fixed baseline.
    ap.add_argument(
        "--sweep", action="store_true",
        help="Sweep mode: produce one plot per --variants entry, each compared "
             "against --baseline. Output filenames are auto-derived under --out_dir.",
    )
    ap.add_argument(
        "--baseline", default=f"vllm={DEFAULT_OUT_DIR / 'vllm_results.csv'}",
        help="Baseline NAME=PATH for sweep mode. Default: vllm=outputs/vllm_results.csv",
    )
    ap.add_argument(
        "--variants", nargs="+", default=None,
        help="One or more variant CSV paths (globs allowed). The variant name "
             "is derived from the filename stem. Example: "
             "'outputs/sglang_*.csv'.",
    )
    ap.add_argument(
        "--out_dir", default=str(DEFAULT_OUT_DIR),
        help="Output directory for per-variant plots in sweep mode.",
    )
    args = ap.parse_args()

    # If neither --runs nor --variants was passed, default to sweep mode and
    # auto-discover every results CSV in outputs/ as a variant vs the
    # baseline. This makes `python plot_results.py` produce one plot per
    # variant out of the box.
    if not args.sweep and args.runs is None and args.variants is None:
        args.sweep = True

    if args.sweep:
        base_specs = parse_runs([args.baseline])
        if not base_specs:
            sys.exit("--baseline must be NAME=PATH")
        baseline_runs: Dict[str, Dict[str, List[float]]] = {}
        for name, path in base_specs.items():
            if not path.exists():
                sys.exit(f"baseline {name}: {path} not found")
            baseline_runs[name] = load_run(path)
            print(f"[ok] baseline {name}: {len(baseline_runs[name]['t_rel_s'])} ok rows from {path}")

        # If --variants wasn't passed, auto-discover every results CSV under
        # outputs/ that isn't the baseline.
        baseline_path = next(iter(base_specs.values()))
        if args.variants:
            variants_in = args.variants
        else:
            variants_in = _auto_discover_variants(baseline_path)
            if not variants_in:
                sys.exit(
                    f"no variant CSVs found in {DEFAULT_OUT_DIR} "
                    f"(expected files like sglang_*.csv, mini_sglang_*.csv, ...)"
                )
            print(f"[auto] discovered {len(variants_in)} variants in {DEFAULT_OUT_DIR}")
        variant_map = _resolve_variants(variants_in)
        if not variant_map:
            sys.exit("no variants found")

        out_dir = Path(args.out_dir)
        baseline_name = next(iter(base_specs))
        for v_name, v_path in variant_map.items():
            if v_name == baseline_name:
                # Don't compare the baseline to itself.
                print(f"[skip] variant {v_name} matches baseline name; skipping")
                continue
            v_data = load_run(v_path)
            print(f"[ok] variant {v_name}: {len(v_data['t_rel_s'])} ok rows from {v_path}")
            combined = dict(baseline_runs)
            combined[v_name] = v_data
            out_path = out_dir / f"compare_{baseline_name}_vs_{v_name}.png"
            make_combined_plot(
                combined, out_path=out_path,
                ymax_latency=args.ymax_latency, ymax_ttft=args.ymax_ttft,
                bin_s=args.rate_bin_s,
            )
        return

    # Single-plot mode.
    if args.runs is None:
        run_specs = {
            "vllm": DEFAULT_OUT_DIR / "vllm_results.csv",
            "sglang": DEFAULT_OUT_DIR / "sglang_results.csv",
        }
    else:
        run_specs = parse_runs(args.runs)

    runs: Dict[str, Dict[str, List[float]]] = {}
    for name, path in run_specs.items():
        if not path.exists():
            print(f"[warn] skipping {name}: {path} not found", file=sys.stderr)
            continue
        runs[name] = load_run(path)
        print(f"[ok] loaded {name}: {len(runs[name]['t_rel_s'])} ok rows from {path}")

    if not runs:
        sys.exit("no result CSVs loaded — nothing to plot")

    make_combined_plot(
        runs,
        out_path=Path(args.out),
        ymax_latency=args.ymax_latency,
        ymax_ttft=args.ymax_ttft,
        bin_s=args.rate_bin_s,
    )


if __name__ == "__main__":
    main()
