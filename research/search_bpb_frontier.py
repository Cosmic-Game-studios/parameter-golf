#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


METRIC_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
BYTES_RE = re.compile(r"serialized_model_int8_zlib:(?P<bytes>\d+) bytes")
BLOCK_RE = re.compile(r"^blocks\.(?P<layer>\d+)\.(?P<suffix>.+)$")


@dataclass(frozen=True)
class Candidate:
    name: str
    patterns: tuple[str, ...]


@dataclass
class EvalResult:
    name: str
    patterns: list[str]
    log_path: str
    compressed_bytes: int
    counted_code_bytes: int
    total_bytes: int
    val_loss: float
    val_bpb: float
    bytes_saved_vs_baseline: int
    bpb_delta_vs_baseline: float
    efficiency_bytes_per_bpb: float | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search a byte-vs-bpb frontier for mixed int4/int8 exports.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--num-unique-layers", type=int, required=True)
    p.add_argument("--model-dim", type=int, required=True)
    p.add_argument("--num-heads", type=int, required=True)
    p.add_argument("--num-kv-heads", type=int, required=True)
    p.add_argument("--mlp-mult", type=int, required=True)
    p.add_argument("--val-max-tokens", type=int, default=262144)
    p.add_argument("--counted-code-path", default="train_gpt.py")
    p.add_argument("--python-exe", default="./.venv/bin/python")
    p.add_argument("--train-script", default="train_gpt_mlx.py")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--train-batch-tokens", type=int, default=8192)
    p.add_argument("--val-batch-size", type=int, default=8192)
    p.add_argument("--lr-schedule", default="cosine")
    p.add_argument("--lr-warmup-iters", type=int, default=20)
    p.add_argument("--min-lr-scale", type=float, default=0.1)
    p.add_argument("--tied-embed-lr", type=float, default=0.04)
    p.add_argument("--matrix-lr", type=float, default=0.03)
    p.add_argument("--scalar-lr", type=float, default=0.03)
    p.add_argument("--int4-block-size", type=int, default=64)
    p.add_argument("--max-greedy-steps", type=int, default=4)
    p.add_argument("--max-bpb-delta", type=float, default=0.01)
    p.add_argument("--top-k-candidates", type=int, default=12)
    return p.parse_args()


def sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def discover_candidates(checkpoint: Path, num_layers: int) -> list[Candidate]:
    arrs = np.load(checkpoint)
    families: dict[str, list[int]] = {}
    per_layer: list[tuple[int, str, str]] = []
    for key in arrs.files:
        arr = arrs[key]
        if arr.ndim != 2 or arr.size <= 65_536:
            continue
        m = BLOCK_RE.match(key)
        if not m:
            continue
        layer = int(m.group("layer"))
        suffix = m.group("suffix")
        families.setdefault(suffix, []).append(layer)
        per_layer.append((layer, suffix, key))

    candidates: list[Candidate] = []
    seen: set[tuple[str, ...]] = set()

    def add(name: str, patterns: list[str]) -> None:
        key = tuple(sorted(patterns))
        if not key or key in seen:
            return
        seen.add(key)
        candidates.append(Candidate(name=name, patterns=key))

    half = num_layers // 2
    upper_layers = list(range(half, num_layers))
    lower_layers = list(range(0, half))
    for suffix, layers in sorted(families.items()):
        layers = sorted(set(layers))
        add(f"{suffix}__all", [suffix])
        upper = [f"blocks.{layer}.{suffix}" for layer in layers if layer in upper_layers]
        lower = [f"blocks.{layer}.{suffix}" for layer in layers if layer in lower_layers]
        add(f"{suffix}__upper_half", upper)
        add(f"{suffix}__lower_half", lower)
        for layer in layers:
            if layer in upper_layers:
                add(f"{suffix}__layer_{layer}", [f"blocks.{layer}.{suffix}"])
    return candidates


def parse_log_metrics(log_path: Path) -> tuple[int, float, float]:
    text = log_path.read_text(encoding="utf-8")
    bytes_matches = BYTES_RE.findall(text)
    metric_matches = METRIC_RE.findall(text)
    if not bytes_matches or not metric_matches:
        raise ValueError(f"missing metrics in {log_path}")
    compressed_bytes = int(bytes_matches[-1])
    val_loss, val_bpb = metric_matches[-1]
    return compressed_bytes, float(val_loss), float(val_bpb)


def run_eval(
    args: argparse.Namespace,
    candidate: Candidate,
    baseline_total: int,
    baseline_bpb: float,
) -> EvalResult:
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"frontier_{sanitize_slug(candidate.name)}"
    log_path = logs_dir / f"{run_id}.txt"
    if not log_path.exists():
        env = os.environ.copy()
        quant_format = "mixed_int4_int8_packed_v2" if candidate.patterns else "int8_clean_per_row_v1"
        env.update(
            {
                "RUN_ID": run_id,
                "DATA_PATH": args.data_path,
                "TOKENIZER_PATH": args.tokenizer_path,
                "VOCAB_SIZE": str(args.vocab_size),
                "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
                "VAL_BATCH_SIZE": str(args.val_batch_size),
                "VAL_MAX_TOKENS": str(args.val_max_tokens),
                "VAL_LOSS_EVERY": "0",
                "MAX_WALLCLOCK_SECONDS": "0",
                "ITERATIONS": "0",
                "WARMUP_STEPS": "0",
                "NUM_LAYERS": str(args.num_layers),
                "NUM_UNIQUE_LAYERS": str(args.num_unique_layers),
                "MODEL_DIM": str(args.model_dim),
                "NUM_HEADS": str(args.num_heads),
                "NUM_KV_HEADS": str(args.num_kv_heads),
                "MLP_MULT": str(args.mlp_mult),
                "LR_SCHEDULE": args.lr_schedule,
                "LR_WARMUP_ITERS": str(args.lr_warmup_iters),
                "MIN_LR_SCALE": str(args.min_lr_scale),
                "TIED_EMBED_LR": str(args.tied_embed_lr),
                "MATRIX_LR": str(args.matrix_lr),
                "SCALAR_LR": str(args.scalar_lr),
                "INIT_MODEL_PATH": args.checkpoint,
                "QUANT_FORMAT": quant_format,
                "INT4_NAME_PATTERNS": ",".join(candidate.patterns),
                "INT4_BLOCK_SIZE": str(args.int4_block_size),
            }
        )
        cmd = [args.python_exe, args.train_script]
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=Path(args.train_script).resolve().parent, env=env, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"eval failed for {candidate.name}: see {log_path}")

    compressed_bytes, val_loss, val_bpb = parse_log_metrics(log_path)
    counted_code_bytes = Path(args.counted_code_path).stat().st_size
    total_bytes = compressed_bytes + counted_code_bytes
    bytes_saved = baseline_total - total_bytes
    bpb_delta = val_bpb - baseline_bpb
    efficiency = None
    if bytes_saved > 0:
        if bpb_delta <= 0:
            efficiency = float("inf")
        else:
            efficiency = bytes_saved / bpb_delta
    return EvalResult(
        name=candidate.name,
        patterns=list(candidate.patterns),
        log_path=str(log_path),
        compressed_bytes=compressed_bytes,
        counted_code_bytes=counted_code_bytes,
        total_bytes=total_bytes,
        val_loss=val_loss,
        val_bpb=val_bpb,
        bytes_saved_vs_baseline=bytes_saved,
        bpb_delta_vs_baseline=bpb_delta,
        efficiency_bytes_per_bpb=efficiency,
    )


def write_outputs(out_json: Path, out_md: Path, payload: dict[str, Any]) -> None:
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# BPB Frontier Search",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Baseline total bytes: `{payload['baseline']['total_bytes']}`",
        f"- Baseline val_bpb: `{payload['baseline']['val_bpb']:.8f}`",
        "",
        "## Top Single Candidates",
        "",
        "| Name | Bytes saved | bpb delta | total bytes | val_bpb |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["top_single_candidates"]:
        lines.append(
            f"| `{row['name']}` | `{row['bytes_saved_vs_baseline']}` | "
            f"`{row['bpb_delta_vs_baseline']:.8f}` | `{row['total_bytes']}` | `{row['val_bpb']:.8f}` |"
        )
    lines += [
        "",
        "## Best Greedy Combo",
        "",
        f"- Name: `{payload['best_combo']['name']}`",
        f"- Patterns: `{','.join(payload['best_combo']['patterns'])}`",
        f"- Total bytes: `{payload['best_combo']['total_bytes']}`",
        f"- Val bpb: `{payload['best_combo']['val_bpb']:.8f}`",
        f"- Bytes saved vs baseline: `{payload['best_combo']['bytes_saved_vs_baseline']}`",
        f"- bpb delta vs baseline: `{payload['best_combo']['bpb_delta_vs_baseline']:.8f}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    candidates = discover_candidates(checkpoint, args.num_layers)
    if not candidates:
        raise ValueError("no candidates discovered")

    baseline = run_eval(args, Candidate(name="baseline", patterns=tuple()), baseline_total=0, baseline_bpb=0.0)
    single_results = [
        run_eval(args, candidate, baseline_total=baseline.total_bytes, baseline_bpb=baseline.val_bpb)
        for candidate in candidates
    ]
    def single_sort_key(result: EvalResult) -> tuple[int, float, float, int]:
        if result.efficiency_bytes_per_bpb is None:
            eff = -1.0
        elif result.efficiency_bytes_per_bpb == float("inf"):
            eff = 1e30
        else:
            eff = result.efficiency_bytes_per_bpb
        return (int(result.bytes_saved_vs_baseline > 0), eff, -result.bpb_delta_vs_baseline, result.bytes_saved_vs_baseline)

    single_results.sort(key=single_sort_key, reverse=True)
    greedy_pool = single_results[: args.top_k_candidates]

    greedy_selected: list[Candidate] = []
    greedy_best = baseline
    remaining = {
        result.name: Candidate(name=result.name, patterns=tuple(result.patterns))
        for result in greedy_pool
    }
    for step in range(args.max_greedy_steps):
        best_step: EvalResult | None = None
        best_candidate: Candidate | None = None
        for candidate in list(remaining.values()):
            combo_patterns = sorted({*greedy_best.patterns, *candidate.patterns})
            combo = Candidate(
                name=f"greedy_step_{step + 1}__{'__'.join([c.name for c in greedy_selected] + [candidate.name])}",
                patterns=tuple(combo_patterns),
            )
            result = run_eval(args, combo, baseline_total=baseline.total_bytes, baseline_bpb=baseline.val_bpb)
            if result.bytes_saved_vs_baseline <= 0 or result.bpb_delta_vs_baseline > args.max_bpb_delta:
                continue
            if best_step is None:
                best_step, best_candidate = result, candidate
                continue
            left = (best_step.bytes_saved_vs_baseline, -best_step.bpb_delta_vs_baseline)
            right = (result.bytes_saved_vs_baseline, -result.bpb_delta_vs_baseline)
            if right > left:
                best_step, best_candidate = result, candidate
        if best_step is None or best_candidate is None:
            break
        greedy_selected.append(best_candidate)
        greedy_best = best_step
        remaining.pop(best_candidate.name, None)

    stem = checkpoint.stem
    out_json = Path(args.out_json) if args.out_json else Path("research") / f"{stem}_bpb_frontier.json"
    out_md = Path(args.out_md) if args.out_md else Path("research") / f"{stem}_bpb_frontier.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": str(checkpoint),
        "baseline": asdict(baseline),
        "top_single_candidates": [asdict(r) for r in single_results[: args.top_k_candidates]],
        "best_combo": asdict(greedy_best),
        "selected_candidates": [c.name for c in greedy_selected],
    }
    write_outputs(out_json, out_md, payload)
    print(out_json)
    print(out_md)
    print(json.dumps(payload["best_combo"], indent=2))


if __name__ == "__main__":
    main()
