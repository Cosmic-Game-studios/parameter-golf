#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_TABLE = ROOT / "research" / "experiment_table.md"
DEFAULT_HANDOFF = ROOT / "HANDOFF.md"
DEFAULT_OUT_JSON = ROOT / "research" / "sota_search_plan.json"
DEFAULT_OUT_MD = ROOT / "research" / "sota_search_plan.md"
COUNTED_CODE_PATH = ROOT / "train_gpt.py"

TABLE_ROW_RE = re.compile(r"^\|(?P<cells>.+)\|$")
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
TOTAL_BYTES_RE = re.compile(r"(?P<bytes>\d[\d,]*)\s+total")
RUN_SHAPE_RE = re.compile(r"(?P<layers>\d+)x(?P<dim>\d+)")
CONFIG_LAYERS_RE = re.compile(r"layers=`?(?P<layers>\d+)`?")
CONFIG_DIM_RE = re.compile(r"dim=`?(?P<dim>\d+)`?")
CONFIG_KV_RE = re.compile(r"(?:kv|num_kv_heads)=`?(?P<kv>\d+)`?|KV(?P<kv_short>\d+)")
CONFIG_MLP_RE = re.compile(r"(?:mlp|mlp_mult)=`?(?P<mlp>\d+)`?|MLP(?P<mlp_short>\d+)")


@dataclass(frozen=True)
class Experiment:
    run: str
    hypothesis: str
    config: str
    runtime: str
    artifact_bytes: str
    metrics: str
    conclusion: str
    total_bytes: int | None
    clean_bpb: float | None
    shipped_bpb: float | None
    official: bool
    dense: bool
    tokenizer: str
    layers: int | None
    model_dim: int | None
    num_kv_heads: int | None
    mlp_mult: int | None


@dataclass(frozen=True)
class ShapeConfig:
    layers: int
    model_dim: int
    num_kv_heads: int
    mlp_mult: int
    init_mode: str
    evidence_weight: float
    notes: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.layers, self.model_dim)


@dataclass(frozen=True)
class Policy:
    name: str
    label: str
    quant_format: str
    int4_groups: tuple[str, ...]
    fp16_groups: tuple[str, ...]
    quality_gain: float
    bytes_delta_mb: float
    byte_risk: float
    notes: str


@dataclass
class CandidatePlan:
    rank: int
    name: str
    layers: int
    model_dim: int
    num_kv_heads: int
    mlp_mult: int
    init_mode: str
    export_policy: str
    estimated_clean_bpb: float
    estimated_shipped_bpb: float
    estimated_gap: float
    estimated_total_bytes: int
    legal: bool
    confidence: float
    search_score: float
    distance_to_target: float
    rationale: list[str]
    command: str


SHAPE_SPECS = (
    ShapeConfig(12, 608, 2, 2, "continue", 1.0, "Only official-path dense family with real life; best clean anchor at 1.3329."),
    ShapeConfig(14, 576, 2, 2, "scratch", 0.7, "Main scale-up fallback; local raw adapted result nearly tied with 12x608."),
    ShapeConfig(12, 576, 2, 2, "warmstart_expand", 0.55, "Smaller control with useful fchi-only export behavior."),
    ShapeConfig(13, 576, 2, 2, "warmstart_expand", 0.3, "Mathematically attractive but repeatedly weaker than 12x608."),
    ShapeConfig(12, 640, 2, 2, "warmstart_expand", 0.15, "Useful negative control; over-cap and weaker."),
    ShapeConfig(15, 576, 2, 2, "scratch", 0.1, "Speculative near-cap stretch candidate with very low confidence."),
)


POLICIES = (
    Policy(
        "fc_hi_int4",
        "Upper `mlp.fc` only int4",
        "mixed_int4_int8_packed_v2",
        ("fc_hi",),
        (),
        0.06,
        0.6,
        0.08,
        "Best byte-efficient local pattern family; conservative legalizer.",
    ),
    Policy(
        "fcproj_hi_int4",
        "Upper `mlp.fc + mlp.proj` int4",
        "mixed_int4_int8_packed_v2",
        ("fc_hi", "proj_hi"),
        (),
        0.0,
        0.0,
        0.0,
        "Current official baseline export target.",
    ),
    Policy(
        "int8_all",
        "Int8 everywhere",
        "int8_clean_per_row_v1",
        (),
        (),
        0.10,
        2.6,
        0.24,
        "Best generic way to buy back quality when current int4 path is too lossy.",
    ),
    Policy(
        "int8_tok_fp16",
        "Int8 everywhere + fp16 token embedding",
        "int8_clean_per_row_v1",
        (),
        ("tok",),
        0.12,
        4.6,
        0.42,
        "High-upside quality recovery with moderate byte pressure.",
    ),
    Policy(
        "fcproj_attnhi_fp16",
        "Upper `fc+proj` int4 + upper `attn.proj` fp16",
        "mixed_int4_int8_packed_v2",
        ("fc_hi", "proj_hi"),
        ("attn_hi",),
        0.14,
        4.8,
        0.46,
        "Best hybrid frontier direction when the export gap dominates.",
    ),
    Policy(
        "int8_projhi_fp16",
        "Int8 everywhere + upper `mlp.proj` fp16",
        "int8_clean_per_row_v1",
        (),
        ("proj_hi",),
        0.16,
        10.0,
        0.9,
        "Very strong quality recovery candidate, but typically over-cap unless offset elsewhere.",
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline SOTA planner for Parameter Golf. Reads repo evidence and ranks next model/export/training settings without running training."
    )
    p.add_argument("--experiment-table", type=Path, default=DEFAULT_EXPERIMENT_TABLE)
    p.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    p.add_argument("--target-bpb", type=float, default=1.1)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return p.parse_args()


def split_markdown_row(line: str) -> list[str] | None:
    m = TABLE_ROW_RE.match(line.strip())
    if not m:
        return None
    return [cell.strip() for cell in m.group("cells").split("|")]


def strip_ticks(text: str) -> str:
    return text.replace("`", "").strip()


def parse_total_bytes(text: str) -> int | None:
    m = TOTAL_BYTES_RE.search(strip_ticks(text))
    if not m:
        return None
    return int(m.group("bytes").replace(",", ""))


def parse_metric_pair(text: str) -> tuple[float | None, float | None]:
    clean = None
    shipped = None
    plain = strip_ticks(text).lower()
    pairs = re.findall(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", plain)
    if "clean" in plain and "post-quant" in plain and len(pairs) >= 2:
        clean = float(pairs[0][1])
        shipped = float(pairs[1][1])
        return clean, shipped
    if "clean" in plain and "roundtrip" in plain and len(pairs) >= 2:
        clean = float(pairs[0][1])
        shipped = float(pairs[1][1])
        return clean, shipped
    if pairs:
        shipped = float(pairs[-1][1])
        return clean, shipped
    vals = [float(v) for v in FLOAT_RE.findall(plain)]
    if vals:
        shipped = vals[-1]
    return clean, shipped


def infer_shape(run: str, config: str) -> tuple[int | None, int | None]:
    run_match = RUN_SHAPE_RE.search(run)
    if run_match:
        return int(run_match.group("layers")), int(run_match.group("dim"))
    cfg = strip_ticks(config)
    layers = None
    model_dim = None
    m = CONFIG_LAYERS_RE.search(cfg)
    if m:
        layers = int(m.group("layers"))
    m = CONFIG_DIM_RE.search(cfg)
    if m:
        model_dim = int(m.group("dim"))
    return layers, model_dim


def infer_int(cfg: str, pattern: re.Pattern[str], primary: str, secondary: str | None = None) -> int | None:
    m = pattern.search(strip_ticks(cfg))
    if not m:
        return None
    if m.group(primary):
        return int(m.group(primary))
    if secondary and m.group(secondary):
        return int(m.group(secondary))
    return None


def infer_tokenizer(text: str) -> str:
    lower = text.lower()
    if "spu4096" in lower or "unigram-4096" in lower or "u4k" in lower:
        return "u4k"
    if "sp1024" in lower:
        return "sp1024"
    return "unknown"


def parse_experiment_table(path: Path) -> list[Experiment]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[Experiment] = []
    for line in lines:
        cells = split_markdown_row(line)
        if not cells or len(cells) != 7:
            continue
        if cells[0].startswith("---") or cells[0].lower() == "run":
            continue
        run, hypothesis, config, runtime, artifact, metrics, conclusion = (strip_ticks(c) for c in cells)
        clean_bpb, shipped_bpb = parse_metric_pair(metrics)
        official = "official" in config.lower() or "fineweb10b_spu4096_docs" in config.lower()
        dense = "dense" in run.lower() or "dense" in hypothesis.lower()
        tokenizer = infer_tokenizer(" ".join((run, config, hypothesis)))
        layers, model_dim = infer_shape(run, config)
        rows.append(
            Experiment(
                run=run,
                hypothesis=hypothesis,
                config=config,
                runtime=runtime,
                artifact_bytes=artifact,
                metrics=metrics,
                conclusion=conclusion,
                total_bytes=parse_total_bytes(artifact),
                clean_bpb=clean_bpb,
                shipped_bpb=shipped_bpb,
                official=official,
                dense=dense,
                tokenizer=tokenizer,
                layers=layers,
                model_dim=model_dim,
                num_kv_heads=infer_int(config, CONFIG_KV_RE, "kv", "kv_short"),
                mlp_mult=infer_int(config, CONFIG_MLP_RE, "mlp", "mlp_short"),
            )
        )
    return rows


def find_best(rows: Iterable[Experiment], *, predicate) -> Experiment | None:
    candidates = [row for row in rows if predicate(row) and row.shipped_bpb is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row.shipped_bpb)


def average(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def shape_anchor(rows: list[Experiment], shape: ShapeConfig) -> tuple[float, int, list[str]]:
    exact = [r for r in rows if r.layers == shape.layers and r.model_dim == shape.model_dim and r.dense and r.tokenizer == "u4k" and r.shipped_bpb is not None]
    rationale: list[str] = []
    if exact:
        best = min(exact, key=lambda r: r.shipped_bpb or math.inf)
        rationale.append(f"Closest measured shape anchor is `{best.run}` at `{best.shipped_bpb:.4f}` shipped bpb.")
        if best.official and best.clean_bpb is not None:
            rationale.append(f"This shape also has official-path evidence with clean `{best.clean_bpb:.4f}` bpb.")
        return best.shipped_bpb or 99.0, best.total_bytes or 16_000_000, rationale

    comparable = [r for r in rows if r.dense and r.tokenizer == "u4k" and r.layers and r.model_dim and r.shipped_bpb is not None]
    if not comparable:
        return 99.0, 16_000_000, ["No dense U4K evidence was found; using a very pessimistic prior."]
    nearest = min(comparable, key=lambda r: abs((r.layers or 0) - shape.layers) * 0.3 + abs((r.model_dim or 0) - shape.model_dim) / 64.0)
    scale_penalty = 0.025 * abs((nearest.layers or shape.layers) - shape.layers) + 0.02 * abs((nearest.model_dim or shape.model_dim) - shape.model_dim) / 32.0
    anchor_bpb = (nearest.shipped_bpb or 99.0) + scale_penalty
    anchor_bytes = int((nearest.total_bytes or 16_000_000) * ((shape.layers * shape.model_dim * shape.model_dim) / max((nearest.layers or shape.layers) * (nearest.model_dim or shape.model_dim) ** 2, 1)) ** 0.85)
    rationale.append(f"No exact measured anchor exists; extrapolated from `{nearest.run}` with a scale penalty of `{scale_penalty:.4f}` bpb.")
    return anchor_bpb, anchor_bytes, rationale


TOP_GROUP_RE = re.compile(r"^(?P<kind>fc|proj|attn)_top(?P<count>\d+)$")


def _group_layers(layers: int, group: str) -> range:
    if group.endswith("_hi"):
        return range(layers // 2, layers)
    m = TOP_GROUP_RE.match(group)
    if m:
        count = max(1, min(layers, int(m.group("count"))))
        return range(layers - count, layers)
    return range(0)


def build_group_patterns(layers: int, groups: tuple[str, ...]) -> tuple[str, ...]:
    pats: list[str] = []
    for group in groups:
        if group.startswith("fc_"):
            pats.extend(f"blocks.{layer}.mlp.fc.weight" for layer in _group_layers(layers, group))
        elif group.startswith("proj_"):
            pats.extend(f"blocks.{layer}.mlp.proj.weight" for layer in _group_layers(layers, group))
        elif group.startswith("attn_"):
            pats.extend(f"blocks.{layer}.attn.proj.weight" for layer in _group_layers(layers, group))
    return tuple(pats)


def build_fp16_patterns(groups: tuple[str, ...], layers: int) -> tuple[str, ...]:
    pats: list[str] = []
    if "tok" in groups:
        pats.append("tok_emb.weight")
    non_tok = tuple(g for g in groups if g != "tok")
    if non_tok:
        pats.extend(build_group_patterns(layers, non_tok))
    return tuple(pats)


def policy_estimated_bytes(base_bytes: int, policy: Policy, shape: ShapeConfig) -> int:
    adjusted = base_bytes + int(policy.bytes_delta_mb * 1_000_000)
    if shape.layers == 15 and shape.model_dim == 576:
        adjusted += 400_000
    return adjusted


def base_clean_from_official(rows: list[Experiment]) -> float:
    best = min(
        (
            r.clean_bpb
            for r in rows
            if r.official and r.dense and r.tokenizer == "u4k" and r.clean_bpb is not None
        ),
        default=1.65,
    )
    return best


def public_baseline(rows: list[Experiment]) -> float | None:
    for row in rows:
        if row.run == "baseline_public_10min":
            return row.shipped_bpb
    return None


def shape_quality_prior(rows: list[Experiment], shape: ShapeConfig, official_clean_anchor: float) -> tuple[float, int, list[str]]:
    anchor_bpb, anchor_total_bytes, rationale = shape_anchor(rows, shape)
    if shape.key == (12, 608):
        predicted_clean = official_clean_anchor
        rationale.append("Official dense anchor keeps this shape's clean bpb prior at the measured best official value.")
    else:
        transfer_penalty = {"scratch": 0.24, "warmstart_expand": 0.12, "continue": 0.0}[shape.init_mode]
        predicted_clean = min(anchor_bpb - 0.18 + transfer_penalty, anchor_bpb - 0.05)
        rationale.append(
            f"Converted local shipped evidence into a clean prior by subtracting a conservative export gap and adding `{transfer_penalty:.2f}` init penalty."
        )
    return predicted_clean, anchor_total_bytes, rationale


def estimate_candidate(rows: list[Experiment], shape: ShapeConfig, policy: Policy, target_bpb: float) -> CandidatePlan:
    code_bytes = COUNTED_CODE_PATH.stat().st_size
    official_clean_anchor = base_clean_from_official(rows)
    baseline = public_baseline(rows)
    clean_bpb, anchor_total_bytes, rationale = shape_quality_prior(rows, shape, official_clean_anchor)

    init_penalty = {"continue": 0.0, "warmstart_expand": 0.05, "scratch": 0.14}[shape.init_mode]
    training_bonus = 0.03 if shape.init_mode == "continue" else 0.0
    estimated_gap = max(0.11, 0.2772 - policy.quality_gain + init_penalty - training_bonus)
    estimated_shipped = clean_bpb + estimated_gap
    estimated_total = policy_estimated_bytes(anchor_total_bytes, policy, shape)
    legal = estimated_total <= 16_000_000
    confidence = max(0.05, min(0.95, shape.evidence_weight - policy.byte_risk * 0.15 - (0.12 if not legal else 0.0)))
    distance = estimated_shipped - target_bpb

    score = distance
    score += policy.byte_risk * 0.08
    score += (1.0 - confidence) * 0.15
    if not legal:
        score += 0.75 + (estimated_total - 16_000_000) / 4_000_000
    if baseline is not None and estimated_shipped > baseline:
        score += min(0.35, (estimated_shipped - baseline) * 0.1)

    rationale.extend(
        [
            f"Policy prior `{policy.name}` assumes `{policy.quality_gain:.2f}` bpb of export-gap recovery for `{policy.bytes_delta_mb:.1f}MB` extra pressure.",
            f"Estimated clean-to-shipped gap is `{estimated_gap:.4f}` bpb after init-mode adjustment.",
            f"Current target remains `{target_bpb:.3f}` shipped bpb, so this candidate still needs `{max(distance, 0.0):.4f}` bpb of further gain." if distance > 0 else "This candidate is estimated to reach or beat the target.",
        ]
    )

    run_id = f"planner_dense_u4k_{shape.layers}x{shape.model_dim}_{policy.name}"
    command = build_command(run_id, shape, policy)
    return CandidatePlan(
        rank=0,
        name=run_id,
        layers=shape.layers,
        model_dim=shape.model_dim,
        num_kv_heads=shape.num_kv_heads,
        mlp_mult=shape.mlp_mult,
        init_mode=shape.init_mode,
        export_policy=policy.name,
        estimated_clean_bpb=clean_bpb,
        estimated_shipped_bpb=estimated_shipped,
        estimated_gap=estimated_gap,
        estimated_total_bytes=estimated_total + max(code_bytes - 66_325, 0),
        legal=legal,
        confidence=confidence,
        search_score=score,
        distance_to_target=distance,
        rationale=rationale,
        command=command,
    )


def build_command(run_id: str, shape: ShapeConfig, policy: Policy) -> str:
    int4_patterns = ",".join(build_group_patterns(shape.layers, policy.int4_groups))
    fp16_patterns = ",".join(build_fp16_patterns(policy.fp16_groups, shape.layers))
    env_parts = [
        f"RUN_ID={run_id}",
        "MAX_WALLCLOCK_SECONDS=600",
        f"NUM_LAYERS={shape.layers}",
        f"NUM_UNIQUE_LAYERS={shape.layers}",
        f"MODEL_DIM={shape.model_dim}",
        "NUM_HEADS=8",
        f"NUM_KV_HEADS={shape.num_kv_heads}",
        f"MLP_MULT={shape.mlp_mult}",
    ]
    if shape.init_mode == "continue":
        env_parts.append("INIT_MODEL_PATH=./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt")
    else:
        env_parts.append("INIT_MODEL_PATH=")

    if policy.quant_format == "int8_clean_per_row_v1":
        env_parts.extend(
            [
                "QUANT_FORMAT=int8_clean_per_row_v1",
                "TARGET_EXPORT_NAME_PATTERNS=",
                "INT4_NAME_PATTERNS=",
                "TRAIN_QAT_NAME_PATTERNS=__never__",
                "TRAIN_COMPRESSION_AWARE_WEIGHT=0",
                "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=",
                "LFQAT_KL_WEIGHT=0",
                "LFQAT_FISHER_WEIGHT=0",
            ]
        )
    else:
        env_parts.extend(
            [
                "QUANT_FORMAT=mixed_int4_int8_packed_v2",
                f"TARGET_EXPORT_NAME_PATTERNS={int4_patterns}",
                f"INT4_NAME_PATTERNS={int4_patterns}",
                f"TRAIN_QAT_NAME_PATTERNS={int4_patterns or '__never__'}",
                "TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015",
                f"TRAIN_COMPRESSION_AWARE_NAME_PATTERNS={int4_patterns}",
                "LFQAT_KL_WEIGHT=0.05",
                "LFQAT_FISHER_WEIGHT=0.01",
            ]
        )
    if fp16_patterns:
        env_parts.append(f"INT8_KEEP_FLOAT_FP16_NAME_PATTERNS={fp16_patterns}")
    return " \\\n  ".join(env_parts + ["scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh"])


def build_candidates(rows: list[Experiment], target_bpb: float) -> list[CandidatePlan]:
    plans: list[CandidatePlan] = []
    for shape in SHAPE_SPECS:
        for policy in POLICIES:
            if shape.key == (13, 576) and policy.name in {"int8_projhi_fp16", "fcproj_attnhi_fp16"}:
                continue
            if shape.key == (12, 640) and policy.name not in {"fc_hi_int4", "fcproj_hi_int4"}:
                continue
            plans.append(estimate_candidate(rows, shape, policy, target_bpb))
    plans.sort(key=lambda plan: plan.search_score)
    for idx, plan in enumerate(plans, start=1):
        plan.rank = idx
    return plans


def best_scaleups(plans: list[CandidatePlan], limit: int = 3) -> list[CandidatePlan]:
    anchor_complexity = 12 * 608 * 608
    seen: set[tuple[int, int]] = set()
    winners: list[CandidatePlan] = []
    for plan in plans:
        if plan.init_mode == "continue":
            continue
        if plan.layers * plan.model_dim * plan.model_dim <= anchor_complexity:
            continue
        key = (plan.layers, plan.model_dim)
        if key in seen:
            continue
        seen.add(key)
        winners.append(plan)
        if len(winners) >= limit:
            break
    return winners


def render_markdown(plans: list[CandidatePlan], rows: list[Experiment], args: argparse.Namespace) -> str:
    baseline = public_baseline(rows)
    scaleups = best_scaleups(plans)
    lines = [
        "# SOTA Search Plan",
        "",
        f"- Target shipped bpb: `{args.target_bpb:.4f}`",
        f"- Counted code bytes: `{COUNTED_CODE_PATH.stat().st_size}`",
        f"- Public baseline shipped bpb: `{baseline:.4f}`" if baseline is not None else "- Public baseline shipped bpb: `unknown`",
        "",
        "## Top Candidates",
        "",
        "| Rank | Candidate | Shape | Init | Export policy | Est. shipped bpb | Est. bytes | Legal | Confidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for plan in plans[: args.top_k]:
        lines.append(
            f"| `{plan.rank}` | `{plan.name}` | `{plan.layers}x{plan.model_dim}` | `{plan.init_mode}` | "
            f"`{plan.export_policy}` | `{plan.estimated_shipped_bpb:.4f}` | `{plan.estimated_total_bytes}` | "
            f"`{'yes' if plan.legal else 'no'}` | `{plan.confidence:.2f}` |"
        )
    lines.extend(["", "## High-Upside Scale-Ups", ""])
    lines.append("| Candidate | Shape | Init | Export policy | Est. shipped bpb | Est. bytes | Legal |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for plan in scaleups:
        lines.append(
            f"| `{plan.name}` | `{plan.layers}x{plan.model_dim}` | `{plan.init_mode}` | `{plan.export_policy}` | "
            f"`{plan.estimated_shipped_bpb:.4f}` | `{plan.estimated_total_bytes}` | `{'yes' if plan.legal else 'no'}` |"
        )
    lines.extend(["", "## Commands", ""])
    for plan in plans[: min(args.top_k, 3)]:
        lines.extend(
            [
                f"### `{plan.name}`",
                "",
                f"- Estimated clean bpb: `{plan.estimated_clean_bpb:.4f}`",
                f"- Estimated clean-to-shipped gap: `{plan.estimated_gap:.4f}`",
                f"- Distance to `{args.target_bpb:.4f}`: `{plan.distance_to_target:.4f}`",
                "",
                "```bash",
                plan.command,
                "```",
                "",
                "Rationale:",
            ]
        )
        for item in plan.rationale:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = parse_experiment_table(args.experiment_table)
    plans = build_candidates(rows, args.target_bpb)
    payload = {
        "target_bpb": args.target_bpb,
        "counted_code_bytes": COUNTED_CODE_PATH.stat().st_size,
        "public_baseline_bpb": public_baseline(rows),
        "top_candidates": [asdict(plan) for plan in plans[: args.top_k]],
        "high_upside_scaleups": [asdict(plan) for plan in best_scaleups(plans)],
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(plans, rows, args), encoding="utf-8")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    best = plans[0]
    print(
        f"Best offline candidate: {best.name} "
        f"(est_shipped_bpb={best.estimated_shipped_bpb:.4f}, est_total_bytes={best.estimated_total_bytes}, legal={best.legal})"
    )


if __name__ == "__main__":
    main()
