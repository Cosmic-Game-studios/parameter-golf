#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.plan_sota_search import (
    COUNTED_CODE_PATH,
    POLICIES,
    SHAPE_SPECS,
    Policy,
    ShapeConfig,
    build_fp16_patterns,
    build_group_patterns,
    parse_experiment_table,
    policy_estimated_bytes,
    public_baseline,
    shape_quality_prior,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "research" / "sota_trial_search.json"
DEFAULT_OUT_MD = ROOT / "research" / "sota_trial_search.md"


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    applies_to: tuple[str, ...]
    lr_schedule: str
    lr_warmup_iters: int
    min_lr_scale: float
    tied_embed_lr: float
    matrix_lr: float
    scalar_lr: float
    train_batch_tokens: int
    val_batch_size: int
    lfqat_start_step: int
    lfqat_full_step: int
    lfqat_min_prob: float
    lfqat_max_prob: float
    lfqat_kl_weight: float
    lfqat_fisher_weight: float
    train_compression_aware_weight: float
    clean_delta: float
    gap_delta_mixed: float
    gap_delta_int8: float
    runtime_multiplier: float
    notes: str


@dataclass
class TrialCandidate:
    rank: int
    stage: str
    name: str
    layers: int
    model_dim: int
    num_kv_heads: int
    mlp_mult: int
    init_mode: str
    recipe: str
    export_policy: str
    estimated_clean_bpb: float
    estimated_shipped_bpb: float
    estimated_gap: float
    estimated_total_bytes: int
    estimated_runtime_ratio: float
    legal: bool
    optimistic_gap_floor: float
    best_case_shipped_bpb: float
    export_headroom_bpb: float
    clean_gain_needed_after_best_export: float
    target_band: str
    stage1_objective: float
    confidence: float
    rationale: list[str]
    screen_command: str
    final_8xh100_command: str


RECIPE_LIBRARY = (
    RecipeSpec(
        "continue_lfqat_base",
        ("continue",),
        "cosine",
        20,
        0.2,
        0.002,
        0.0015,
        0.0015,
        524_288,
        524_288,
        0,
        0,
        1.0,
        1.0,
        0.05,
        0.01,
        0.0015,
        0.0,
        0.0,
        0.0,
        1.0,
        "Best current official continuation recipe anchor.",
    ),
    RecipeSpec(
        "continue_lfqat_ramp",
        ("continue",),
        "cosine",
        20,
        0.15,
        0.0022,
        0.0016,
        0.0016,
        524_288,
        524_288,
        0,
        40,
        0.2,
        1.0,
        0.05,
        0.01,
        0.0015,
        -0.02,
        -0.02,
        0.01,
        1.03,
        "Gentler export-aware continuation for shipping-gap recovery.",
    ),
    RecipeSpec(
        "continue_int8_recovery",
        ("continue",),
        "cosine",
        20,
        0.2,
        0.0018,
        0.0012,
        0.0012,
        524_288,
        524_288,
        0,
        20,
        0.1,
        0.7,
        0.02,
        0.005,
        0.0005,
        -0.01,
        0.03,
        -0.03,
        1.0,
        "Bias toward int8 export recovery with less mixed-precision pressure.",
    ),
    RecipeSpec(
        "warmstart_lfqat",
        ("warmstart_expand",),
        "constant",
        0,
        0.0,
        0.004,
        0.003,
        0.003,
        524_288,
        524_288,
        0,
        30,
        0.15,
        1.0,
        0.05,
        0.01,
        0.0015,
        -0.03,
        -0.01,
        0.02,
        1.0,
        "Matches the strongest local LFQAT continuation family.",
    ),
    RecipeSpec(
        "warmstart_plain_tail",
        ("warmstart_expand",),
        "constant",
        0,
        0.0,
        0.006,
        0.005,
        0.005,
        524_288,
        524_288,
        0,
        0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        -0.01,
        0.03,
        0.0,
        0.96,
        "Cheap warm-start continuation; useful control but historically weaker than LFQAT.",
    ),
    RecipeSpec(
        "scratch_cosine_stable",
        ("scratch",),
        "cosine",
        20,
        0.2,
        0.0035,
        0.0025,
        0.0025,
        524_288,
        524_288,
        0,
        60,
        0.15,
        0.95,
        0.05,
        0.01,
        0.0015,
        -0.05,
        -0.02,
        0.01,
        1.08,
        "Most stable scratch recipe for a larger dense candidate.",
    ),
    RecipeSpec(
        "scratch_cosine_fast",
        ("scratch",),
        "cosine",
        30,
        0.1,
        0.0045,
        0.0035,
        0.0035,
        524_288,
        524_288,
        10,
        80,
        0.1,
        0.9,
        0.04,
        0.008,
        0.001,
        -0.03,
        -0.01,
        0.02,
        0.98,
        "Higher-risk scratch recipe that may learn faster in the 10-minute budget.",
    ),
)

TRIAL_SHAPES = SHAPE_SPECS + (
    ShapeConfig(14, 608, 2, 2, "scratch", 0.22, "Aggressive larger dense moonshot near the edge of the byte/runtime budget."),
    ShapeConfig(16, 576, 2, 2, "scratch", 0.18, "Deeper moonshot that may buy raw quality if export remains disciplined."),
    ShapeConfig(15, 608, 2, 2, "scratch", 0.12, "High-risk near-cap stretch target for explicit 1.1 search."),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline staged H100 search planner for Parameter Golf.")
    p.add_argument("--experiment-table", type=Path, default=ROOT / "research" / "experiment_table.md")
    p.add_argument("--target-bpb", type=float, default=1.1)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return p.parse_args()


def base_gap(shape: ShapeConfig) -> float:
    if shape.init_mode == "continue":
        return 0.2772
    if shape.init_mode == "warmstart_expand":
        return 0.22
    return 0.26


def estimate_runtime_ratio(shape: ShapeConfig, recipe: RecipeSpec) -> float:
    anchor = 12 * 608 * 608
    complexity = shape.layers * shape.model_dim * shape.model_dim
    return (complexity / anchor) * recipe.runtime_multiplier


def compatible(shape: ShapeConfig, recipe: RecipeSpec, policy: Policy) -> bool:
    if shape.init_mode not in recipe.applies_to:
        return False
    if shape.key == (13, 576) and policy.name in {"fcproj_attnhi_fp16", "int8_projhi_fp16"}:
        return False
    if shape.key == (12, 640) and policy.name not in {"fc_hi_int4", "fcproj_hi_int4"}:
        return False
    if shape.key == (15, 576) and recipe.name == "warmstart_lfqat":
        return False
    if shape.key in {(14, 608), (15, 608), (16, 576)} and recipe.name.startswith("warmstart"):
        return False
    return True


def recipe_gap_delta(recipe: RecipeSpec, policy: Policy) -> float:
    if policy.quant_format == "int8_clean_per_row_v1":
        return recipe.gap_delta_int8
    return recipe.gap_delta_mixed


def optimistic_gap_floor(shape: ShapeConfig, policy: Policy) -> float:
    floor = 0.08 if shape.init_mode == "continue" else 0.10
    if policy.name == "fcproj_attnhi_fp16":
        floor -= 0.02
    elif policy.name in {"int8_tok_fp16", "int8_projhi_fp16"}:
        floor -= 0.015
    elif policy.name == "int8_all":
        floor -= 0.01
    elif policy.name == "fc_hi_int4":
        floor += 0.01
    if shape.key in {(14, 608), (15, 608), (16, 576)}:
        floor += 0.02
    return max(0.05, floor)


def classify_target_band(best_case_shipped: float, target_bpb: float) -> str:
    remaining = best_case_shipped - target_bpb
    if remaining <= 0.0:
        return "target-reachable"
    if remaining <= 0.10:
        return "target-stretch"
    if remaining <= 0.25:
        return "target-far"
    return "target-remote"


def estimate_trial(rows, shape: ShapeConfig, recipe: RecipeSpec, policy: Policy, target_bpb: float) -> TrialCandidate:
    clean_prior, anchor_total_bytes, rationale = shape_quality_prior(rows, shape, min(r.clean_bpb for r in rows if r.clean_bpb is not None and r.official and r.dense and r.tokenizer == "u4k"))
    clean = clean_prior + recipe.clean_delta
    gap = max(0.08, base_gap(shape) - policy.quality_gain + recipe_gap_delta(recipe, policy))
    shipped = clean + gap
    total_bytes = policy_estimated_bytes(anchor_total_bytes, policy, shape)
    if shape.init_mode == "scratch":
        total_bytes += 150_000
    total_bytes += max(COUNTED_CODE_PATH.stat().st_size - 66_325, 0)
    legal = total_bytes <= 16_000_000
    runtime_ratio = estimate_runtime_ratio(shape, recipe)
    gap_floor = optimistic_gap_floor(shape, policy)
    best_case_shipped = clean + gap_floor
    export_headroom = max(0.0, gap - gap_floor)
    clean_gain_needed = max(0.0, best_case_shipped - target_bpb)
    target_band = classify_target_band(best_case_shipped, target_bpb)
    conf = max(
        0.05,
        min(
            0.96,
            shape.evidence_weight
            - max(0.0, runtime_ratio - 1.0) * 0.18
            - (0.12 if shape.init_mode == "scratch" else 0.0)
            - policy.byte_risk * 0.12,
        ),
    )
    objective = shipped
    objective += max(0.0, (total_bytes - 16_000_000) / 1_000_000) * 0.7
    objective += max(0.0, runtime_ratio - 1.0) * 0.25
    objective += gap * 0.35
    objective += clean_gain_needed * 0.9
    objective -= export_headroom * 0.15
    if shape.key == (13, 576):
        objective += 0.08
    if shape.key == (12, 640):
        objective += 0.12
    if shape.key == (15, 576):
        objective += 0.18
    if shape.key == (14, 608):
        objective += 0.14
    if shape.key == (15, 608):
        objective += 0.22
    if shape.key == (16, 576):
        objective += 0.18

    rationale = list(rationale)
    rationale.extend(
        [
            f"Recipe `{recipe.name}` contributes `{recipe.clean_delta:+.4f}` clean-bpb delta and `{recipe_gap_delta(recipe, policy):+.4f}` export-gap delta.",
            f"Policy `{policy.name}` contributes `{policy.quality_gain:.4f}` expected shipped-bpb recovery for roughly `{policy.bytes_delta_mb:.1f}MB` extra bytes.",
            f"Estimated runtime ratio vs the current `12x608` official anchor is `{runtime_ratio:.3f}`.",
            f"Estimated clean `{clean:.4f}` and shipped `{shipped:.4f}` bpb still leave `{max(shipped - target_bpb, 0.0):.4f}` to the `1.1` goal.",
            f"With optimistic export floor `{gap_floor:.4f}`, best-case shipped becomes `{best_case_shipped:.4f}` and still needs `{clean_gain_needed:.4f}` more clean improvement.",
        ]
    )
    name = f"trial_{shape.layers}x{shape.model_dim}_{recipe.name}_{policy.name}"
    return TrialCandidate(
        rank=0,
        stage="stage1_screen",
        name=name,
        layers=shape.layers,
        model_dim=shape.model_dim,
        num_kv_heads=shape.num_kv_heads,
        mlp_mult=shape.mlp_mult,
        init_mode=shape.init_mode,
        recipe=recipe.name,
        export_policy=policy.name,
        estimated_clean_bpb=clean,
        estimated_shipped_bpb=shipped,
        estimated_gap=gap,
        estimated_total_bytes=total_bytes,
        estimated_runtime_ratio=runtime_ratio,
        legal=legal,
        optimistic_gap_floor=gap_floor,
        best_case_shipped_bpb=best_case_shipped,
        export_headroom_bpb=export_headroom,
        clean_gain_needed_after_best_export=clean_gain_needed,
        target_band=target_band,
        stage1_objective=objective,
        confidence=conf,
        rationale=rationale,
        screen_command=build_trial_command(name, shape, recipe, policy, nproc=1),
        final_8xh100_command=build_trial_command(name.replace("trial_", "final_"), shape, recipe, policy, nproc=8),
    )


def build_trial_command(run_id: str, shape: ShapeConfig, recipe: RecipeSpec, policy: Policy, *, nproc: int) -> str:
    int4_patterns = ",".join(build_group_patterns(shape.layers, policy.int4_groups))
    fp16_patterns = ",".join(build_fp16_patterns(policy.fp16_groups, shape.layers))
    env_parts = [
        f"RUN_ID={run_id}",
        f"NPROC_PER_NODE={nproc}",
        "MAX_WALLCLOCK_SECONDS=600",
        f"NUM_LAYERS={shape.layers}",
        f"NUM_UNIQUE_LAYERS={shape.layers}",
        f"MODEL_DIM={shape.model_dim}",
        "NUM_HEADS=8",
        f"NUM_KV_HEADS={shape.num_kv_heads}",
        f"MLP_MULT={shape.mlp_mult}",
        f"LR_SCHEDULE={recipe.lr_schedule}",
        f"LR_WARMUP_ITERS={recipe.lr_warmup_iters}",
        f"MIN_LR_SCALE={recipe.min_lr_scale}",
        f"TIED_EMBED_LR={recipe.tied_embed_lr}",
        f"MATRIX_LR={recipe.matrix_lr}",
        f"SCALAR_LR={recipe.scalar_lr}",
        f"TRAIN_BATCH_TOKENS={recipe.train_batch_tokens}",
        f"VAL_BATCH_SIZE={recipe.val_batch_size}",
        "ITERATIONS=2500",
        "VAL_LOSS_EVERY=100",
        "TRAIN_LOG_EVERY=25",
        f"LFQAT_START_STEP={recipe.lfqat_start_step}",
        f"LFQAT_FULL_STEP={recipe.lfqat_full_step}",
        f"LFQAT_MIN_PROB={recipe.lfqat_min_prob}",
        f"LFQAT_MAX_PROB={recipe.lfqat_max_prob}",
        f"LFQAT_KL_WEIGHT={recipe.lfqat_kl_weight if policy.quant_format != 'int8_clean_per_row_v1' else max(recipe.lfqat_kl_weight * 0.4, 0.0)}",
        f"LFQAT_FISHER_WEIGHT={recipe.lfqat_fisher_weight if policy.quant_format != 'int8_clean_per_row_v1' else max(recipe.lfqat_fisher_weight * 0.4, 0.0)}",
        f"TRAIN_COMPRESSION_AWARE_WEIGHT={recipe.train_compression_aware_weight if policy.quant_format != 'int8_clean_per_row_v1' else max(recipe.train_compression_aware_weight * 0.2, 0.0)}",
        f"INIT_MODEL_PATH={'./checkpoints/runpod_dense_u4k_12x608_lfqat_continue80m_20260319_final_model.pt' if shape.init_mode == 'continue' else ''}",
    ]
    if policy.quant_format == "int8_clean_per_row_v1":
        env_parts.extend(
            [
                "QUANT_FORMAT=int8_clean_per_row_v1",
                "TARGET_EXPORT_NAME_PATTERNS=",
                "INT4_NAME_PATTERNS=",
                "TRAIN_QAT_NAME_PATTERNS=__never__",
                "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=",
            ]
        )
    else:
        env_parts.extend(
            [
                "QUANT_FORMAT=mixed_int4_int8_packed_v2",
                f"TARGET_EXPORT_NAME_PATTERNS={int4_patterns}",
                f"INT4_NAME_PATTERNS={int4_patterns}",
                f"TRAIN_QAT_NAME_PATTERNS={int4_patterns or '__never__'}",
                f"TRAIN_COMPRESSION_AWARE_NAME_PATTERNS={int4_patterns}",
            ]
        )
    if fp16_patterns:
        env_parts.append(f"INT8_KEEP_FLOAT_FP16_NAME_PATTERNS={fp16_patterns}")
    return " \\\n  ".join(env_parts + ["scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh"])


def search_trials(rows, target_bpb: float) -> list[TrialCandidate]:
    trials: list[TrialCandidate] = []
    for shape in TRIAL_SHAPES:
        for recipe in RECIPE_LIBRARY:
            for policy in POLICIES:
                if not compatible(shape, recipe, policy):
                    continue
                trials.append(estimate_trial(rows, shape, recipe, policy, target_bpb))
    trials.sort(key=lambda trial: trial.stage1_objective)
    for idx, trial in enumerate(trials, start=1):
        trial.rank = idx
        if idx <= 6:
            trial.stage = "stage1_screen"
        elif idx <= 12:
            trial.stage = "stage2_export_finalize"
        else:
            trial.stage = "stage3_backup"
    return trials


def render_markdown(trials: list[TrialCandidate], rows, args: argparse.Namespace) -> str:
    baseline = public_baseline(rows)
    best = trials[0]
    closest_best_case = min(trials, key=lambda trial: (trial.best_case_shipped_bpb, trial.stage1_objective))
    stretch_trials = [
        trial
        for trial in trials
        if trial.legal and trial.target_band in {"target-reachable", "target-stretch"}
    ]
    lines = [
        "# Staged SOTA Trial Search",
        "",
        "Objective:",
        "",
        "`J = shipped_val_bpb + lambda1*byte_penalty + lambda2*runtime_penalty + lambda3*clean_to_shipped_gap`",
        "",
        f"- Target shipped bpb: `{args.target_bpb:.4f}`",
        f"- Counted code bytes: `{COUNTED_CODE_PATH.stat().st_size}`",
        f"- Public baseline shipped bpb: `{baseline:.4f}`" if baseline is not None else "- Public baseline shipped bpb: `unknown`",
        "",
        "## 1.1 Verdict",
        "",
        (
            f"- Best evidence-backed candidate today is `{best.name}` at estimated shipped `{best.estimated_shipped_bpb:.4f}` bpb."
        ),
        (
            f"- Even under the searcher's optimistic export floor, the closest candidate is `{closest_best_case.name}` at `{closest_best_case.best_case_shipped_bpb:.4f}` bpb."
        ),
        (
            f"- Clean improvement still needed after best-case export: `{closest_best_case.clean_gain_needed_after_best_export:.4f}` bpb."
        ),
        (
            "- Current verdict: no legal candidate in the present search space is yet target-reachable or even target-stretch for `1.1`; "
            "the current family still needs a materially stronger raw checkpoint, not just marginal export tuning."
            if not stretch_trials
            else f"- Current verdict: `{len(stretch_trials)}` legal candidate(s) are within the searcher's target-stretch band."
        ),
        "",
        "## Stage 1 Screeners",
        "",
        "| Rank | Candidate | Shape | Recipe | Export | Est. shipped bpb | Best-case shipped | Clean gain still needed | Legal |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for trial in trials[: min(args.top_k, 8)]:
        lines.append(
            f"| `{trial.rank}` | `{trial.name}` | `{trial.layers}x{trial.model_dim}` | `{trial.recipe}` | "
            f"`{trial.export_policy}` | `{trial.estimated_shipped_bpb:.4f}` | `{trial.best_case_shipped_bpb:.4f}` | "
            f"`{trial.clean_gain_needed_after_best_export:.4f}` | `{'yes' if trial.legal else 'no'}` |"
        )
    lines.extend(["", "## Best Moonshots", ""])
    moonshots = [trial for trial in trials if trial.init_mode != "continue"][:3]
    lines.append("| Candidate | Shape | Recipe | Export | Est. shipped bpb | Best-case shipped | Clean gain still needed | Legal |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for trial in moonshots:
        lines.append(
            f"| `{trial.name}` | `{trial.layers}x{trial.model_dim}` | `{trial.recipe}` | `{trial.export_policy}` | "
            f"`{trial.estimated_shipped_bpb:.4f}` | `{trial.best_case_shipped_bpb:.4f}` | "
            f"`{trial.clean_gain_needed_after_best_export:.4f}` | `{'yes' if trial.legal else 'no'}` |"
        )
    lines.extend(["", "## Commands", ""])
    for trial in trials[:3]:
        lines.extend(
            [
                f"### `{trial.name}`",
                "",
                f"- Estimated clean bpb: `{trial.estimated_clean_bpb:.4f}`",
                f"- Estimated shipped bpb: `{trial.estimated_shipped_bpb:.4f}`",
                f"- Estimated gap: `{trial.estimated_gap:.4f}`",
                f"- Best-case shipped with optimistic export: `{trial.best_case_shipped_bpb:.4f}`",
                f"- Clean gain still needed after best export: `{trial.clean_gain_needed_after_best_export:.4f}`",
                f"- Estimated bytes: `{trial.estimated_total_bytes}`",
                "",
                "1xH100 screen:",
                "```bash",
                trial.screen_command,
                "```",
                "",
                "8xH100 final:",
                "```bash",
                trial.final_8xh100_command,
                "```",
                "",
                "Rationale:",
            ]
        )
        for item in trial.rationale:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = parse_experiment_table(args.experiment_table)
    trials = search_trials(rows, args.target_bpb)
    best = trials[0]
    closest_best_case = min(trials, key=lambda trial: (trial.best_case_shipped_bpb, trial.stage1_objective))
    stretch_trials = [
        trial
        for trial in trials
        if trial.legal and trial.target_band in {"target-reachable", "target-stretch"}
    ]
    payload = {
        "target_bpb": args.target_bpb,
        "counted_code_bytes": COUNTED_CODE_PATH.stat().st_size,
        "public_baseline_bpb": public_baseline(rows),
        "top_trials": [asdict(trial) for trial in trials[: args.top_k]],
        "moonshots": [asdict(trial) for trial in [trial for trial in trials if trial.init_mode != "continue"][:3]],
        "goal_seek_summary": {
            "best_estimated_trial": best.name,
            "best_estimated_shipped_bpb": best.estimated_shipped_bpb,
            "best_case_shipped_bpb": best.best_case_shipped_bpb,
            "clean_gain_needed_after_best_export": best.clean_gain_needed_after_best_export,
            "closest_best_case_trial": {
                "trial_id": closest_best_case.name,
                "best_case_shipped_bpb": closest_best_case.best_case_shipped_bpb,
                "clean_gain_needed_after_best_export": closest_best_case.clean_gain_needed_after_best_export,
                "estimated_total_bytes": closest_best_case.estimated_total_bytes,
                "estimated_runtime_ratio": closest_best_case.estimated_runtime_ratio,
                "target_band": closest_best_case.target_band,
            },
            "credible_trial_count": len(stretch_trials),
            "credible_trials": [trial.name for trial in stretch_trials[:5]],
            "verdict": (
                "no_current_candidate_is_target_stretch"
                if not stretch_trials
                else "target_stretch_candidates_exist"
            ),
        },
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(trials, rows, args), encoding="utf-8")
    best = trials[0]
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(
        f"Best staged trial: {best.name} "
        f"(est_shipped_bpb={best.estimated_shipped_bpb:.4f}, est_total_bytes={best.estimated_total_bytes}, legal={best.legal})"
    )


if __name__ == "__main__":
    main()
