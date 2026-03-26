#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.plan_sota_search import Experiment, parse_experiment_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_TABLE = ROOT / "research" / "experiment_table.md"
DEFAULT_SEEKER_JSON = ROOT / "research" / "sota_seeker.json"
DEFAULT_EXPORT_GAP_JSON = ROOT / "research" / "local_sp1024_10x560_joint_boundary_recovery_m4_export_gap.json"
DEFAULT_RESULTS_TSV = ROOT / "research" / "autoresearch_results.tsv"
DEFAULT_OUT_JSON = ROOT / "research" / "autoresearch_loop.json"
DEFAULT_OUT_MD = ROOT / "research" / "autoresearch_loop.md"
LOCAL_LAUNCHER = "bash scripts/local/train_seeker_sp1024_best_m4.sh"
MIN_MEANINGFUL_LOCAL_GAIN_BPB = 0.001


@dataclass(frozen=True)
class LoopItem:
    rank: int
    name: str
    track: str
    source: str
    hypothesis: str
    keep_if: str
    kill_if: str
    command: str
    rationale: list[str]


@dataclass(frozen=True)
class ResultEntry:
    name: str
    run_id: str
    tier: str
    status: str
    clean_bpb: float | None
    shipped_bpb: float | None
    export_gap_bpb: float | None
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the next Parameter Golf autoresearch loop from measured frontier data.")
    p.add_argument("--experiment-table", type=Path, default=DEFAULT_EXPERIMENT_TABLE)
    p.add_argument("--seeker-json", type=Path, default=DEFAULT_SEEKER_JSON)
    p.add_argument("--export-gap-json", type=Path, default=DEFAULT_EXPORT_GAP_JSON)
    p.add_argument("--results-tsv", type=Path, default=DEFAULT_RESULTS_TSV)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--mode", choices=("local-first", "official-first"), default="local-first")
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path) -> list[ResultEntry]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        results: list[ResultEntry] = []
        for row in rows:
            results.append(
                ResultEntry(
                    name=row["name"],
                    run_id=row.get("run_id", ""),
                    tier=row.get("tier", ""),
                    status=row.get("status", ""),
                    clean_bpb=float(row["clean_bpb"]) if row.get("clean_bpb") else None,
                    shipped_bpb=float(row["shipped_bpb"]) if row.get("shipped_bpb") else None,
                    export_gap_bpb=float(row["export_gap_bpb"]) if row.get("export_gap_bpb") else None,
                    note=row.get("note", ""),
                )
            )
    return results


def best_official_shipped(experiments: list[Experiment]) -> Experiment | None:
    official = [e for e in experiments if e.official and e.shipped_bpb is not None]
    return min(official, key=lambda e: float(e.shipped_bpb)) if official else None


def best_local_shipped(experiments: list[Experiment]) -> Experiment | None:
    local_markers = ("local", "mlx", "m4")
    local = [
        e
        for e in experiments
        if not e.official
        and e.shipped_bpb is not None
        and any(marker in f"{e.run} {e.config} {e.runtime} {e.conclusion}".lower() for marker in local_markers)
    ]
    return min(local, key=lambda e: float(e.shipped_bpb)) if local else None


def best_local_result(results: list[ResultEntry]) -> ResultEntry | None:
    local = [result for result in results if result.tier == "local" and result.shipped_bpb is not None]
    if not local:
        return None

    def score(result: ResultEntry) -> tuple[int, float]:
        status_rank = 0 if result.status == "kept" else 1
        return (status_rank, float(result.shipped_bpb))

    return min(local, key=score)


def previous_kept_local_result(anchor: ResultEntry, results: list[ResultEntry], candidate_by_name: dict[str, dict]) -> ResultEntry | None:
    anchor_family = infer_result_family_key(anchor, candidate_by_name)
    if anchor_family is None or anchor.shipped_bpb is None:
        return None
    candidates: list[ResultEntry] = []
    for result in results:
        if result.tier != "local" or result.status != "kept" or result.shipped_bpb is None:
            continue
        if result.run_id == anchor.run_id:
            continue
        if infer_result_family_key(result, candidate_by_name) != anchor_family:
            continue
        if float(result.shipped_bpb) <= float(anchor.shipped_bpb):
            continue
        candidates.append(result)
    if not candidates:
        return None
    return min(candidates, key=lambda result: float(result.shipped_bpb))


def build_export_first_command(best_export: dict, *, local: bool) -> str:
    fp16_patterns = ",".join(best_export["fp16_patterns"])
    parts = [
        "RUN_ID=autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_fp16",
        "QUANT_FORMAT=int8_clean_per_row_v1",
        "TARGET_EXPORT_NAME_PATTERNS=",
        "INT4_NAME_PATTERNS=",
        "TRAIN_QAT_NAME_PATTERNS=",
        "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=",
        f"INT8_KEEP_FLOAT_FP16_NAME_PATTERNS={fp16_patterns}",
    ]
    if local:
        parts.extend(
            [
                "ITERATIONS=300",
                "VAL_LOSS_EVERY=100",
                "TRAIN_LOG_EVERY=25",
                LOCAL_LAUNCHER,
            ]
        )
    else:
        parts.append("bash scripts/runpod/train_seeker_sp1024_best_8xh100.sh")
    return " ".join(parts)


def build_local_continuation_command(anchor: ResultEntry, best_export: dict) -> str:
    if "keptfp16_recovery" in anchor.run_id:
        return build_local_fp16_recovery_continuation_command(anchor, best_export)
    fp16_patterns = ",".join(best_export["fp16_patterns"])
    checkpoint = f"./logs/{anchor.run_id}_mlx_model.npz"
    next_run_id = f"{anchor.run_id}_continue120_microtail"
    parts = [
        f"RUN_ID={next_run_id}",
        f"INIT_MODEL_PATH={checkpoint}",
        "LR_SCHEDULE=constant",
        "WARMUP_STEPS=0",
        "LR_WARMUP_ITERS=0",
        "MIN_LR_SCALE=1.0",
        "TIED_EMBED_LR=0.0005",
        "MATRIX_LR=0.00035",
        "SCALAR_LR=0.00035",
        "ITERATIONS=120",
        "MAX_WALLCLOCK_SECONDS=0",
        "VAL_LOSS_EVERY=100",
        "TRAIN_LOG_EVERY=25",
        "QUANT_FORMAT=int8_clean_per_row_v1",
        "TARGET_EXPORT_NAME_PATTERNS=",
        "INT4_NAME_PATTERNS=",
        "TRAIN_QAT_NAME_PATTERNS=",
        "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=",
        f"INT8_KEEP_FLOAT_FP16_NAME_PATTERNS={fp16_patterns}",
        LOCAL_LAUNCHER,
    ]
    return " ".join(parts)


def build_local_fp16_recovery_command(source: ResultEntry, best_export: dict) -> str:
    fp16_patterns = ",".join(best_export["fp16_patterns"])
    checkpoint = f"./logs/{source.run_id}_mlx_model.npz"
    run_id = f"{source.run_id}_keptfp16_recovery120"
    parts = [
        f"RUN_ID={run_id}",
        f"INIT_MODEL_PATH={checkpoint}",
        "LR_SCHEDULE=constant",
        "WARMUP_STEPS=0",
        "LR_WARMUP_ITERS=0",
        "MIN_LR_SCALE=1.0",
        "TIED_EMBED_LR=0.0",
        "MATRIX_LR=0.0002",
        "SCALAR_LR=0.0",
        "ITERATIONS=120",
        "MAX_WALLCLOCK_SECONDS=0",
        "VAL_LOSS_EVERY=100",
        "TRAIN_LOG_EVERY=25",
        f"TRAIN_GRAD_ONLY_NAME_PATTERNS={fp16_patterns}",
        "TRAIN_GRAD_SKIP_NAME_PATTERNS=",
        "QUANT_FORMAT=int8_clean_per_row_v1",
        "TARGET_EXPORT_NAME_PATTERNS=",
        "INT4_NAME_PATTERNS=",
        "TRAIN_QAT_NAME_PATTERNS=",
        "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=",
        f"INT8_KEEP_FLOAT_FP16_NAME_PATTERNS={fp16_patterns}",
        LOCAL_LAUNCHER,
    ]
    return " ".join(parts)


def build_local_fp16_recovery_continuation_command(anchor: ResultEntry, best_export: dict) -> str:
    fp16_patterns = ",".join(best_export["fp16_patterns"])
    checkpoint = f"./logs/{anchor.run_id}_mlx_model.npz"
    run_id = f"{anchor.run_id}_continue80_keptfp16"
    parts = [
        f"RUN_ID={run_id}",
        f"INIT_MODEL_PATH={checkpoint}",
        "LR_SCHEDULE=constant",
        "WARMUP_STEPS=0",
        "LR_WARMUP_ITERS=0",
        "MIN_LR_SCALE=1.0",
        "TIED_EMBED_LR=0.0",
        "MATRIX_LR=0.00015",
        "SCALAR_LR=0.0",
        "ITERATIONS=80",
        "MAX_WALLCLOCK_SECONDS=0",
        "VAL_LOSS_EVERY=80",
        "TRAIN_LOG_EVERY=20",
        f"TRAIN_GRAD_ONLY_NAME_PATTERNS={fp16_patterns}",
        "TRAIN_GRAD_SKIP_NAME_PATTERNS=",
        "QUANT_FORMAT=int8_clean_per_row_v1",
        "TARGET_EXPORT_NAME_PATTERNS=",
        "INT4_NAME_PATTERNS=",
        "TRAIN_QAT_NAME_PATTERNS=",
        "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=",
        f"INT8_KEEP_FLOAT_FP16_NAME_PATTERNS={fp16_patterns}",
        LOCAL_LAUNCHER,
    ]
    return " ".join(parts)


def build_local_u5k_tokenflow_command(anchor: ResultEntry) -> str:
    checkpoint = f"./logs/{anchor.run_id}_mlx_model.npz"
    run_id = f"{anchor.run_id}_batch9216_continue40"
    parts = [
        f"RUN_ID={run_id}",
        f"INIT_MODEL_PATH={checkpoint}",
        "DATA_PATH=./data/local_u5k_unigram/datasets/fineweb10B_spu5120_local",
        "TOKENIZER_PATH=./data/local_u5k_unigram/tokenizers/fineweb_5120_unigram.model",
        "VOCAB_SIZE=5120",
        "TRAIN_SEQ_LEN=896",
        "TRAIN_BATCH_TOKENS=9216",
        "VAL_BATCH_SIZE=131072",
        "VAL_MAX_TOKENS=262144",
        "LR_SCHEDULE=constant",
        "WARMUP_STEPS=0",
        "LR_WARMUP_ITERS=0",
        "MIN_LR_SCALE=1.0",
        "TIED_EMBED_LR=0.0009",
        "MATRIX_LR=0.0006",
        "SCALAR_LR=0.0006",
        "ITERATIONS=40",
        "MAX_WALLCLOCK_SECONDS=0",
        "VAL_LOSS_EVERY=20",
        "TRAIN_LOG_EVERY=10",
        "LFQAT_FULL_STEP=5",
        "LFQAT_MIN_PROB=0.50",
        "LFQAT_KL_WEIGHT=0.015",
        "LFQAT_FISHER_WEIGHT=0.003",
        "TRAIN_COMPRESSION_AWARE_WEIGHT=0.0008",
        "bash scripts/local/train_lab_u5k_12x608_transplant_lowlr40_m4.sh",
    ]
    return " ".join(parts)


def parse_env_assignments(command: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for token in shlex.split(command.replace("\\\n", " ")):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.isupper():
            env[key] = value
    return env


def format_env_command(env: dict[str, str], launcher: str) -> str:
    return " ".join([*(f"{key}={value}" for key, value in env.items()), launcher])


def build_local_candidate_command(candidate: dict) -> str:
    env = parse_env_assignments(candidate["final_8xh100_command"])
    local_overrides = {
        "RUN_ID": candidate["name"].replace("seeker_", "local_"),
        "TRAIN_BATCH_TOKENS": "16384",
        "VAL_BATCH_SIZE": "32768",
        "VAL_MAX_TOKENS": "262144",
        "GRAD_ACCUM_STEPS": "2",
        "MLX_MAX_MICROBATCH_TOKENS": "8192",
        "ITERATIONS": "300",
        "MAX_WALLCLOCK_SECONDS": "0",
        "VAL_LOSS_EVERY": "100",
        "TRAIN_LOG_EVERY": "25",
    }
    keys = [
        "RUN_ID",
        "NUM_LAYERS",
        "NUM_UNIQUE_LAYERS",
        "MODEL_DIM",
        "NUM_HEADS",
        "NUM_KV_HEADS",
        "MLP_MULT",
        "TRAIN_SEQ_LEN",
        "QK_GAIN_INIT",
        "ROPE_BASE",
        "LOGIT_SOFTCAP",
        "LR_SCHEDULE",
        "WARMUP_STEPS",
        "WARMDOWN_ITERS",
        "LR_WARMUP_ITERS",
        "MIN_LR_SCALE",
        "TIED_EMBED_LR",
        "MATRIX_LR",
        "SCALAR_LR",
        "TIED_EMBED_INIT_STD",
        "MUON_MOMENTUM",
        "MUON_BACKEND_STEPS",
        "MUON_MOMENTUM_WARMUP_START",
        "MUON_MOMENTUM_WARMUP_STEPS",
        "BETA1",
        "BETA2",
        "ADAM_EPS",
        "QUANT_FORMAT",
        "TARGET_EXPORT_NAME_PATTERNS",
        "INT4_NAME_PATTERNS",
        "TRAIN_QAT_NAME_PATTERNS",
        "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS",
        "INT8_KEEP_FLOAT_FP16_NAME_PATTERNS",
        "INT4_BLOCK_SIZE",
        "INT4_CLIP_PERCENTILE",
        "TRAIN_QAT_BLOCK_SIZE",
        "LFQAT_START_STEP",
        "LFQAT_FULL_STEP",
        "LFQAT_MIN_PROB",
        "LFQAT_MAX_PROB",
        "LFQAT_KL_WEIGHT",
        "LFQAT_FISHER_WEIGHT",
        "LFQAT_TEMPERATURE",
        "TRAIN_COMPRESSION_AWARE_WEIGHT",
        "TRAIN_GRAD_ONLY_NAME_PATTERNS",
        "TRAIN_GRAD_SKIP_NAME_PATTERNS",
    ]
    parts: list[str] = []
    for key in keys:
        value = local_overrides.get(key, env.get(key))
        if value is None:
            continue
        parts.append(f"{key}={value}")
    for key in ("TRAIN_BATCH_TOKENS", "VAL_BATCH_SIZE", "VAL_MAX_TOKENS", "GRAD_ACCUM_STEPS", "MLX_MAX_MICROBATCH_TOKENS", "ITERATIONS", "MAX_WALLCLOCK_SECONDS", "VAL_LOSS_EVERY", "TRAIN_LOG_EVERY"):
        if key not in env:
            parts.append(f"{key}={local_overrides[key]}")
    parts.append(LOCAL_LAUNCHER)
    return " ".join(parts)


def build_local_export_only_command(candidate: dict, checkpoint: Path) -> str:
    env = parse_env_assignments(build_local_candidate_command(candidate))
    env["RUN_ID"] = f"{candidate['name'].replace('seeker_', 'local_').replace('_fp16', '')}_export_eval"
    env["INIT_MODEL_PATH"] = str(checkpoint)
    env["ITERATIONS"] = "0"
    env["WARMUP_STEPS"] = "0"
    env["VAL_LOSS_EVERY"] = "0"
    env["TRAIN_LOG_EVERY"] = "0"
    env["MAX_WALLCLOCK_SECONDS"] = "0"
    return format_env_command(env, LOCAL_LAUNCHER)


def unique_official_candidates(seeker: dict) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for key in ("best_official_ready_candidates", "top_candidates"):
        for candidate in seeker.get(key, []):
            name = candidate["name"]
            if not candidate.get("official_eligible") or not candidate.get("legal"):
                continue
            if name in seen:
                continue
            seen.add(name)
            unique.append(candidate)
    return unique


def candidate_family_key(candidate: dict) -> tuple[str, str]:
    return (candidate["model"], candidate["training"])


def result_checkpoint_path(result: ResultEntry) -> Path:
    return ROOT / "logs" / f"{result.run_id}_mlx_model.npz"


def infer_result_family_key(result: ResultEntry, candidate_by_name: dict[str, dict]) -> tuple[str, str] | None:
    matched = candidate_by_name.get(result.name)
    if matched is not None:
        return candidate_family_key(matched)

    text = " ".join(part for part in (result.name, result.run_id, result.note) if part).lower()
    model = None
    if "10x560" in text:
        model = "10x560"
    elif "10x544" in text:
        model = "10x544"
    elif "10x512" in text:
        model = "10x512"

    training = None
    if (
        "maxtrain" in text
        or "keptfp16_recovery" in text
        or "hybrid_int8aware" in text
        or "int8aware_block4proj" in text
    ):
        training = "stable_muon97_compiled_maxtrain"
    elif "longtail" in text or "export_first" in text:
        training = "stable_muon97_compiled_longtail"

    if model is None or training is None:
        return None
    return (model, training)


def best_family_checkpoint_for_candidate(candidate: dict, results: list[ResultEntry], candidate_by_name: dict[str, dict]) -> ResultEntry | None:
    family = candidate_family_key(candidate)
    family_results: list[ResultEntry] = []
    for result in results:
        result_family = infer_result_family_key(result, candidate_by_name)
        if result_family != family:
            continue
        if not result_checkpoint_path(result).exists():
            continue
        family_results.append(result)
    if not family_results:
        return None

    def score(result: ResultEntry) -> tuple[int, float, float]:
        export_only_penalty = 1 if "_export_eval_" in result.run_id else 0
        clean = result.clean_bpb if result.clean_bpb is not None else float("inf")
        shipped = result.shipped_bpb if result.shipped_bpb is not None else float("inf")
        return (export_only_penalty, clean, shipped)

    return min(family_results, key=score)


def find_gap_recovery_source(
    anchor: ResultEntry | None,
    results: list[ResultEntry],
    candidate_by_name: dict[str, dict],
) -> ResultEntry | None:
    if anchor is None or anchor.clean_bpb is None or anchor.shipped_bpb is None or anchor.export_gap_bpb is None:
        return None
    anchor_family = infer_result_family_key(anchor, candidate_by_name)
    if anchor_family is None:
        return None

    candidates: list[ResultEntry] = []
    for result in results:
        if result.tier != "local" or result.status != "discarded":
            continue
        if result.clean_bpb is None or result.shipped_bpb is None or result.export_gap_bpb is None:
            continue
        if infer_result_family_key(result, candidate_by_name) != anchor_family:
            continue
        if not result_checkpoint_path(result).exists():
            continue
        if result.clean_bpb >= anchor.clean_bpb:
            continue
        if result.shipped_bpb > anchor.shipped_bpb + 0.002:
            continue
        if result.export_gap_bpb < anchor.export_gap_bpb + 0.02:
            continue
        candidates.append(result)
    if not candidates:
        return None
    return min(candidates, key=lambda result: (float(result.shipped_bpb), float(result.clean_bpb)))


def build_loop(experiments: list[Experiment], seeker: dict, export_gap: dict, results: list[ResultEntry], *, mode: str) -> dict:
    best_official = best_official_shipped(experiments)
    best_local = best_local_shipped(experiments)
    best_local_result_anchor = best_local_result(results)
    best_official_bpb = float(best_official.shipped_bpb) if best_official and best_official.shipped_bpb is not None else None
    best_local_bpb = float(best_local.shipped_bpb) if best_local and best_local.shipped_bpb is not None else None
    local_anchor_bpb = (
        float(best_local_result_anchor.shipped_bpb)
        if best_local_result_anchor and best_local_result_anchor.shipped_bpb is not None
        else best_local_bpb
    )
    best_export = export_gap["best_legal"]
    candidates = unique_official_candidates(seeker)
    if not candidates:
        raise ValueError("No official-ready candidates found in seeker payload.")
    candidate_by_name = {candidate["name"]: candidate for candidate in candidates}
    gap_recovery_source = find_gap_recovery_source(best_local_result_anchor, results, candidate_by_name)
    previous_kept_anchor = (
        previous_kept_local_result(best_local_result_anchor, results, candidate_by_name)
        if best_local_result_anchor is not None
        else None
    )

    local_first = mode == "local-first"
    resolved_names = {
        token
        for result in results
        if result.status in {"kept", "discarded"}
        for token in (result.name, result.run_id)
        if token
    }
    items: list[LoopItem] = []
    if (
        local_first
        and gap_recovery_source is not None
        and best_local_result_anchor is not None
        and best_local_result_anchor.shipped_bpb is not None
        and best_local_result_anchor.clean_bpb is not None
    ):
        recovery_name = f"{gap_recovery_source.run_id}_keptfp16_recovery120"
        if recovery_name not in resolved_names:
            items.append(
                LoopItem(
                    rank=1,
                    name=recovery_name,
                    track="M4 local",
                    source="measured_gap_recovery",
                    hypothesis="The last raw continuation improved clean again but widened the export gap. Recover shipped quality by tuning only the tensors already kept in fp16 so the int8 path stays fixed while the preserved projections absorb the new raw headroom.",
                    keep_if=(
                        f"Keep if shipped val_bpb beats the current best measured local autoresearch anchor ({best_local_result_anchor.shipped_bpb:.4f}) "
                        "or if clean improves by >= 0.02 while export gap stays <= the anchor."
                    ),
                    kill_if=(
                        f"Kill if the `step 100` clean val_bpb does not beat the source checkpoint ({gap_recovery_source.clean_bpb:.4f}) "
                        f"or if final shipped stays >= {best_local_result_anchor.shipped_bpb:.4f}."
                    ),
                    command=build_local_fp16_recovery_command(gap_recovery_source, best_export),
                    rationale=[
                        f"Current kept anchor `{best_local_result_anchor.run_id}` is best on shipped at {best_local_result_anchor.shipped_bpb:.4f}, but the newer discarded checkpoint `{gap_recovery_source.run_id}` is cleaner at {gap_recovery_source.clean_bpb:.4f}.",
                        f"The discarded checkpoint only failed because its export gap widened from {best_local_result_anchor.export_gap_bpb:.4f} to {gap_recovery_source.export_gap_bpb:.4f}.",
                        "Training only the already-kept fp16 upper projections is the cheapest new hypothesis that can convert that raw gain into shipped gain without moving the int8 tensors further off-grid.",
                    ],
                )
            )
    if (
        local_first
        and best_local_result_anchor is not None
        and best_local_result_anchor.shipped_bpb is not None
        and "u5k_12x608" in f"{best_local_result_anchor.name} {best_local_result_anchor.run_id}"
        and result_checkpoint_path(best_local_result_anchor).exists()
    ):
        continuation_name = f"{best_local_result_anchor.run_id}_batch9216_continue40"
        continuation_gain = None
        if (
            previous_kept_anchor is not None
            and previous_kept_anchor.shipped_bpb is not None
            and best_local_result_anchor.shipped_bpb is not None
        ):
            continuation_gain = float(previous_kept_anchor.shipped_bpb) - float(best_local_result_anchor.shipped_bpb)
        if continuation_name not in resolved_names:
            items.append(
                LoopItem(
                    rank=len(items) + 1,
                    name=continuation_name,
                    track="M4 local",
                    source="measured_u5k_tokenflow_continuation",
                    hypothesis="The revived `u5k` branch has already produced one small `seq_len=896` keep. A second bounded token-flow step that only increases train tokens per update may convert the same alive branch into a larger real gain without reopening raw tokenizer search.",
                    keep_if=(
                        f"Keep if shipped val_bpb beats the current best measured local autoresearch anchor ({best_local_result_anchor.shipped_bpb:.4f}) "
                        "or if clean improves by >= 0.01 with no shipping regression."
                    ),
                    kill_if=(
                        "Kill if the first `step 20` validation is flat to worse than the current anchor or if final shipped does not improve."
                    ),
                    command=build_local_u5k_tokenflow_command(best_local_result_anchor),
                    rationale=[
                        f"Current alive tokenizer/token-flow anchor: {best_local_result_anchor.run_id} at shipped {best_local_result_anchor.shipped_bpb:.4f}.",
                        *(
                            [f"Latest measured gain over the previous kept `u5k` result: {continuation_gain:.6f} bpb."]
                            if continuation_gain is not None
                            else []
                        ),
                        "The first sideways tokenizer probes (`u6k`, `b4k`) both lost, so the next low-entropy lever stays on the alive `u5k` branch.",
                    ],
                )
            )
    if (
        local_first
        and best_local_result_anchor is not None
        and best_local_result_anchor.shipped_bpb is not None
        and "10x560" in f"{best_local_result_anchor.name} {best_local_result_anchor.run_id}"
        and result_checkpoint_path(best_local_result_anchor).exists()
    ):
        recovery_anchor = "keptfp16_recovery" in best_local_result_anchor.run_id
        continuation_name = (
            f"{best_local_result_anchor.run_id}_continue80_keptfp16"
            if recovery_anchor
            else f"{best_local_result_anchor.run_id}_continue120_microtail"
        )
        continuation_gain = None
        if (
            previous_kept_anchor is not None
            and previous_kept_anchor.shipped_bpb is not None
            and best_local_result_anchor.shipped_bpb is not None
        ):
            continuation_gain = float(previous_kept_anchor.shipped_bpb) - float(best_local_result_anchor.shipped_bpb)
        continuation_is_meaningful = (
            continuation_gain is None or continuation_gain >= MIN_MEANINGFUL_LOCAL_GAIN_BPB
        )
        if continuation_name not in resolved_names and gap_recovery_source is None and continuation_is_meaningful:
            items.append(
                LoopItem(
                    rank=len(items) + 1,
                    name=continuation_name,
                    track="M4 local",
                    source="measured_gap_recovery_continuation" if recovery_anchor else "measured_local_continuation",
                    hypothesis=(
                        "The first fp16-kept recovery tail improved shipped bpb on the stronger raw checkpoint; one shorter follow-up on the same restricted parameter set may keep converting clean headroom into shipped gain without disturbing the int8 path."
                        if recovery_anchor
                        else "The current 10x560 local winner has improved through multiple continuation tails already; one shorter micro-tail at even lower LR may buy another shipped gain before the line truly saturates."
                    ),
                    keep_if=(
                        f"Keep if shipped val_bpb beats the current best measured local autoresearch anchor ({best_local_result_anchor.shipped_bpb:.4f}) "
                        + ("or if clean improves by >= 0.01 while export gap stays <= the current anchor." if recovery_anchor else "or if clean improves by >= 0.03 while export gap stays <= 0.11.")
                    ),
                    kill_if=(
                        f"Kill if the `step 100` clean val_bpb does not beat the current anchor ({best_local_result_anchor.clean_bpb:.4f}) "
                        + ("or if final shipped regresses or export gap widens further." if recovery_anchor else "or if final shipped regresses.")
                        if best_local_result_anchor.clean_bpb is not None
                        else "Kill if early clean validation does not improve or if final shipped regresses."
                    ),
                    command=build_local_continuation_command(best_local_result_anchor, best_export),
                    rationale=[
                        (
                            "The first fp16-kept recovery tail already improved shipped bpb on the cleaner checkpoint, so the most conservative next move is to continue that same restricted update set."
                            if recovery_anchor
                            else "Three consecutive continuation tails on the same 10x560 line have all improved shipped bpb."
                        ),
                        f"Current local autoresearch anchor: {best_local_result_anchor.run_id} at shipped {best_local_result_anchor.shipped_bpb:.4f}.",
                        *(
                            [f"Latest incremental shipped gain over the previous kept result: {continuation_gain:.6f} bpb."]
                            if continuation_gain is not None
                            else []
                        ),
                        f"Best measured export on that checkpoint is still {best_export['name']} with shipped {best_export['shipped_val_bpb']:.4f}.",
                    ],
                )
            )

    if "sp1024_10x560_export_first" not in resolved_names:
        items.append(
            LoopItem(
                rank=len(items) + 1,
                name="sp1024_10x560_export_first",
                track="M4 local" if local_first else "8xH100 official",
                source="measured_fixed_settings_export_gap",
                hypothesis="The current 10x560 path is likely held back more by shipping loss than by raw training. Re-run it with the measured low-gap export instead of the aggressive top-layer int4 default.",
                keep_if=(
                    f"Keep if local shipped val_bpb beats the current best measured local anchor ({local_anchor_bpb:.4f}) "
                    f"or if the export gap stays <= 0.10 while clean improves."
                    if local_first and local_anchor_bpb is not None
                    else (
                        f"Keep if shipped val_bpb beats the current best measured official anchor ({best_official_bpb:.4f}) by at least 0.01, or if clean improves while shipped gap is <= 0.10."
                        if best_official_bpb is not None
                        else "Keep if shipped val_bpb improves the best measured official anchor."
                    )
                ),
                kill_if="Kill if early validation is clearly behind the current best anchor or if the run crashes in the same way twice.",
                command=build_export_first_command(best_export, local=local_first),
                rationale=[
                    "Measured fixed-settings export search on the same 10x560 checkpoint found a much better legal policy.",
                    f"Best export policy: {best_export['name']} with shipped val_bpb {best_export['shipped_val_bpb']:.4f} and export gap {best_export['export_gap_bpb']:.4f}.",
                    "This is the fastest path to turn one measured weakness into a directly testable rerun.",
                ],
            )
        )

    candidate_rank = len(items) + 1
    for candidate in candidates:
        if candidate["name"] in resolved_names:
            continue
        shipped = candidate["estimated_shipped_bpb"]
        keep_if = (
            f"Keep if local shipped val_bpb beats the current best measured local anchor ({local_anchor_bpb:.4f}) "
            f"or if clean/local export-gap direction is clearly better than the 10x560 local rerun."
            if local_first and local_anchor_bpb is not None
            else (
                f"Keep if shipped val_bpb beats the current best measured official anchor ({best_official_bpb:.4f}) "
                f"or lands <= {min(best_official_bpb - 0.01, shipped):.4f}."
                if best_official_bpb is not None
                else "Keep if shipped val_bpb improves the current official anchor."
            )
        )
        local_command = build_local_candidate_command(candidate)
        rationale = [
            f"Estimated shipped val_bpb: {candidate['estimated_shipped_bpb']:.4f}.",
            f"Estimated total bytes: {candidate['estimated_total_bytes']}.",
            f"Runtime ratio: {candidate['estimated_runtime_ratio']:.3f}.",
        ] + candidate["rationale"][:3]
        hypothesis = (
            f"Test the seeker-ranked official-ready candidate `{candidate['model']}` with "
            f"`{candidate['training']}` and `{candidate['export_policy']}` on the real budget."
        )
        if local_first:
            checkpoint_result = best_family_checkpoint_for_candidate(candidate, results, candidate_by_name)
            if checkpoint_result is not None:
                if (
                    local_anchor_bpb is not None
                    and checkpoint_result.shipped_bpb is not None
                    and checkpoint_result.shipped_bpb > local_anchor_bpb + 0.05
                ):
                    continue
                checkpoint = result_checkpoint_path(checkpoint_result)
                local_command = build_local_export_only_command(candidate, checkpoint)
                rationale = [
                    f"Reuse measured raw checkpoint from `{checkpoint_result.run_id}` instead of retraining the same `{candidate['model']} + {candidate['training']}` family.",
                    "Only the export policy changes, so a local `ITERATIONS=0` roundtrip eval is the higher-EV experiment.",
                ] + rationale
                hypothesis = (
                    f"Measure whether `{candidate['export_policy']}` beats the already-trained `{candidate['model']} / {candidate['training']}` raw checkpoint without paying for another full local training run."
                )
        items.append(
            LoopItem(
                rank=candidate_rank,
                name=candidate["name"],
                track="M4 local" if local_first else "8xH100 official",
                source="offline_seeker",
                hypothesis=hypothesis,
                keep_if=keep_if,
                kill_if="Kill if early validation is materially behind the current local anchor with no sign of catching up." if local_first else "Kill if early validation is materially behind the measured 10x512 anchor with no sign of catching up.",
                command=local_command if local_first else candidate["final_8xh100_command"],
                rationale=rationale,
            )
        )
        candidate_rank += 1
        if len(items) >= 3:
            break

    return {
        "mode": mode,
        "best_measured_official": asdict(best_official) if best_official else None,
        "best_measured_local": asdict(best_local) if best_local else None,
        "best_measured_local_result": asdict(best_local_result_anchor) if best_local_result_anchor else None,
        "best_measured_export_gap": best_export,
        "resolved_results": [asdict(result) for result in results],
        "loop": [asdict(item) for item in items],
    }


def write_markdown(path: Path, payload: dict) -> None:
    best_official = payload["best_measured_official"]
    best_local = payload["best_measured_local"]
    best_local_result_anchor = payload.get("best_measured_local_result")
    best_export = payload["best_measured_export_gap"]
    lines = [
        "# Autoresearch Loop",
        "",
        f"- Mode: `{payload['mode']}`",
        "",
        "## Anchors",
        "",
    ]
    if best_official:
        lines.append(
            f"- Best measured official shipped anchor: `{best_official['run']}` at `{best_official['shipped_bpb']:.4f}` bpb."
        )
    if best_local:
        lines.append(
            f"- Best measured local shipped anchor: `{best_local['run']}` at `{best_local['shipped_bpb']:.4f}` bpb."
        )
    if best_local_result_anchor:
        lines.append(
            f"- Best measured local autoresearch anchor: `{best_local_result_anchor['run_id']}` at `{best_local_result_anchor['shipped_bpb']:.4f}` bpb."
        )
    lines.extend(
        [
            f"- Best fixed-settings export policy for the current `10x560` local checkpoint: `{best_export['name']}` "
            f"with shipped `{best_export['shipped_val_bpb']:.4f}` and gap `{best_export['export_gap_bpb']:.4f}`.",
            "",
        ]
    )
    if payload["resolved_results"]:
        lines.extend(["## Resolved Results", ""])
        for result in payload["resolved_results"]:
            shipped = f"{result['shipped_bpb']:.4f}" if result["shipped_bpb"] is not None else "-"
            clean = f"{result['clean_bpb']:.4f}" if result["clean_bpb"] is not None else "-"
            gap = f"{result['export_gap_bpb']:.4f}" if result["export_gap_bpb"] is not None else "-"
            lines.append(
                f"- `{result['name']}`: `{result['status']}` clean `{clean}` shipped `{shipped}` gap `{gap}`"
                + (f" ({result['note']})" if result["note"] else "")
            )
        lines.extend(
            [
                "",
            ]
        )
    lines.extend(
        [
            "## Next Experiments",
            "",
        ]
    )
    if not payload["loop"]:
        lines.extend(
            [
                "No unresolved high-EV local experiments remain in the current loop.",
                "",
                "The next move is to expand the search space or add a new measured family, not to rerun stale export-only siblings.",
                "",
            ]
        )
    for item in payload["loop"]:
        lines.extend(
            [
                f"### {item['rank']}. `{item['name']}`",
                "",
                f"- Track: `{item['track']}`",
                f"- Source: `{item['source']}`",
                f"- Hypothesis: {item['hypothesis']}",
                f"- Keep if: {item['keep_if']}",
                f"- Kill if: {item['kill_if']}",
                "- Why:",
            ]
        )
        for reason in item["rationale"]:
            lines.append(f"  - {reason}")
        lines.extend(
            [
                "- Command:",
                "```bash",
                item["command"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiments = parse_experiment_table(args.experiment_table)
    seeker = load_json(args.seeker_json)
    export_gap = load_json(args.export_gap_json)
    results = load_results(args.results_tsv)
    payload = build_loop(experiments, seeker, export_gap, results, mode=args.mode)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.out_md, payload)
    print(args.out_json)
    print(args.out_md)


if __name__ == "__main__":
    main()
