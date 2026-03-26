#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.plan_sota_search import (
    COUNTED_CODE_PATH,
    build_fp16_patterns,
    build_group_patterns,
    parse_experiment_table,
)
from research.search_sota_trials import TrialCandidate, search_trials


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "research" / "sota_seeker.json"
DEFAULT_OUT_MD = ROOT / "research" / "sota_seeker.md"
SINGLE_GPU_PROBE_TOKEN_CAP = 524_288
LOCAL_EXPORT_GAP_JSON = ROOT / "research" / "local_sp1024_10x560_block4attn_m4_export_gap.json"


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    family: str
    vocab_size: int
    data_path: str
    tokenizer_path: str
    launcher: str
    clean_delta: float
    gap_delta: float
    runtime_multiplier: float
    notes: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tokenizer_family: str
    layers: int
    num_unique_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: int
    init_mode: str
    clean_delta: float
    gap_delta: float
    byte_delta: int
    runtime_multiplier: float
    evidence_weight: float
    notes: str


@dataclass(frozen=True)
class TokenFlowSpec:
    name: str
    train_seq_len: int
    train_batch_tokens: int
    val_batch_size: int
    grad_clip_norm: float
    grad_accum_steps: int
    qk_gain_init: float
    rope_base: float
    logit_softcap: float
    clean_delta: float
    gap_delta: float
    runtime_multiplier: float
    notes: str


@dataclass(frozen=True)
class TrainingSpec:
    name: str
    lr_schedule: str
    warmup_steps: int
    warmdown_iters: int
    lr_warmup_iters: int
    min_lr_scale: float
    tied_embed_lr: float
    matrix_lr: float
    scalar_lr: float
    tied_embed_init_std: float
    muon_momentum: float
    muon_backend_steps: int
    muon_momentum_warmup_start: float
    muon_momentum_warmup_steps: int
    beta1: float
    beta2: float
    adam_eps: float
    lfqat_start_step: int
    lfqat_full_step: int
    lfqat_min_prob: float
    lfqat_max_prob: float
    lfqat_kl_weight: float
    lfqat_fisher_weight: float
    lfqat_temperature: float
    train_compression_aware_weight: float
    train_grad_only_name_patterns: str
    train_grad_skip_name_patterns: str
    clean_delta: float
    gap_delta: float
    runtime_multiplier: float
    notes: str
    iterations: int = 2500
    val_loss_every: int = 100
    train_log_every: int = 25


@dataclass(frozen=True)
class ExportSpec:
    name: str
    quant_format: str
    int4_groups: tuple[str, ...]
    fp16_groups: tuple[str, ...]
    fp32_name_patterns: str
    int4_block_size: int
    int4_clip_percentile: float
    train_qat_block_size: int
    gap_recovery: float
    optimistic_gap_floor: float
    byte_delta: int
    byte_risk: float
    notes: str


@dataclass
class SeekerCandidate:
    rank: int
    name: str
    source: str
    tokenizer: str
    model: str
    token_flow: str
    training: str
    export_policy: str
    layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: int
    init_mode: str
    estimated_clean_bpb: float
    estimated_shipped_bpb: float
    estimated_gap: float
    best_case_shipped_bpb: float
    clean_gain_needed_after_best_export: float
    estimated_total_bytes: int
    estimated_runtime_ratio: float
    official_eligible: bool
    launch_tier: str
    legal: bool
    target_band: str
    objective: float
    confidence: float
    rationale: list[str]
    screen_command: str
    final_6xh100_command: str
    final_8xh100_command: str


@dataclass(frozen=True)
class SpCalibrationAnchor:
    run_id: str
    model_name: str
    flow_name: str
    training_name: str
    export_name: str
    raw_clean: float
    raw_gap: float
    raw_total_bytes: int
    raw_runtime_ratio: float
    actual_clean: float
    actual_gap: float
    actual_total_bytes: int
    actual_runtime_ratio: float


@dataclass(frozen=True)
class ExportGapCalibration:
    name: str
    num_layers: int
    model_dim: int
    num_kv_heads: int
    train_seq_len: int
    measured_gap: float
    measured_total_bytes: int


TOKENIZERS = (
    TokenizerSpec(
        name="sp1024_bpe",
        family="sp1024",
        vocab_size=1024,
        data_path="./data/datasets/fineweb10B_sp1024",
        tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
        launcher="scripts/runpod/train_baseline_sp1024.sh",
        clean_delta=0.0,
        gap_delta=0.0,
        runtime_multiplier=1.0,
        notes="Closest official anchor to the public leaderboard.",
    ),
    TokenizerSpec(
        name="u4k_unigram",
        family="u4k",
        vocab_size=4096,
        data_path="./data/official_u4k_unigram/datasets/fineweb10B_spu4096_docs",
        tokenizer_path="./data/official_u4k_unigram/tokenizers/fineweb_4096_unigram.model",
        launcher="scripts/runpod/train_dense_u4k_12x608_lfqat_continue80m.sh",
        clean_delta=0.0,
        gap_delta=0.0,
        runtime_multiplier=1.0,
        notes="Useful for research recovery, but much weaker than the sp1024 public anchor today.",
    ),
)

SP1024_MODELS = (
    ModelSpec("sp_9x512_kv4", "sp1024", 9, 9, 512, 8, 4, 2, "scratch", 0.0, 0.0, 0, 1.0, 1.0, "Public baseline anchor shape."),
    ModelSpec("sp_10x512_kv2", "sp1024", 10, 10, 512, 8, 2, 2, "scratch", -0.018, -0.004, -450_000, 1.01, 0.86, "Use GQA headroom for one extra layer without pushing bytes over cap."),
    ModelSpec("sp_10x528_kv2", "sp1024", 10, 10, 528, 8, 2, 2, "scratch", -0.024, -0.005, -300_000, 1.03, 0.82, "Small width bump over the strongest measured 10x512 path without fully committing to 544."),
    ModelSpec("sp_10x592_kv2", "sp1024", 10, 10, 592, 8, 2, 2, "scratch", -0.043, -0.007, 760_000, 1.12, 0.56, "Near-cap width extension beyond 576 for hybrid-export stretch paths only."),
    ModelSpec("sp_11x512_kv2", "sp1024", 11, 11, 512, 8, 2, 2, "scratch", -0.028, -0.005, 150_000, 1.05, 0.72, "Deeper GQA variant near the budget edge."),
    ModelSpec("sp_10x544_kv2", "sp1024", 10, 10, 544, 8, 2, 2, "scratch", -0.032, -0.006, -150_000, 1.06, 0.78, "Most balanced width/depth stretch inside a plausible legal envelope."),
    ModelSpec("sp_11x560_kv2", "sp1024", 11, 11, 560, 8, 2, 2, "scratch", -0.041, -0.006, 540_000, 1.11, 0.58, "Near-cap depth-width compromise that stays in the validated even head-dim regime."),
    ModelSpec("sp_11x528_kv2", "sp1024", 11, 11, 528, 8, 2, 2, "scratch", -0.035, -0.005, 60_000, 1.07, 0.70, "Depth-width compromise just above the proven 10x512 line."),
    ModelSpec("sp_10x560_kv2", "sp1024", 10, 10, 560, 8, 2, 2, "scratch", -0.037, -0.006, 40_000, 1.08, 0.72, "Width bump above 544 while still plausibly legal with selective export."), 
    ModelSpec("sp_10x568_kv2", "sp1024", 10, 10, 568, 8, 2, 2, "scratch", -0.039, -0.006, 220_000, 1.09, 0.67, "Small additional width stretch that still fits the proven 10-layer training regime."),
    ModelSpec("sp_10x576_kv2", "sp1024", 10, 10, 576, 8, 2, 2, "scratch", -0.040, -0.007, 500_000, 1.10, 0.64, "Highest-upside near-cap scratch variant with moderate runtime pressure."),
    ModelSpec("sp_12x512_kv2", "sp1024", 12, 12, 512, 8, 2, 2, "scratch", -0.034, -0.004, 650_000, 1.08, 0.60, "Depth-heavy alternative if width fails to convert."),
    ModelSpec("sp_12x528_kv2", "sp1024", 12, 12, 528, 8, 2, 2, "scratch", -0.043, -0.005, 820_000, 1.12, 0.52, "Stretch depth-width candidate that may become legal only with stronger hybrid export policies."),
    ModelSpec("sp_9x544_kv2", "sp1024", 9, 9, 544, 8, 2, 2, "scratch", -0.020, -0.006, -650_000, 1.02, 0.74, "Width-biased GQA variant with modest runtime overhead."),
    ModelSpec("sp_10x512_kv1", "sp1024", 10, 10, 512, 8, 1, 2, "scratch", -0.012, -0.003, -700_000, 0.99, 0.55, "MQA-style stretch with more byte headroom but lower confidence."),
    ModelSpec("sp_10x544_kv1", "sp1024", 10, 10, 544, 8, 1, 2, "scratch", -0.025, -0.004, -400_000, 1.03, 0.48, "Width plus stronger head sharing."),
    ModelSpec("sp_10x576_kv1", "sp1024", 10, 10, 576, 8, 1, 2, "scratch", -0.031, -0.004, 150_000, 1.05, 0.42, "Near-cap width stretch with MQA."),
    ModelSpec("sp_11x544_kv2", "sp1024", 11, 11, 544, 8, 2, 2, "scratch", -0.038, -0.006, 450_000, 1.09, 0.58, "Deeper width-balanced variant."),
    ModelSpec("sp_12x544_kv2", "sp1024", 12, 12, 544, 8, 2, 2, "scratch", -0.046, -0.006, 1_050_000, 1.13, 0.46, "High-upside but runtime-stretch dense variant."),
    ModelSpec("sp_10x576_kv2_h16", "sp1024", 10, 10, 576, 16, 2, 2, "scratch", -0.036, -0.006, 520_000, 1.12, 0.38, "Higher head count variant for finer attention granularity."),
    ModelSpec("sp_10x544_kv2_share5", "sp1024", 10, 5, 544, 8, 2, 2, "scratch", -0.010, 0.008, -1_150_000, 0.97, 0.24, "Partial block sharing to buy extra width inside the cap."),
    ModelSpec("sp_12x544_kv2_share6", "sp1024", 12, 6, 544, 8, 2, 2, "scratch", -0.015, 0.012, -900_000, 1.01, 0.18, "Aggressive shared-depth moonshot."),
)

TOKEN_FLOWS = (
    TokenFlowSpec("official_base", 1024, 524_288, 524_288, 0.0, 1, 1.5, 10_000.0, 30.0, 0.0, 0.0, 1.0, "Control setup matching the public baseline regime."),
    TokenFlowSpec("clip_base", 1024, 524_288, 524_288, 0.7, 1, 1.5, 10_000.0, 28.0, -0.008, -0.003, 1.01, "Adds clipping and mild stability without changing token geometry."),
    TokenFlowSpec("context_1280", 1280, 491_520, 491_520, 0.6, 1, 1.6, 20_000.0, 28.0, -0.010, -0.004, 1.03, "Small context increase while keeping batch geometry efficient."),
    TokenFlowSpec("context_1536", 1536, 393_216, 393_216, 0.7, 1, 1.7, 50_000.0, 26.0, -0.014, -0.005, 1.06, "More context per update at a moderate runtime cost."),
    TokenFlowSpec("throughput_640k", 1024, 655_360, 655_360, 0.7, 1, 1.5, 10_000.0, 30.0, -0.012, -0.002, 1.07, "Aggressive token throughput for smoother optimization under fixed wallclock."),
    TokenFlowSpec("context_896_dense", 896, 573_440, 573_440, 0.6, 1, 1.45, 10_000.0, 30.0, -0.006, -0.001, 0.98, "Slightly shorter context to push more updates through the wallclock cap."),
    TokenFlowSpec("throughput_816k_896ctx", 896, 817_152, 817_152, 0.88, 1, 1.45, 10_000.0, 28.8, -0.019, 0.0, 1.13, "Intermediate 896-context throughput point between the proven 802k and denser 860k regimes."),
    TokenFlowSpec("throughput_896k_896ctx", 896, 802_816, 802_816, 0.85, 1, 1.45, 10_000.0, 29.0, -0.018, -0.001, 1.12, "Push significantly more tokens through the proven 896-context regime while staying on the same attention geometry."),
    TokenFlowSpec("throughput_832k_896ctx", 896, 860_160, 860_160, 0.90, 1, 1.45, 10_000.0, 28.5, -0.021, 0.0, 1.15, "Slightly more aggressive 896-context throughput push for candidates that still fit the wallclock budget."),
    TokenFlowSpec("context_960_balanced", 960, 768_000, 768_000, 0.75, 1, 1.48, 12_000.0, 29.0, -0.015, -0.001, 1.06, "Small context bump above the proven 896 path without jumping to the clearly weak 1536 regime."),
    TokenFlowSpec("context_1280_clip2", 1280, 524_288, 524_288, 0.9, 1, 1.7, 40_000.0, 26.0, -0.012, -0.004, 1.05, "Longer context with stronger clipping and adjusted RoPE."),
    TokenFlowSpec("throughput_720k_896ctx", 896, 720_896, 720_896, 0.8, 1, 1.45, 10_000.0, 30.0, -0.014, -0.001, 1.08, "Token-throughput-biased schedule for compact shapes."),
)

TRAINING_SPECS = (
    TrainingSpec(
        name="baseline_publicish",
        lr_schedule="cosine",
        warmup_steps=20,
        warmdown_iters=1200,
        lr_warmup_iters=20,
        min_lr_scale=0.2,
        tied_embed_lr=0.0020,
        matrix_lr=0.0015,
        scalar_lr=0.0015,
        tied_embed_init_std=0.005,
        muon_momentum=0.95,
        muon_backend_steps=5,
        muon_momentum_warmup_start=0.85,
        muon_momentum_warmup_steps=500,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=0,
        lfqat_min_prob=1.0,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.0,
        lfqat_fisher_weight=0.0,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=0.0,
        gap_delta=0.0,
        runtime_multiplier=1.0,
        notes="Baseline-like control recipe.",
    ),
    TrainingSpec(
        name="export_aware_lfqat",
        lr_schedule="cosine",
        warmup_steps=20,
        warmdown_iters=1200,
        lr_warmup_iters=20,
        min_lr_scale=0.15,
        tied_embed_lr=0.0022,
        matrix_lr=0.0016,
        scalar_lr=0.0016,
        tied_embed_init_std=0.005,
        muon_momentum=0.95,
        muon_backend_steps=5,
        muon_momentum_warmup_start=0.85,
        muon_momentum_warmup_steps=500,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=40,
        lfqat_min_prob=0.2,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.05,
        lfqat_fisher_weight=0.01,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0015,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.018,
        gap_delta=-0.015,
        runtime_multiplier=1.03,
        notes="Best current export-aware training bet.",
    ),
    TrainingSpec(
        name="int8_recovery",
        lr_schedule="cosine",
        warmup_steps=20,
        warmdown_iters=1200,
        lr_warmup_iters=20,
        min_lr_scale=0.2,
        tied_embed_lr=0.0018,
        matrix_lr=0.0012,
        scalar_lr=0.0012,
        tied_embed_init_std=0.005,
        muon_momentum=0.95,
        muon_backend_steps=5,
        muon_momentum_warmup_start=0.85,
        muon_momentum_warmup_steps=500,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=20,
        lfqat_min_prob=0.1,
        lfqat_max_prob=0.7,
        lfqat_kl_weight=0.02,
        lfqat_fisher_weight=0.005,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0005,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.010,
        gap_delta=-0.022,
        runtime_multiplier=1.0,
        notes="Bias toward recovering int8 shipping quality.",
    ),
    TrainingSpec(
        name="depth_push",
        lr_schedule="cosine",
        warmup_steps=30,
        warmdown_iters=1000,
        lr_warmup_iters=30,
        min_lr_scale=0.12,
        tied_embed_lr=0.0024,
        matrix_lr=0.0018,
        scalar_lr=0.0018,
        tied_embed_init_std=0.0045,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.88,
        muon_momentum_warmup_steps=700,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=10,
        lfqat_full_step=80,
        lfqat_min_prob=0.15,
        lfqat_max_prob=0.95,
        lfqat_kl_weight=0.04,
        lfqat_fisher_weight=0.008,
        lfqat_temperature=2.2,
        train_compression_aware_weight=0.0010,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.024,
        gap_delta=-0.010,
        runtime_multiplier=1.06,
        notes="Higher-upside scratch recipe for deeper/wider dense variants.",
    ),
    TrainingSpec(
        name="depth_push_targeted",
        lr_schedule="cosine",
        warmup_steps=30,
        warmdown_iters=900,
        lr_warmup_iters=30,
        min_lr_scale=0.10,
        tied_embed_lr=0.0026,
        matrix_lr=0.0019,
        scalar_lr=0.0019,
        tied_embed_init_std=0.0040,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.88,
        muon_momentum_warmup_steps=700,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=10,
        lfqat_full_step=90,
        lfqat_min_prob=0.15,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.05,
        lfqat_fisher_weight=0.010,
        lfqat_temperature=2.2,
        train_compression_aware_weight=0.0012,
        train_grad_only_name_patterns="blocks.5,blocks.6,blocks.7,blocks.8,blocks.9,blocks.10,blocks.11",
        train_grad_skip_name_patterns="",
        clean_delta=-0.028,
        gap_delta=-0.014,
        runtime_multiplier=1.05,
        notes="Focus updates on upper blocks for export-aware larger-shape recovery.",
    ),
    TrainingSpec(
        name="stable_muon97",
        lr_schedule="cosine",
        warmup_steps=20,
        warmdown_iters=1400,
        lr_warmup_iters=24,
        min_lr_scale=0.22,
        tied_embed_lr=0.0019,
        matrix_lr=0.0013,
        scalar_lr=0.0013,
        tied_embed_init_std=0.0055,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.90,
        muon_momentum_warmup_steps=900,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=30,
        lfqat_min_prob=0.1,
        lfqat_max_prob=0.8,
        lfqat_kl_weight=0.025,
        lfqat_fisher_weight=0.006,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0008,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.014,
        gap_delta=-0.008,
        runtime_multiplier=1.01,
        notes="Leans on a smoother Muon schedule and longer warmdown.",
    ),
    TrainingSpec(
        name="stable_muon97_compiled",
        lr_schedule="cosine",
        warmup_steps=20,
        warmdown_iters=1400,
        lr_warmup_iters=24,
        min_lr_scale=0.22,
        tied_embed_lr=0.0019,
        matrix_lr=0.0013,
        scalar_lr=0.0013,
        tied_embed_init_std=0.0055,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.90,
        muon_momentum_warmup_steps=900,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=0,
        lfqat_min_prob=1.0,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.0,
        lfqat_fisher_weight=0.0,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.022,
        gap_delta=0.002,
        runtime_multiplier=0.91,
        notes="Compile-friendly version of the stable Muon recipe with LFQAT/QAT fully disabled for pure int8 shipping paths.",
    ),
    TrainingSpec(
        name="stable_muon97_fullbudget",
        lr_schedule="cosine",
        warmup_steps=20,
        warmdown_iters=1900,
        lr_warmup_iters=24,
        min_lr_scale=0.22,
        tied_embed_lr=0.0019,
        matrix_lr=0.0013,
        scalar_lr=0.0013,
        tied_embed_init_std=0.0055,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.90,
        muon_momentum_warmup_steps=900,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=40,
        lfqat_min_prob=0.1,
        lfqat_max_prob=0.85,
        lfqat_kl_weight=0.028,
        lfqat_fisher_weight=0.007,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0009,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.026,
        gap_delta=-0.010,
        runtime_multiplier=1.42,
        notes="Same stable recipe, but sized to actually spend the full 10-minute budget.",
        iterations=3500,
        val_loss_every=200,
        train_log_every=50,
    ),
    TrainingSpec(
        name="stable_muon97_compiled_fullbudget",
        lr_schedule="cosine",
        warmup_steps=20,
        warmdown_iters=1900,
        lr_warmup_iters=24,
        min_lr_scale=0.22,
        tied_embed_lr=0.0019,
        matrix_lr=0.0013,
        scalar_lr=0.0013,
        tied_embed_init_std=0.0055,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.90,
        muon_momentum_warmup_steps=900,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=0,
        lfqat_min_prob=1.0,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.0,
        lfqat_fisher_weight=0.0,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.036,
        gap_delta=0.001,
        runtime_multiplier=0.98,
        notes="Full-budget compile-friendly stable Muon recipe: maximizes raw quality per minute by keeping the fast compiled path alive.",
        iterations=5000,
        val_loss_every=200,
        train_log_every=50,
    ),
    TrainingSpec(
        name="stable_muon97_compiled_throughput",
        lr_schedule="cosine",
        warmup_steps=24,
        warmdown_iters=1700,
        lr_warmup_iters=24,
        min_lr_scale=0.18,
        tied_embed_lr=0.0020,
        matrix_lr=0.0014,
        scalar_lr=0.0014,
        tied_embed_init_std=0.0052,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.90,
        muon_momentum_warmup_steps=850,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=0,
        lfqat_min_prob=1.0,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.0,
        lfqat_fisher_weight=0.0,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.040,
        gap_delta=0.002,
        runtime_multiplier=0.95,
        notes="Compile-first throughput recipe that leans harder into extra update count on proven dense/int8 paths.",
        iterations=5200,
        val_loss_every=200,
        train_log_every=50,
    ),
    TrainingSpec(
        name="stable_muon97_compiled_sparseeval",
        lr_schedule="cosine",
        warmup_steps=24,
        warmdown_iters=1850,
        lr_warmup_iters=24,
        min_lr_scale=0.17,
        tied_embed_lr=0.0020,
        matrix_lr=0.0014,
        scalar_lr=0.0014,
        tied_embed_init_std=0.0052,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.90,
        muon_momentum_warmup_steps=850,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=0,
        lfqat_min_prob=1.0,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.0,
        lfqat_fisher_weight=0.0,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.043,
        gap_delta=0.002,
        runtime_multiplier=0.94,
        notes="Compile-first throughput recipe with sparser eval/log cadence to turn more of the 600s budget into optimizer steps.",
        iterations=6000,
        val_loss_every=250,
        train_log_every=100,
    ),
    TrainingSpec(
        name="stable_muon97_compiled_maxtrain",
        lr_schedule="cosine",
        warmup_steps=24,
        warmdown_iters=2050,
        lr_warmup_iters=24,
        min_lr_scale=0.16,
        tied_embed_lr=0.0020,
        matrix_lr=0.00135,
        scalar_lr=0.00135,
        tied_embed_init_std=0.0052,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.90,
        muon_momentum_warmup_steps=900,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=0,
        lfqat_min_prob=1.0,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.0,
        lfqat_fisher_weight=0.0,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.046,
        gap_delta=0.003,
        runtime_multiplier=0.99,
        notes="Full-budget compiled tail that trades eval frequency for raw optimization time while keeping the fast pure-training path intact.",
        iterations=6200,
        val_loss_every=250,
        train_log_every=100,
    ),
    TrainingSpec(
        name="stable_muon97_compiled_longtail",
        lr_schedule="cosine",
        warmup_steps=24,
        warmdown_iters=2250,
        lr_warmup_iters=24,
        min_lr_scale=0.14,
        tied_embed_lr=0.00195,
        matrix_lr=0.0013,
        scalar_lr=0.0013,
        tied_embed_init_std=0.0052,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.90,
        muon_momentum_warmup_steps=900,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=0,
        lfqat_full_step=0,
        lfqat_min_prob=1.0,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.0,
        lfqat_fisher_weight=0.0,
        lfqat_temperature=2.0,
        train_compression_aware_weight=0.0,
        train_grad_only_name_patterns="",
        train_grad_skip_name_patterns="",
        clean_delta=-0.047,
        gap_delta=0.003,
        runtime_multiplier=1.00,
        notes="Compile-friendly long-tail recipe that pushes a bit harder on raw quality while staying close to the official runtime boundary.",
        iterations=6600,
        val_loss_every=300,
        train_log_every=100,
    ),
    TrainingSpec(
        name="depth_push_targeted_fullbudget",
        lr_schedule="cosine",
        warmup_steps=30,
        warmdown_iters=2000,
        lr_warmup_iters=30,
        min_lr_scale=0.10,
        tied_embed_lr=0.0026,
        matrix_lr=0.0019,
        scalar_lr=0.0019,
        tied_embed_init_std=0.0040,
        muon_momentum=0.97,
        muon_backend_steps=6,
        muon_momentum_warmup_start=0.88,
        muon_momentum_warmup_steps=700,
        beta1=0.9,
        beta2=0.96,
        adam_eps=1e-8,
        lfqat_start_step=10,
        lfqat_full_step=120,
        lfqat_min_prob=0.15,
        lfqat_max_prob=1.0,
        lfqat_kl_weight=0.05,
        lfqat_fisher_weight=0.010,
        lfqat_temperature=2.2,
        train_compression_aware_weight=0.0012,
        train_grad_only_name_patterns="blocks.5,blocks.6,blocks.7,blocks.8,blocks.9,blocks.10,blocks.11",
        train_grad_skip_name_patterns="",
        clean_delta=-0.040,
        gap_delta=-0.016,
        runtime_multiplier=1.48,
        notes="Full-budget targeted upper-block recipe derived from the underused 8xH100 anchor.",
        iterations=3500,
        val_loss_every=200,
        train_log_every=50,
    ),
)

EXPORT_SPECS = (
    ExportSpec("baseline_export", "mixed_int4_int8_packed_v2", ("fc_hi", "proj_hi"), (), "", 64, 99.9, 64, 0.0, 0.055, 0, 0.05, "Anchor mixed export."),
    ExportSpec("fchi_only", "mixed_int4_int8_packed_v2", ("fc_hi",), (), "", 64, 99.9, 64, 0.008, 0.050, 350_000, 0.06, "Conservative byte-efficient export."),
    ExportSpec("fchi_attnhi_fp16", "mixed_int4_int8_packed_v2", ("fc_hi",), ("attn_hi",), "", 64, 99.9, 64, 0.030, 0.044, 520_000, 0.16, "Compromise hybrid: keep proj in int8 while protecting upper attention outputs in fp16."),
    ExportSpec("fchi_attnhi_fp16_b128", "mixed_int4_int8_packed_v2", ("fc_hi",), ("attn_hi",), "", 128, 99.7, 128, 0.027, 0.045, 360_000, 0.13, "Larger-block variant of the best current fc-hi hybrid for tighter legal headroom."),
    ExportSpec("fchi_top4_attn_top4_fp16", "mixed_int4_int8_packed_v2", ("fc_top4",), ("attn_top4",), "", 64, 99.9, 64, 0.034, 0.043, 430_000, 0.17, "Target only the last four blocks when full upper-half protection is unnecessarily broad."),
    ExportSpec("int8_all", "int8_clean_per_row_v1", (), (), "", 64, 99.9, 64, 0.018, 0.050, -350_000, 0.10, "Clean int8 path when mixed int4 is too lossy."),
    ExportSpec("int8_tok", "int8_clean_per_row_v1", (), ("tok",), "", 64, 99.9, 64, 0.028, 0.047, 450_000, 0.16, "Spend bytes on token embeddings for a strong generic recovery."),
    ExportSpec("int8_attnhi", "int8_clean_per_row_v1", (), ("attn_hi",), "", 64, 99.9, 64, 0.026, 0.046, 550_000, 0.15, "Protect upper attention outputs directly."),
    ExportSpec("int8_attnhi_tok", "int8_clean_per_row_v1", (), ("tok", "attn_hi"), "", 64, 99.9, 64, 0.038, 0.042, 900_000, 0.24, "High-upside shipping recovery while staying plausibly legal with GQA shapes."),
    ExportSpec("projhi_attnhi_fp16", "int8_clean_per_row_v1", (), ("proj_hi", "attn_hi"), "", 64, 99.9, 64, 0.041, 0.040, 1_350_000, 0.26, "Measured low-gap policy for the 10x560 fixed-settings checkpoint: keep upper proj+attn in fp16, everything else int8."),
    ExportSpec("fcproj_attnhi_fp16", "mixed_int4_int8_packed_v2", ("fc_hi", "proj_hi"), ("attn_hi",), "", 64, 99.9, 64, 0.032, 0.044, 650_000, 0.22, "Strong hybrid policy when the checkpoint is already export-aware."),
    ExportSpec("fcproj_attnhi_fp16_b128", "mixed_int4_int8_packed_v2", ("fc_hi", "proj_hi"), ("attn_hi",), "", 128, 99.7, 128, 0.028, 0.045, 500_000, 0.19, "Same hybrid policy with larger int4 blocks to buy back bytes on near-cap shapes."),
    ExportSpec("fcproj_top4_attn_top4_fp16", "mixed_int4_int8_packed_v2", ("fc_top4", "proj_top4"), ("attn_top4",), "", 64, 99.9, 64, 0.036, 0.043, 780_000, 0.25, "Focus the hybrid policy on only the final four blocks when full upper-half fp16/int4 is too byte-heavy."),
    ExportSpec("fcproj_top5_attn_top5_fp16", "mixed_int4_int8_packed_v2", ("fc_top5", "proj_top5"), ("attn_top5",), "", 64, 99.9, 64, 0.038, 0.042, 860_000, 0.27, "Slightly broader top-of-stack hybrid for depth-heavy candidates that can still stay under the byte cap."),
    ExportSpec("fcproj_tok_fp16", "mixed_int4_int8_packed_v2", ("fc_hi", "proj_hi"), ("tok",), "", 64, 99.9, 64, 0.030, 0.045, 700_000, 0.20, "Spend bytes on token embeddings instead of attention projections."),
    ExportSpec("fchi_tok_attnhi_fp16", "mixed_int4_int8_packed_v2", ("fc_hi",), ("tok", "attn_hi"), "", 64, 99.7, 64, 0.040, 0.041, 900_000, 0.29, "Stronger recovery path that keeps proj in int8 but spends extra bytes on tok+attn protection."),
    ExportSpec("fcproj_tok_attnhi_fp16", "mixed_int4_int8_packed_v2", ("fc_hi", "proj_hi"), ("tok", "attn_hi"), "", 64, 99.7, 64, 0.045, 0.040, 1_050_000, 0.34, "Aggressive near-cap hybrid policy for final finalists only."),
    ExportSpec("baseline_export_b128", "mixed_int4_int8_packed_v2", ("fc_hi", "proj_hi"), (), "", 128, 99.7, 128, 0.006, 0.052, -120_000, 0.06, "Larger block export with slightly better packing and slightly worse distortion."),
)


def load_export_gap_calibration(path: Path = LOCAL_EXPORT_GAP_JSON) -> ExportGapCalibration | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    best = payload.get("best_legal", {})
    try:
        return ExportGapCalibration(
            name=str(best["name"]),
            num_layers=int(cfg["num_layers"]),
            model_dim=int(cfg["model_dim"]),
            num_kv_heads=int(cfg["num_kv_heads"]),
            train_seq_len=int(cfg["train_seq_len"]),
            measured_gap=float(best["export_gap_bpb"]),
            measured_total_bytes=int(best["total_bytes"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


LOCAL_EXPORT_GAP_CALIBRATION = load_export_gap_calibration()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline SOTA seeker across tokenizer, model, token flow, training, and export settings.")
    p.add_argument("--experiment-table", type=Path, default=ROOT / "research" / "experiment_table.md")
    p.add_argument("--target-bpb", type=float, default=1.1)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return p.parse_args()


def classify_target_band(best_case_shipped: float, target_bpb: float) -> str:
    remaining = best_case_shipped - target_bpb
    if remaining <= 0:
        return "target-reachable"
    if remaining <= 0.06:
        return "target-stretch"
    if remaining <= 0.15:
        return "target-near"
    if remaining <= 0.30:
        return "target-far"
    return "target-remote"


def classify_launch_tier(*, legal: bool, runtime_ratio: float, init_mode: str) -> str:
    if not legal:
        return "illegal"
    if init_mode != "scratch":
        return "research-only"
    if runtime_ratio <= 1.0:
        return "official-ready"
    if runtime_ratio <= 1.08:
        return "official-stretch"
    return "research-only"


def matched_wallclock_seconds(nproc: int) -> int:
    return int(round((8 * 600) / nproc))


def find_run(rows, run_id: str):
    for row in rows:
        if row.run == run_id:
            return row
    return None


def training_family(name: str) -> str:
    return name.removesuffix("_fullbudget")


def parse_train_runtime_seconds(runtime: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)s train", runtime)
    if match:
        return float(match.group(1))
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)ms", runtime)
    if match:
        return float(match.group(1)) / 1000.0
    return None


def sp1024_anchor(rows) -> tuple[float, float, int, list[str]]:
    base = find_run(rows, "baseline_public_10min")
    long = find_run(rows, "baseline_public_4h")
    shipped = base.shipped_bpb if base and base.shipped_bpb is not None else 1.2244
    clean = max(1.15, shipped - 0.055)
    rationale = [f"Public `sp1024` baseline anchors shipped quality at `{shipped:.4f}` bpb."]
    if long and long.shipped_bpb is not None:
        rationale.append(f"Longer `sp1024` training reached `{long.shipped_bpb:.4f}` shipped bpb, proving additional raw headroom exists.")
        clean = min(clean, long.shipped_bpb - 0.030)
    total = base.total_bytes if base and base.total_bytes is not None else 15_863_489
    return clean, shipped, total, rationale


def raw_sp1024_metrics(
    rows,
    tokenizer: TokenizerSpec,
    model: ModelSpec,
    flow: TokenFlowSpec,
    training: TrainingSpec,
    export: ExportSpec,
) -> tuple[float, float, int, float]:
    anchor_clean, anchor_shipped, anchor_total, _ = sp1024_anchor(rows)
    base_gap = max(anchor_shipped - anchor_clean, 0.045)
    clean = anchor_clean + tokenizer.clean_delta + model.clean_delta + flow.clean_delta + training.clean_delta
    gap = max(
        export.optimistic_gap_floor,
        base_gap + tokenizer.gap_delta + model.gap_delta + flow.gap_delta + training.gap_delta - export.gap_recovery,
    )
    total_bytes = anchor_total + model.byte_delta + export.byte_delta + max(COUNTED_CODE_PATH.stat().st_size - 66_325, 0)
    runtime_ratio = tokenizer.runtime_multiplier * model.runtime_multiplier * flow.runtime_multiplier * training.runtime_multiplier
    return clean, gap, total_bytes, runtime_ratio


def sp1024_calibration(rows) -> SpCalibrationAnchor | None:
    run = find_run(rows, "final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi")
    if not run or run.clean_bpb is None or run.shipped_bpb is None or run.total_bytes is None:
        return None
    runtime_seconds = parse_train_runtime_seconds(run.runtime)
    if runtime_seconds is None:
        return None

    tokenizer = next(spec for spec in TOKENIZERS if spec.name == "sp1024_bpe")
    model = next(spec for spec in SP1024_MODELS if spec.name == "sp_10x512_kv2")
    flow = next(spec for spec in TOKEN_FLOWS if spec.name == "context_896_dense")
    training = next(spec for spec in TRAINING_SPECS if spec.name == "stable_muon97")
    export = next(spec for spec in EXPORT_SPECS if spec.name == "int8_attnhi")
    raw_clean, raw_gap, raw_total_bytes, raw_runtime_ratio = raw_sp1024_metrics(rows, tokenizer, model, flow, training, export)
    return SpCalibrationAnchor(
        run_id=run.run,
        model_name=model.name,
        flow_name=flow.name,
        training_name=training.name,
        export_name=export.name,
        raw_clean=raw_clean,
        raw_gap=raw_gap,
        raw_total_bytes=raw_total_bytes,
        raw_runtime_ratio=raw_runtime_ratio,
        actual_clean=run.clean_bpb,
        actual_gap=run.shipped_bpb - run.clean_bpb,
        actual_total_bytes=run.total_bytes,
        actual_runtime_ratio=runtime_seconds / 600.0,
    )


def u4k_anchor(rows) -> tuple[float, float, int, list[str]]:
    official = find_run(rows, "runpod_dense_u4k_12x608_lfqat_continue80m_20260319")
    clean = official.clean_bpb if official and official.clean_bpb is not None else 1.3329
    shipped = official.shipped_bpb if official and official.shipped_bpb is not None else 1.6101
    total = official.total_bytes if official and official.total_bytes is not None else 10_842_824
    rationale = [f"Official dense `u4k` anchor sits at clean `{clean:.4f}` / shipped `{shipped:.4f}` bpb."]
    return clean, shipped, total, rationale


def build_patterns(layers: int, export: ExportSpec) -> tuple[str, str]:
    int4 = ",".join(build_group_patterns(layers, export.int4_groups))
    fp16 = ",".join(build_fp16_patterns(export.fp16_groups, layers))
    return int4, fp16


def compatible_combo(model: ModelSpec, flow: TokenFlowSpec, training: TrainingSpec, export: ExportSpec) -> bool:
    head_dim = model.model_dim // model.num_heads
    if model.model_dim % model.num_heads != 0:
        return False
    if model.num_heads % model.num_kv_heads != 0:
        return False
    if head_dim % 2 != 0:
        return False
    if model.num_unique_layers <= 0 or model.num_unique_layers > model.layers:
        return False
    if flow.val_batch_size // max(flow.grad_accum_steps, 1) < flow.train_seq_len:
        return False
    if export.int4_block_size <= 0 or export.train_qat_block_size <= 0:
        return False
    if export.quant_format == "int8_clean_per_row_v1" and export.int4_groups:
        return False
    if model.num_unique_layers < model.layers and training.train_grad_only_name_patterns:
        return False
    return True


def calibration_similarity(model: ModelSpec, flow: TokenFlowSpec, training: TrainingSpec, export: ExportSpec, calibration: SpCalibrationAnchor) -> float:
    if (
        model.name == calibration.model_name
        and flow.name == calibration.flow_name
        and training.name == calibration.training_name
        and export.name == calibration.export_name
    ):
        return 1.0

    similarity = 0.15
    if model.name == calibration.model_name:
        similarity += 0.35
    elif model.layers == 10 and model.model_dim == 512 and model.num_kv_heads == 2:
        similarity += 0.18
    if flow.name == calibration.flow_name:
        similarity += 0.18
    if training_family(training.name) == training_family(calibration.training_name):
        similarity += 0.16
    if export.name == calibration.export_name:
        similarity += 0.16
    elif export.quant_format == "int8_clean_per_row_v1":
        similarity += 0.08
    return min(1.0, similarity)


def apply_sp1024_calibration(
    *,
    raw_clean: float,
    raw_gap: float,
    raw_total_bytes: int,
    raw_runtime_ratio: float,
    model: ModelSpec,
    flow: TokenFlowSpec,
    training: TrainingSpec,
    export: ExportSpec,
    calibration: SpCalibrationAnchor | None,
) -> tuple[float, float, int, float, float, list[str]]:
    if calibration is None:
        return raw_clean, raw_gap, raw_total_bytes, raw_runtime_ratio, 0.0, []

    if (
        model.name == calibration.model_name
        and flow.name == calibration.flow_name
        and training.name == calibration.training_name
        and export.name == calibration.export_name
    ):
        runtime_ratio = max(0.05, calibration.actual_runtime_ratio)
        return (
            calibration.actual_clean,
            max(export.optimistic_gap_floor, calibration.actual_gap),
            calibration.actual_total_bytes,
            runtime_ratio,
            1.0,
            [
                f"Exact `8xH100` calibration anchor `{calibration.run_id}` overrides the old offline estimate.",
                f"Measured clean/shipped is `{calibration.actual_clean:.4f}` / `{calibration.actual_clean + calibration.actual_gap:.4f}` at runtime ratio `{runtime_ratio:.3f}`.",
            ],
        )

    blend = calibration_similarity(model, flow, training, export, calibration)
    clean_shift = calibration.actual_clean - calibration.raw_clean
    gap_shift = calibration.actual_gap - calibration.raw_gap
    byte_shift = calibration.actual_total_bytes - calibration.raw_total_bytes
    runtime_scale = calibration.actual_runtime_ratio / max(calibration.raw_runtime_ratio, 1e-9)
    calibrated_clean = raw_clean + clean_shift * blend
    calibrated_gap = max(export.optimistic_gap_floor, raw_gap + gap_shift * blend)
    calibrated_total_bytes = max(COUNTED_CODE_PATH.stat().st_size + 500_000, int(round(raw_total_bytes + byte_shift * blend)))
    calibrated_runtime_ratio = max(
        0.05,
        raw_runtime_ratio * (1.0 + (runtime_scale - 1.0) * max(0.65, blend)),
    )
    notes = [
        (
            f"Calibrated against real `8xH100` anchor `{calibration.run_id}` "
            f"(blend `{blend:.2f}`, clean shift `{clean_shift:+.4f}`, gap shift `{gap_shift:+.4f}`)."
        )
    ]
    return calibrated_clean, calibrated_gap, calibrated_total_bytes, calibrated_runtime_ratio, blend, notes


def apply_export_gap_calibration(
    *,
    gap: float,
    total_bytes: int,
    model: ModelSpec,
    flow: TokenFlowSpec,
    export: ExportSpec,
) -> tuple[float, int, float, list[str]]:
    calibration = LOCAL_EXPORT_GAP_CALIBRATION
    if calibration is None:
        return gap, total_bytes, 0.0, []
    if export.name != calibration.name:
        return gap, total_bytes, 0.0, []
    if model.layers != calibration.num_layers or model.model_dim != calibration.model_dim:
        return gap, total_bytes, 0.0, []
    if model.num_kv_heads != calibration.num_kv_heads or flow.train_seq_len != calibration.train_seq_len:
        return gap, total_bytes, 0.0, []

    calibrated_gap = max(export.optimistic_gap_floor, min(gap, calibration.measured_gap + 0.010))
    calibrated_total_bytes = max(total_bytes, calibration.measured_total_bytes)
    notes = [
        (
            f"Measured export-gap calibration `{calibration.name}` overrides the old shipping guess "
            f"for `{model.layers}x{model.model_dim}` / `seq={flow.train_seq_len}` "
            f"(gap `{calibration.measured_gap:.4f}`, total `{calibration.measured_total_bytes}`)."
        )
    ]
    return calibrated_gap, calibrated_total_bytes, 0.65, notes


def build_sp_command(
    candidate_name: str,
    tokenizer: TokenizerSpec,
    model: ModelSpec,
    flow: TokenFlowSpec,
    training: TrainingSpec,
    export: ExportSpec,
    *,
    nproc: int,
    wallclock_seconds: int | None = None,
) -> str:
    int4_patterns, fp16_patterns = build_patterns(model.layers, export)
    compile_friendly_training = (
        training.lfqat_kl_weight <= 0.0
        and training.lfqat_fisher_weight <= 0.0
        and training.train_compression_aware_weight <= 0.0
        and training.lfqat_min_prob >= 1.0
        and training.lfqat_max_prob >= 1.0
    )
    compile_friendly_int8 = (
        export.quant_format == "int8_clean_per_row_v1"
        and compile_friendly_training
    )
    if wallclock_seconds is None:
        wallclock_seconds = matched_wallclock_seconds(nproc) if nproc != 1 else 600
    grad_accum_steps = flow.grad_accum_steps
    if nproc == 1 and flow.train_batch_tokens > SINGLE_GPU_PROBE_TOKEN_CAP:
        grad_accum_steps = max(
            grad_accum_steps,
            math.ceil(flow.train_batch_tokens / SINGLE_GPU_PROBE_TOKEN_CAP),
        )
    env_parts = [
        f"RUN_ID={candidate_name}",
        f"NPROC_PER_NODE={nproc}",
        f"MAX_WALLCLOCK_SECONDS={wallclock_seconds}",
        *(["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"] if nproc == 1 else []),
        f"DATA_PATH={tokenizer.data_path}",
        f"TOKENIZER_PATH={tokenizer.tokenizer_path}",
        f"VOCAB_SIZE={tokenizer.vocab_size}",
        f"NUM_LAYERS={model.layers}",
        f"NUM_UNIQUE_LAYERS={model.num_unique_layers}",
        f"MODEL_DIM={model.model_dim}",
        f"NUM_HEADS={model.num_heads}",
        f"NUM_KV_HEADS={model.num_kv_heads}",
        f"MLP_MULT={model.mlp_mult}",
        "TIE_EMBEDDINGS=1",
        "INIT_MODEL_PATH=",
        f"TRAIN_SEQ_LEN={flow.train_seq_len}",
        f"TRAIN_BATCH_TOKENS={flow.train_batch_tokens}",
        f"VAL_BATCH_SIZE={flow.val_batch_size}",
        f"GRAD_CLIP_NORM={flow.grad_clip_norm}",
        f"GRAD_ACCUM_STEPS={grad_accum_steps}",
        f"QK_GAIN_INIT={flow.qk_gain_init}",
        f"ROPE_BASE={flow.rope_base}",
        f"LOGIT_SOFTCAP={flow.logit_softcap}",
        f"LR_SCHEDULE={training.lr_schedule}",
        f"WARMUP_STEPS={training.warmup_steps}",
        f"WARMDOWN_ITERS={training.warmdown_iters}",
        f"LR_WARMUP_ITERS={training.lr_warmup_iters}",
        f"MIN_LR_SCALE={training.min_lr_scale}",
        f"TIED_EMBED_LR={training.tied_embed_lr}",
        f"MATRIX_LR={training.matrix_lr}",
        f"SCALAR_LR={training.scalar_lr}",
        f"TIED_EMBED_INIT_STD={training.tied_embed_init_std}",
        f"MUON_MOMENTUM={training.muon_momentum}",
        f"MUON_BACKEND_STEPS={training.muon_backend_steps}",
        f"MUON_MOMENTUM_WARMUP_START={training.muon_momentum_warmup_start}",
        f"MUON_MOMENTUM_WARMUP_STEPS={training.muon_momentum_warmup_steps}",
        f"BETA1={training.beta1}",
        f"BETA2={training.beta2}",
        f"ADAM_EPS={training.adam_eps}",
        f"ITERATIONS={training.iterations}",
        f"VAL_LOSS_EVERY={training.val_loss_every}",
        f"TRAIN_LOG_EVERY={training.train_log_every}",
        "VAL_AT_STEP_ZERO=0",
        f"WALLCLOCK_SYNC_EVERY={16 if nproc > 1 else 1}",
        f"DDP_STATIC_GRAPH={1 if nproc > 1 and compile_friendly_training else 0}",
        f"DDP_GRADIENT_AS_BUCKET_VIEW={1 if nproc > 1 else 0}",
        f"LFQAT_START_STEP={training.lfqat_start_step}",
        f"LFQAT_FULL_STEP={training.lfqat_full_step}",
        f"LFQAT_MIN_PROB={training.lfqat_min_prob}",
        f"LFQAT_MAX_PROB={training.lfqat_max_prob}",
        f"LFQAT_KL_WEIGHT={training.lfqat_kl_weight if export.quant_format != 'int8_clean_per_row_v1' else max(training.lfqat_kl_weight * 0.35, 0.0)}",
        f"LFQAT_FISHER_WEIGHT={training.lfqat_fisher_weight if export.quant_format != 'int8_clean_per_row_v1' else max(training.lfqat_fisher_weight * 0.35, 0.0)}",
        f"LFQAT_TEMPERATURE={training.lfqat_temperature}",
        f"TRAIN_COMPRESSION_AWARE_WEIGHT={training.train_compression_aware_weight if export.quant_format != 'int8_clean_per_row_v1' else max(training.train_compression_aware_weight * 0.2, 0.0)}",
        f"TRAIN_GRAD_ONLY_NAME_PATTERNS={training.train_grad_only_name_patterns}",
        f"TRAIN_GRAD_SKIP_NAME_PATTERNS={training.train_grad_skip_name_patterns}",
        f"INT4_BLOCK_SIZE={export.int4_block_size}",
        f"INT4_CLIP_PERCENTILE={export.int4_clip_percentile}",
        f"TRAIN_QAT_BLOCK_SIZE={export.train_qat_block_size}",
    ]
    if export.quant_format == "int8_clean_per_row_v1":
        env_parts.extend(
            [
                "QUANT_FORMAT=int8_clean_per_row_v1",
                "TARGET_EXPORT_NAME_PATTERNS=",
                "INT4_NAME_PATTERNS=",
                "TRAIN_QAT_NAME_PATTERNS=",
                "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=",
            ]
        )
    else:
        env_parts.extend(
            [
                "QUANT_FORMAT=mixed_int4_int8_packed_v2",
                f"TARGET_EXPORT_NAME_PATTERNS={int4_patterns}",
                f"INT4_NAME_PATTERNS={int4_patterns}",
                f"TRAIN_QAT_NAME_PATTERNS={'' if compile_friendly_training else (int4_patterns or '__never__')}",
                f"TRAIN_COMPRESSION_AWARE_NAME_PATTERNS={'' if compile_friendly_training else int4_patterns}",
            ]
        )
    if fp16_patterns:
        env_parts.append(f"INT8_KEEP_FLOAT_FP16_NAME_PATTERNS={fp16_patterns}")
    if export.fp32_name_patterns:
        env_parts.append(f"INT8_KEEP_FLOAT_FP32_NAME_PATTERNS={export.fp32_name_patterns}")
    return " \\\n  ".join(env_parts + [tokenizer.launcher])


def downshift_command(command: str, *, nproc: int, wallclock_seconds: int) -> str:
    return (
        command.replace("NPROC_PER_NODE=8", f"NPROC_PER_NODE={nproc}")
        .replace("MAX_WALLCLOCK_SECONDS=600", f"MAX_WALLCLOCK_SECONDS={wallclock_seconds}")
        .replace("RUN_ID=final_", f"RUN_ID=final_{nproc}x_")
    )


def estimate_sp_candidate(rows, tokenizer: TokenizerSpec, model: ModelSpec, flow: TokenFlowSpec, training: TrainingSpec, export: ExportSpec, target_bpb: float) -> SeekerCandidate:
    _, _, _, rationale = sp1024_anchor(rows)
    raw_clean, raw_gap, raw_total_bytes, raw_runtime_ratio = raw_sp1024_metrics(rows, tokenizer, model, flow, training, export)
    calibration = sp1024_calibration(rows)
    clean, gap, total_bytes, runtime_ratio, calibration_blend, calibration_notes = apply_sp1024_calibration(
        raw_clean=raw_clean,
        raw_gap=raw_gap,
        raw_total_bytes=raw_total_bytes,
        raw_runtime_ratio=raw_runtime_ratio,
        model=model,
        flow=flow,
        training=training,
        export=export,
        calibration=calibration,
    )
    gap, total_bytes, export_calibration_blend, export_calibration_notes = apply_export_gap_calibration(
        gap=gap,
        total_bytes=total_bytes,
        model=model,
        flow=flow,
        export=export,
    )
    shipped = clean + gap
    best_case_shipped = clean + export.optimistic_gap_floor
    clean_gain_needed = max(0.0, best_case_shipped - target_bpb)
    legal = total_bytes <= 16_000_000
    launch_tier = classify_launch_tier(legal=legal, runtime_ratio=runtime_ratio, init_mode=model.init_mode)
    official_eligible = launch_tier == "official-ready"
    target_band = classify_target_band(best_case_shipped, target_bpb)
    confidence = max(0.08, min(0.95, model.evidence_weight - export.byte_risk * 0.12 - max(0.0, runtime_ratio - 1.0) * 0.25))
    confidence = min(0.97, confidence + calibration_blend * 0.10 + export_calibration_blend * 0.05)
    objective = shipped
    objective += max(0.0, best_case_shipped - target_bpb) * 1.1
    objective += gap * 0.35
    objective += max(0.0, runtime_ratio - 1.0) * 0.55
    objective += max(0.0, total_bytes - 16_000_000) / 1_000_000
    objective += (0.06 if launch_tier == "official-stretch" else 0.0)
    objective += (0.18 if launch_tier == "research-only" else 0.0)
    objective += (1.0 - confidence) * 0.15
    rationale = list(rationale)
    rationale.extend(
        [
            *calibration_notes,
            *export_calibration_notes,
            f"Model `{model.name}` contributes `{model.clean_delta:+.4f}` clean-bpb delta with byte delta `{model.byte_delta:+,}`.",
            f"Token flow `{flow.name}` contributes `{flow.clean_delta:+.4f}` clean and `{flow.gap_delta:+.4f}` export-gap delta.",
            f"Training `{training.name}` contributes `{training.clean_delta:+.4f}` clean and `{training.gap_delta:+.4f}` export-gap delta.",
            f"Export `{export.name}` targets an optimistic gap floor of `{export.optimistic_gap_floor:.4f}`.",
            f"Best-case shipped estimate is `{best_case_shipped:.4f}`, still `{clean_gain_needed:.4f}` away from `1.1`.",
        ]
    )
    name = f"seeker_{tokenizer.name}_{model.name}_{flow.name}_{training.name}_{export.name}"
    return SeekerCandidate(
        rank=0,
        name=name,
        source="offline_joint_search",
        tokenizer=tokenizer.name,
        model=model.name,
        token_flow=flow.name,
        training=training.name,
        export_policy=export.name,
        layers=model.layers,
        model_dim=model.model_dim,
        num_heads=model.num_heads,
        num_kv_heads=model.num_kv_heads,
        mlp_mult=model.mlp_mult,
        init_mode=model.init_mode,
        estimated_clean_bpb=clean,
        estimated_shipped_bpb=shipped,
        estimated_gap=gap,
        best_case_shipped_bpb=best_case_shipped,
        clean_gain_needed_after_best_export=clean_gain_needed,
        estimated_total_bytes=total_bytes,
        estimated_runtime_ratio=runtime_ratio,
        official_eligible=official_eligible,
        launch_tier=launch_tier,
        legal=legal,
        target_band=target_band,
        objective=objective,
        confidence=confidence,
        rationale=rationale,
        screen_command=build_sp_command(name, tokenizer, model, flow, training, export, nproc=1),
        final_6xh100_command=build_sp_command(name.replace("seeker_", "final_6x_"), tokenizer, model, flow, training, export, nproc=6, wallclock_seconds=matched_wallclock_seconds(6)),
        final_8xh100_command=build_sp_command(name.replace("seeker_", "final_"), tokenizer, model, flow, training, export, nproc=8),
    )


def convert_u4k_trial(trial: TrialCandidate, target_bpb: float) -> SeekerCandidate:
    best_case = trial.best_case_shipped_bpb
    clean_gain_needed = max(0.0, best_case - target_bpb)
    launch_tier = classify_launch_tier(legal=trial.legal, runtime_ratio=trial.estimated_runtime_ratio, init_mode=trial.init_mode)
    official_eligible = launch_tier == "official-ready"
    objective = trial.stage1_objective + (0.06 if launch_tier == "official-stretch" else 0.0) + (0.18 if launch_tier == "research-only" else 0.0)
    return SeekerCandidate(
        rank=0,
        name=trial.name,
        source="u4k_trial_search",
        tokenizer="u4k_unigram",
        model=f"u4k_{trial.layers}x{trial.model_dim}",
        token_flow="baked_in",
        training=trial.recipe,
        export_policy=trial.export_policy,
        layers=trial.layers,
        model_dim=trial.model_dim,
        num_heads=8,
        num_kv_heads=trial.num_kv_heads,
        mlp_mult=trial.mlp_mult,
        init_mode=trial.init_mode,
        estimated_clean_bpb=trial.estimated_clean_bpb,
        estimated_shipped_bpb=trial.estimated_shipped_bpb,
        estimated_gap=trial.estimated_gap,
        best_case_shipped_bpb=best_case,
        clean_gain_needed_after_best_export=clean_gain_needed,
        estimated_total_bytes=trial.estimated_total_bytes,
        estimated_runtime_ratio=trial.estimated_runtime_ratio,
        official_eligible=official_eligible,
        launch_tier=launch_tier,
        legal=trial.legal,
        target_band=trial.target_band,
        objective=objective,
        confidence=trial.confidence,
        rationale=list(trial.rationale),
        screen_command=trial.screen_command,
        final_6xh100_command=downshift_command(trial.final_8xh100_command, nproc=6, wallclock_seconds=matched_wallclock_seconds(6)),
        final_8xh100_command=trial.final_8xh100_command,
    )


def search_seeker(rows, target_bpb: float) -> list[SeekerCandidate]:
    candidates: list[SeekerCandidate] = []
    sp = next(spec for spec in TOKENIZERS if spec.family == "sp1024")
    for model in SP1024_MODELS:
        for flow in TOKEN_FLOWS:
            for training in TRAINING_SPECS:
                for export in EXPORT_SPECS:
                    if not compatible_combo(model, flow, training, export):
                        continue
                    candidates.append(estimate_sp_candidate(rows, sp, model, flow, training, export, target_bpb))

    u4k_trials = search_trials(rows, target_bpb)
    candidates.extend(convert_u4k_trial(trial, target_bpb) for trial in u4k_trials[:12])
    candidates.sort(key=lambda c: (c.objective, c.best_case_shipped_bpb, c.estimated_shipped_bpb))
    for idx, candidate in enumerate(candidates, start=1):
        candidate.rank = idx
    return candidates


def render_markdown(candidates: list[SeekerCandidate], target_bpb: float) -> str:
    best = candidates[0]
    best_official = next((c for c in candidates if c.launch_tier in {"official-ready", "official-stretch"}), best)
    lowest_shipped_official = min(
        (c for c in candidates if c.launch_tier == "official-ready"),
        key=lambda c: (c.estimated_shipped_bpb, c.objective),
        default=best_official,
    )
    closest = min(candidates, key=lambda c: (c.best_case_shipped_bpb, c.objective))
    official_ready = [c for c in candidates if c.launch_tier == "official-ready"][:6]
    official_stretch = [c for c in candidates if c.launch_tier == "official-stretch"][:6]
    research = [c for c in candidates if c.launch_tier == "research-only"][:6]
    lines = [
        "# SOTA Seeker",
        "",
        f"- Target shipped bpb: `{target_bpb:.4f}`",
        f"- Counted code bytes: `{COUNTED_CODE_PATH.stat().st_size}`",
        f"- Search space: `1 combinatorial sp1024 family x {len(SP1024_MODELS)} models x {len(TOKEN_FLOWS)} token-flows x {len(TRAINING_SPECS)} training recipes x {len(EXPORT_SPECS)} export policies + 12 carried-over u4k finalists`",
        "",
        "## Verdict",
        "",
        f"- Best official candidate by overall objective: `{best_official.name}` at estimated shipped `{best_official.estimated_shipped_bpb:.4f}` bpb.",
        f"- Lowest-estimated shipped official-ready candidate: `{lowest_shipped_official.name}` at estimated shipped `{lowest_shipped_official.estimated_shipped_bpb:.4f}` bpb.",
        f"- Highest-upside candidate: `{closest.name}` with best-case `{closest.best_case_shipped_bpb:.4f}` bpb.",
        f"- Clean gain still needed after best-case export: `{closest.clean_gain_needed_after_best_export:.4f}` bpb.",
        (
            "- Current verdict: the search space contains launchable target-stretch candidates."
            if any(c.launch_tier in {'official-ready', 'official-stretch'} and c.target_band in {'target-reachable', 'target-stretch'} for c in candidates)
            else "- Current verdict: no current candidate is target-stretch yet; the best route still needs a materially stronger checkpoint."
        ),
        "",
        "## Official-Ready Candidates",
        "",
        "| Rank | Candidate | Tokenizer | Model | Token flow | Training | Export | Est. shipped | Best-case | Bytes | Runtime | Band |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in official_ready:
        lines.append(
            f"| `{candidate.rank}` | `{candidate.name}` | `{candidate.tokenizer}` | `{candidate.layers}x{candidate.model_dim}` | "
            f"`{candidate.token_flow}` | `{candidate.training}` | `{candidate.export_policy}` | "
            f"`{candidate.estimated_shipped_bpb:.4f}` | `{candidate.best_case_shipped_bpb:.4f}` | "
            f"`{candidate.estimated_total_bytes}` | `{candidate.estimated_runtime_ratio:.3f}` | `{candidate.target_band}` |"
        )
    lines.extend(["", "## Official-Stretch Candidates", "", "| Rank | Candidate | Tokenizer | Model | Est. shipped | Best-case | Bytes | Runtime | Band |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for candidate in official_stretch:
        lines.append(
            f"| `{candidate.rank}` | `{candidate.name}` | `{candidate.tokenizer}` | `{candidate.layers}x{candidate.model_dim}` | "
            f"`{candidate.estimated_shipped_bpb:.4f}` | `{candidate.best_case_shipped_bpb:.4f}` | `{candidate.estimated_total_bytes}` | "
            f"`{candidate.estimated_runtime_ratio:.3f}` | `{candidate.target_band}` |"
        )
    lines.extend(["", "## Research Recovery Candidates", "", "| Rank | Candidate | Source | Est. shipped | Best-case | Tier |", "| --- | --- | --- | --- | --- | --- |"])
    for candidate in research:
        lines.append(
            f"| `{candidate.rank}` | `{candidate.name}` | `{candidate.source}` | `{candidate.estimated_shipped_bpb:.4f}` | "
            f"`{candidate.best_case_shipped_bpb:.4f}` | `{candidate.launch_tier}` |"
        )
    lines.extend(["", "## Commands", ""])
    command_candidates: list[SeekerCandidate] = []
    for candidate in (best_official, lowest_shipped_official, closest):
        if candidate.name not in {seen.name for seen in command_candidates}:
            command_candidates.append(candidate)
    for candidate in command_candidates:
        lines.extend(
            [
                f"### `{candidate.name}`",
                "",
                f"- Tokenizer: `{candidate.tokenizer}`",
                f"- Model: `{candidate.layers}x{candidate.model_dim}` / `kv={candidate.num_kv_heads}` / `mlp={candidate.mlp_mult}`",
                f"- Token flow: `{candidate.token_flow}`",
                f"- Training: `{candidate.training}`",
                f"- Export: `{candidate.export_policy}`",
                f"- Estimated shipped: `{candidate.estimated_shipped_bpb:.4f}`",
                f"- Best-case shipped: `{candidate.best_case_shipped_bpb:.4f}`",
                f"- Estimated bytes: `{candidate.estimated_total_bytes}`",
                f"- Runtime ratio: `{candidate.estimated_runtime_ratio:.3f}`",
                f"- Launch tier: `{candidate.launch_tier}`",
                "",
                "1xH100 screen:",
                "```bash",
                candidate.screen_command,
                "```",
                "",
                "6xH100 compute-matched:",
                "```bash",
                candidate.final_6xh100_command,
                "```",
                "",
                "8xH100 final:",
                "```bash",
                candidate.final_8xh100_command,
                "```",
                "",
                "Rationale:",
            ]
        )
        for item in candidate.rationale:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = parse_experiment_table(args.experiment_table)
    candidates = search_seeker(rows, args.target_bpb)
    best = candidates[0]
    best_official = next((c for c in candidates if c.official_eligible), best)
    lowest_shipped_official = min(
        (c for c in candidates if c.official_eligible),
        key=lambda c: (c.estimated_shipped_bpb, c.objective),
        default=best_official,
    )
    closest = min(candidates, key=lambda c: (c.best_case_shipped_bpb, c.objective))
    payload = {
        "target_bpb": args.target_bpb,
        "counted_code_bytes": COUNTED_CODE_PATH.stat().st_size,
        "search_space": {
            "combinatorial_tokenizer_families": 1,
            "carryover_tokenizer_families": 1,
            "sp_models": len(SP1024_MODELS),
            "token_flows": len(TOKEN_FLOWS),
            "training_recipes": len(TRAINING_SPECS),
            "export_policies": len(EXPORT_SPECS),
            "u4k_carryover_finalists": 12,
            "total_joint_combinations": len(SP1024_MODELS) * len(TOKEN_FLOWS) * len(TRAINING_SPECS) * len(EXPORT_SPECS) + 12,
        },
        "top_candidates": [asdict(candidate) for candidate in candidates[: args.top_k]],
        "best_official_ready_candidates": [asdict(candidate) for candidate in candidates if candidate.launch_tier == "official-ready"][:8],
        "best_official_stretch_candidates": [asdict(candidate) for candidate in candidates if candidate.launch_tier == "official-stretch"][:8],
        "goal_seek_summary": {
            "best_estimated_trial": best.name,
            "best_estimated_shipped_bpb": best.estimated_shipped_bpb,
            "best_official_trial": {
                "trial_id": best_official.name,
                "estimated_shipped_bpb": best_official.estimated_shipped_bpb,
                "estimated_total_bytes": best_official.estimated_total_bytes,
                "estimated_runtime_ratio": best_official.estimated_runtime_ratio,
                "launch_tier": best_official.launch_tier,
                "target_band": best_official.target_band,
            },
            "lowest_shipped_official_trial": {
                "trial_id": lowest_shipped_official.name,
                "estimated_shipped_bpb": lowest_shipped_official.estimated_shipped_bpb,
                "estimated_total_bytes": lowest_shipped_official.estimated_total_bytes,
                "estimated_runtime_ratio": lowest_shipped_official.estimated_runtime_ratio,
                "launch_tier": lowest_shipped_official.launch_tier,
                "target_band": lowest_shipped_official.target_band,
            },
            "closest_best_case_trial": {
                "trial_id": closest.name,
                "best_case_shipped_bpb": closest.best_case_shipped_bpb,
                "clean_gain_needed_after_best_export": closest.clean_gain_needed_after_best_export,
                "estimated_total_bytes": closest.estimated_total_bytes,
                "estimated_runtime_ratio": closest.estimated_runtime_ratio,
                "target_band": closest.target_band,
                "official_eligible": closest.official_eligible,
            },
            "official_target_stretch_count": sum(
                1 for candidate in candidates if candidate.launch_tier in {"official-ready", "official-stretch"} and candidate.target_band in {"target-reachable", "target-stretch"}
            ),
        },
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(candidates, args.target_bpb), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(
        f"Best SOTA seeker candidate: {best.name} "
        f"(est_shipped_bpb={best.estimated_shipped_bpb:.4f}, best_case={best.best_case_shipped_bpb:.4f}, legal={best.legal}, official={best.official_eligible})"
    )


if __name__ == "__main__":
    main()
