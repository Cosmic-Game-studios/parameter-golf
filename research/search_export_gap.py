#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.plan_sota_search import build_fp16_patterns, build_group_patterns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "research"
CLEAN_RE = re.compile(r"step:\d+/\d+ val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+)")
ROUNDTRIP_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)"
)
BYTES_RE = re.compile(r"serialized_model_int8_zlib:(?P<bytes>\d+) bytes")
TOP_GROUP_RE = re.compile(r"_(?:top|hi)")
GLOBAL_BUS_LINE_RE = re.compile(
    r"^global_bus_enabled:(?P<enabled>True|False) "
    r"global_bus_read_layers:(?P<read>\S+) "
    r"global_bus_write_layers:(?P<write>\S+) "
    r"global_bus_summary_mode:(?P<mode>\S+) "
    r"global_bus_tail_weight:(?P<tail>[-+0-9.]+) "
    r"allow_init_missing_keys:(?P<allow>True|False)$",
    re.MULTILINE,
)
SECOND_PASS_LINE_RE = re.compile(
    r"^second_pass_enabled:(?P<enabled>True|False) "
    r"second_pass_layers:(?P<layers>\S+) "
    r"second_pass_gate_init:(?P<gate>[-+0-9.]+) "
    r"second_pass_skip_init:(?P<skip>[-+0-9.]+)$",
    re.MULTILINE,
)
CROSS_SKIP_ROUTER_LINE_RE = re.compile(
    r"^cross_skip_router_enabled:(?P<enabled>True|False) "
    r"cross_skip_router_match_init:(?P<match>[-+0-9.]+) "
    r"cross_skip_router_other_init:(?P<other>[-+0-9.]+)"
    r"(?: cross_skip_router_decoder_layers:(?P<decoder>\S+) "
    r"cross_skip_router_source_layers:(?P<source>\S+))?$",
    re.MULTILINE,
)
PAIR_FEATURES_LINE_RE = re.compile(
    r"^smear_gate_enabled:(?P<smear>True|False) "
    r"smear_gate_init:(?P<smear_init>[-+0-9.]+) "
    r"bigram_hash_enabled:(?P<bigram>True|False) "
    r"bigram_hash_buckets:(?P<buckets>\d+) "
    r"bigram_hash_dim:(?P<dim>\d+) "
    r"bigram_hash_init_std:(?P<std>[-+0-9.]+)$",
    re.MULTILINE,
)
LOGICAL_BLOCK_ORDER_RE = re.compile(r"^logical_block_order:(?P<order>\S+)$", re.MULTILINE)


@dataclass(frozen=True)
class EvalConfig:
    name: str
    checkpoint: str
    data_path: str
    tokenizer_path: str
    vocab_size: int
    num_layers: int
    num_unique_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: int
    train_seq_len: int
    qk_gain_init: float
    rope_base: float
    logit_softcap: float
    python_exe: str
    train_script: str
    counted_code_path: str
    train_batch_tokens: int
    val_batch_size: int
    val_max_tokens: int
    lr_schedule: str
    lr_warmup_iters: int
    min_lr_scale: float
    tied_embed_lr: float
    matrix_lr: float
    scalar_lr: float
    int4_block_size: int
    logs_dir: str
    reference_policy_name: str


@dataclass(frozen=True)
class ExportPolicy:
    name: str
    quant_format: str
    int4_groups: tuple[str, ...]
    fp16_groups: tuple[str, ...]
    notes: str
    codebook_name: str = "normal16"
    lowrank_groups: tuple[str, ...] = ()
    lowrank_rank: int = 0
    direct_int4_patterns: tuple[str, ...] = ()
    direct_fp16_patterns: tuple[str, ...] = ()
    direct_lowrank_patterns: tuple[str, ...] = ()


@dataclass
class PolicyResult:
    name: str
    quant_format: str
    codebook_name: str
    int4_groups: list[str]
    fp16_groups: list[str]
    lowrank_groups: list[str]
    lowrank_rank: int
    int4_patterns: list[str]
    fp16_patterns: list[str]
    lowrank_patterns: list[str]
    log_path: str
    compressed_bytes: int
    counted_code_bytes: int
    total_bytes: int
    clean_val_loss: float
    clean_val_bpb: float
    shipped_val_loss: float
    shipped_val_bpb: float
    export_gap_bpb: float
    legal: bool
    score: float
    notes: str


PRESETS = {
    "local_u4k_12x608_lfqat60_m4": EvalConfig(
        name="local_u4k_12x608_lfqat60_m4",
        checkpoint="./logs/dense_u4k_12x608_kv2_lfqat60_fcproj_mlx_model.npz",
        data_path="./data/local_u4k_unigram/datasets/fineweb10B_spu4096_local",
        tokenizer_path="./data/local_u4k_unigram/tokenizers/fineweb_4096_unigram.model",
        vocab_size=4096,
        num_layers=12,
        num_unique_layers=12,
        model_dim=608,
        num_heads=8,
        num_kv_heads=2,
        mlp_mult=2,
        train_seq_len=1024,
        qk_gain_init=1.45,
        rope_base=10000.0,
        logit_softcap=30.0,
        python_exe="./.venv/bin/python",
        train_script="train_gpt_mlx.py",
        counted_code_path="train_gpt.py,quant_reconstruction.py",
        train_batch_tokens=8192,
        val_batch_size=131072,
        val_max_tokens=262144,
        lr_schedule="constant",
        lr_warmup_iters=0,
        min_lr_scale=1.0,
        tied_embed_lr=0.004,
        matrix_lr=0.003,
        scalar_lr=0.003,
        int4_block_size=64,
        logs_dir="logs/export_gap_local_u4k_12x608_lfqat60_seq1024",
        reference_policy_name="fcproj_hi",
    ),
    "local_u5k_12x608_seq896_exactcopy_m4": EvalConfig(
        name="local_u5k_12x608_seq896_exactcopy_m4",
        checkpoint="./logs/lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy_mlx_model.npz",
        data_path="./data/local_u5k_unigram/datasets/fineweb10B_spu5120_local",
        tokenizer_path="./data/local_u5k_unigram/tokenizers/fineweb_5120_unigram.model",
        vocab_size=5120,
        num_layers=12,
        num_unique_layers=12,
        model_dim=608,
        num_heads=8,
        num_kv_heads=2,
        mlp_mult=2,
        train_seq_len=896,
        qk_gain_init=1.45,
        rope_base=10000.0,
        logit_softcap=30.0,
        python_exe="./.venv/bin/python",
        train_script="train_gpt_mlx.py",
        counted_code_path="train_gpt.py,quant_reconstruction.py",
        train_batch_tokens=8192,
        val_batch_size=131072,
        val_max_tokens=262144,
        lr_schedule="constant",
        lr_warmup_iters=0,
        min_lr_scale=1.0,
        tied_embed_lr=0.0012,
        matrix_lr=0.0008,
        scalar_lr=0.0008,
        int4_block_size=64,
        logs_dir="logs/export_gap_local_u5k_12x608_seq896",
        reference_policy_name="fcproj_hi",
    ),
    "local_sp1024_10x560_m4": EvalConfig(
        name="local_sp1024_10x560_m4",
        checkpoint="./logs/local_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_compiled_longtail_fchi_top4_attn_top4_m4_mlx_model.npz",
        data_path="./data/datasets/fineweb10B_sp1024",
        tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
        vocab_size=1024,
        num_layers=10,
        num_unique_layers=10,
        model_dim=560,
        num_heads=8,
        num_kv_heads=2,
        mlp_mult=2,
        train_seq_len=896,
        qk_gain_init=1.45,
        rope_base=10000.0,
        logit_softcap=28.8,
        python_exe="./.venv/bin/python",
        train_script="train_gpt_mlx.py",
        counted_code_path="train_gpt.py,quant_reconstruction.py",
        train_batch_tokens=16384,
        val_batch_size=32768,
        val_max_tokens=262144,
        lr_schedule="cosine",
        lr_warmup_iters=24,
        min_lr_scale=0.14,
        tied_embed_lr=0.00195,
        matrix_lr=0.0013,
        scalar_lr=0.0013,
        int4_block_size=64,
        logs_dir="logs/export_gap_local_sp1024_10x560_seq896",
        reference_policy_name="fchi_top4_attn_top4_fp16",
    ),
    "local_sp1024_12x544_share6_m4": EvalConfig(
        name="local_sp1024_12x544_share6_m4",
        checkpoint="./logs/lab_sp1024_12x544_kv2_share6_896ctx_modern_m4_mlx_model.npz",
        data_path="./data/datasets/fineweb10B_sp1024",
        tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
        vocab_size=1024,
        num_layers=12,
        num_unique_layers=6,
        model_dim=544,
        num_heads=8,
        num_kv_heads=2,
        mlp_mult=2,
        train_seq_len=896,
        qk_gain_init=1.45,
        rope_base=10000.0,
        logit_softcap=28.5,
        python_exe="./.venv/bin/python",
        train_script="train_gpt_mlx.py",
        counted_code_path="train_gpt.py,quant_reconstruction.py",
        train_batch_tokens=16384,
        val_batch_size=32768,
        val_max_tokens=262144,
        lr_schedule="cosine",
        lr_warmup_iters=24,
        min_lr_scale=0.14,
        tied_embed_lr=0.0020,
        matrix_lr=0.00135,
        scalar_lr=0.00135,
        int4_block_size=64,
        logs_dir="logs/export_gap_local_sp1024_12x544_share6_seq896",
        reference_policy_name="proj_top6_attn_top6_fp16",
    ),
    "local_sp1024_12x560_share6_m4": EvalConfig(
        name="local_sp1024_12x560_share6_m4",
        checkpoint="./logs/lab_sp1024_12x560_kv2_share6_896ctx_modern_m4_mlx_model.npz",
        data_path="./data/datasets/fineweb10B_sp1024",
        tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
        vocab_size=1024,
        num_layers=12,
        num_unique_layers=6,
        model_dim=560,
        num_heads=8,
        num_kv_heads=2,
        mlp_mult=2,
        train_seq_len=896,
        qk_gain_init=1.45,
        rope_base=10000.0,
        logit_softcap=28.5,
        python_exe="./.venv/bin/python",
        train_script="train_gpt_mlx.py",
        counted_code_path="train_gpt.py,quant_reconstruction.py",
        train_batch_tokens=16384,
        val_batch_size=32768,
        val_max_tokens=262144,
        lr_schedule="cosine",
        lr_warmup_iters=24,
        min_lr_scale=0.14,
        tied_embed_lr=0.0020,
        matrix_lr=0.00135,
        scalar_lr=0.00135,
        int4_block_size=64,
        logs_dir="logs/export_gap_local_sp1024_12x560_share6_seq896",
        reference_policy_name="proj_top6_attn_top6_fp16",
    ),
    "local_sp1024_12x576_share6_m4": EvalConfig(
        name="local_sp1024_12x576_share6_m4",
        checkpoint="./logs/lab_sp1024_12x576_kv2_share6_codebook_qer192_tail20_lowlr_m4_mlx_model.npz",
        data_path="./data/datasets/fineweb10B_sp1024",
        tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
        vocab_size=1024,
        num_layers=12,
        num_unique_layers=6,
        model_dim=576,
        num_heads=8,
        num_kv_heads=2,
        mlp_mult=2,
        train_seq_len=896,
        qk_gain_init=1.45,
        rope_base=10000.0,
        logit_softcap=28.5,
        python_exe="./.venv/bin/python",
        train_script="train_gpt_mlx.py",
        counted_code_path="train_gpt.py,quant_reconstruction.py",
        train_batch_tokens=16384,
        val_batch_size=32768,
        val_max_tokens=262144,
        lr_schedule="cosine",
        lr_warmup_iters=24,
        min_lr_scale=0.14,
        tied_embed_lr=0.0020,
        matrix_lr=0.00135,
        scalar_lr=0.00135,
        int4_block_size=64,
        logs_dir="logs/export_gap_local_sp1024_12x576_share6_seq896",
        reference_policy_name="proj_top6_attn_top6_fp16",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search export policies for a fixed checkpoint and model settings.")
    p.add_argument("--preset", default="local_sp1024_10x560_m4", choices=sorted(PRESETS))
    p.add_argument("--checkpoint")
    p.add_argument("--data-path")
    p.add_argument("--tokenizer-path")
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--num-unique-layers", type=int)
    p.add_argument("--model-dim", type=int)
    p.add_argument("--num-heads", type=int)
    p.add_argument("--num-kv-heads", type=int)
    p.add_argument("--mlp-mult", type=int)
    p.add_argument("--train-seq-len", type=int)
    p.add_argument("--qk-gain-init", type=float)
    p.add_argument("--rope-base", type=float)
    p.add_argument("--logit-softcap", type=float)
    p.add_argument("--python-exe")
    p.add_argument("--train-script")
    p.add_argument("--counted-code-path")
    p.add_argument("--train-batch-tokens", type=int)
    p.add_argument("--val-batch-size", type=int)
    p.add_argument("--val-max-tokens", type=int)
    p.add_argument("--lr-schedule")
    p.add_argument("--lr-warmup-iters", type=int)
    p.add_argument("--min-lr-scale", type=float)
    p.add_argument("--tied-embed-lr", type=float)
    p.add_argument("--matrix-lr", type=float)
    p.add_argument("--scalar-lr", type=float)
    p.add_argument("--int4-block-size", type=int)
    p.add_argument("--logs-dir")
    p.add_argument("--legal-budget-bytes", type=int, default=16_000_000)
    p.add_argument("--gap-weight", type=float, default=0.35)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument(
        "--policy-name",
        action="append",
        dest="policy_names",
        help="Restrict evaluation to one or more named export policies.",
    )
    p.add_argument("--out-json", type=Path)
    p.add_argument("--out-md", type=Path)
    return p.parse_args()


def merge_config(args: argparse.Namespace) -> EvalConfig:
    base = PRESETS[args.preset]
    data = asdict(base)
    for field in data:
        cli_name = field.replace("_", "-")
        attr = cli_name.replace("-", "_")
        value = getattr(args, attr, None)
        if value is not None:
            data[field] = value
    if args.checkpoint and args.checkpoint != base.checkpoint:
        stem = Path(args.checkpoint).name
        if stem.endswith("_mlx_model.npz"):
            stem = stem[: -len("_mlx_model.npz")]
        else:
            stem = Path(stem).stem
        data["name"] = sanitize_slug(stem)
    return EvalConfig(**data)


def sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def checkpoint_fingerprint(checkpoint: str) -> str:
    path = (ROOT / checkpoint).resolve()
    stat = path.stat()
    payload = f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def checkpoint_log_path(checkpoint: str) -> Path:
    path = (ROOT / checkpoint).resolve()
    if path.name.endswith("_mlx_model.npz"):
        return path.with_name(path.name.replace("_mlx_model.npz", ".txt"))
    return path.with_suffix(".txt")


def parse_checkpoint_runtime_flags(checkpoint: str) -> dict[str, str]:
    log_path = checkpoint_log_path(checkpoint)
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    flags: dict[str, str] = {}
    global_bus_match = GLOBAL_BUS_LINE_RE.search(text)
    if global_bus_match and global_bus_match.group("enabled") == "True":
        flags["GLOBAL_BUS_ENABLED"] = "1"
        flags["GLOBAL_BUS_READ_LAYERS"] = global_bus_match.group("read")
        flags["GLOBAL_BUS_WRITE_LAYERS"] = global_bus_match.group("write")
        flags["GLOBAL_BUS_SUMMARY_MODE"] = global_bus_match.group("mode")
        flags["GLOBAL_BUS_TAIL_WEIGHT"] = global_bus_match.group("tail")
        if global_bus_match.group("allow") == "True":
            flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    second_pass_match = SECOND_PASS_LINE_RE.search(text)
    if second_pass_match and second_pass_match.group("enabled") == "True":
        flags["SECOND_PASS_ENABLED"] = "1"
        flags["SECOND_PASS_LAYERS"] = second_pass_match.group("layers")
        flags["SECOND_PASS_GATE_INIT"] = second_pass_match.group("gate")
        flags["SECOND_PASS_SKIP_INIT"] = second_pass_match.group("skip")
        flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    cross_skip_match = CROSS_SKIP_ROUTER_LINE_RE.search(text)
    if cross_skip_match and cross_skip_match.group("enabled") == "True":
        flags["CROSS_SKIP_ROUTER_ENABLED"] = "1"
        flags["CROSS_SKIP_ROUTER_MATCH_INIT"] = cross_skip_match.group("match")
        flags["CROSS_SKIP_ROUTER_OTHER_INIT"] = cross_skip_match.group("other")
        if cross_skip_match.group("decoder"):
            flags["CROSS_SKIP_ROUTER_DECODER_LAYERS"] = cross_skip_match.group("decoder")
        if cross_skip_match.group("source"):
            flags["CROSS_SKIP_ROUTER_SOURCE_LAYERS"] = cross_skip_match.group("source")
        flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    pair_features_match = PAIR_FEATURES_LINE_RE.search(text)
    if pair_features_match and pair_features_match.group("smear") == "True":
        flags["SMEAR_GATE_ENABLED"] = "1"
        flags["SMEAR_GATE_INIT"] = pair_features_match.group("smear_init")
        flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    if pair_features_match and pair_features_match.group("bigram") == "True":
        flags["BIGRAM_HASH_ENABLED"] = "1"
        flags["BIGRAM_HASH_BUCKETS"] = pair_features_match.group("buckets")
        flags["BIGRAM_HASH_DIM"] = pair_features_match.group("dim")
        flags["BIGRAM_HASH_INIT_STD"] = pair_features_match.group("std")
        flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    logical_order_match = LOGICAL_BLOCK_ORDER_RE.search(text)
    if logical_order_match:
        order = logical_order_match.group("order")
        if order != "modulo":
            flags["LOGICAL_BLOCK_ORDER"] = order
    return flags


@lru_cache(maxsize=None)
def checkpoint_feature_flags(checkpoint: str) -> dict[str, str]:
    path = (ROOT / checkpoint).resolve()
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            bigram_shape = (
                tuple(int(dim) for dim in data["bigram_hash_emb.weight"].shape)
                if "bigram_hash_emb.weight" in data.files
                else None
            )
    except (FileNotFoundError, ValueError, OSError):
        return {}
    flags: dict[str, str] = {}
    global_bus_keys = {"global_bus_read_scales", "global_bus_write_scales", "global_bus_decay_logits"}
    if global_bus_keys.issubset(keys):
        flags["GLOBAL_BUS_ENABLED"] = "1"
    second_pass_keys = {"second_pass_gates", "second_pass_skip_scales"}
    if second_pass_keys.issubset(keys):
        flags["SECOND_PASS_ENABLED"] = "1"
        flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    if "cross_skip_router_logits" in keys:
        flags["CROSS_SKIP_ROUTER_ENABLED"] = "1"
        flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    if "smear_gate" in keys:
        flags["SMEAR_GATE_ENABLED"] = "1"
        flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    if bigram_shape is not None and "bigram_hash_proj.weight" in keys:
        flags["BIGRAM_HASH_ENABLED"] = "1"
        flags["BIGRAM_HASH_BUCKETS"] = str(bigram_shape[0])
        flags["BIGRAM_HASH_DIM"] = str(bigram_shape[1])
        flags["ALLOW_INIT_MISSING_KEYS"] = "1"
    flags.update(parse_checkpoint_runtime_flags(checkpoint))
    return flags


def policy_space(num_layers: int) -> list[ExportPolicy]:
    top4 = min(4, num_layers)
    top5 = min(5, num_layers)
    top6 = min(6, num_layers)
    top4_start = max(num_layers - top4, 0)
    policies = [
        ExportPolicy("int8_all", "int8_clean_per_row_v1", tuple(), tuple(), "Pure int8 control."),
        ExportPolicy("int8_tok_fp16", "int8_clean_per_row_v1", tuple(), ("tok",), "Buy back embedding quality."),
        ExportPolicy(
            "fc_top1_int4",
            "mixed_int4_int8_packed_v2",
            ("fc_top1",),
            tuple(),
            "Sparse legalizer: only the single highest-layer mlp.fc tensor goes to int4.",
        ),
        ExportPolicy(
            "fc_top2_int4",
            "mixed_int4_int8_packed_v2",
            ("fc_top2",),
            tuple(),
            "Sparse legalizer: only the top-2 mlp.fc tensors go to int4.",
        ),
        ExportPolicy(
            "fc_top3_int4",
            "mixed_int4_int8_packed_v2",
            ("fc_top3",),
            tuple(),
            "Boundary-only compression: use int4 on just the top-3 mlp.fc tensors.",
        ),
        ExportPolicy(
            "fc_top4_int4",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            tuple(),
            "Near-cap legalizer: use int4 on the top-4 mlp.fc tensors while keeping the rest at int8.",
        ),
        ExportPolicy("fchi_only", "mixed_int4_int8_packed_v2", ("fc_hi",), tuple(), "Upper mlp.fc only int4."),
        ExportPolicy("fchi_attnhi_fp16", "mixed_int4_int8_packed_v2", ("fc_hi",), ("attn_hi",), "Current strongest family for preserving attention output."),
        ExportPolicy("fcproj_hi", "mixed_int4_int8_packed_v2", ("fc_hi", "proj_hi"), tuple(), "Upper fc+proj int4."),
        ExportPolicy("fcproj_attnhi_fp16", "mixed_int4_int8_packed_v2", ("fc_hi", "proj_hi"), ("attn_hi",), "Hybrid export with upper attn.proj in fp16."),
        ExportPolicy(f"fchi_top{top4}_attn_top{top4}_fp16", "mixed_int4_int8_packed_v2", (f"fc_top{top4}",), (f"attn_top{top4}",), "Top-N fc int4 + top-N attn fp16."),
        ExportPolicy(f"fcproj_top{top5}_attn_top{top5}_fp16", "mixed_int4_int8_packed_v2", (f"fc_top{top5}", f"proj_top{top5}"), (f"attn_top{top5}",), "Top-N fc/proj int4 + top-N attn fp16."),
        ExportPolicy(
            "fc_top4_attn_top1_fp16",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("attn_top1",),
            "Shrink the expensive upper-attention fp16 corridor to only the single highest-layer attn.proj tensor.",
        ),
        ExportPolicy(
            "fc_top4_proj_top1_fp16",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("proj_top1",),
            "Shrink the expensive upper-proj fp16 corridor to only the single highest-layer mlp.proj tensor.",
        ),
        ExportPolicy(
            "fc_top4_proj_top1_attn_top1_fp16",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("proj_top1", "attn_top1"),
            "Legalized mini-hybrid: keep only the very top proj+attn pair in fp16 while preserving the winning fc_top4 int4 corridor.",
        ),
        ExportPolicy(
            "fc_top4_attn_top1_qer_proj_top1_r32",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("attn_top1",),
            "Near-cap legalizer: keep the top attention boundary tensor in fp16 and reconstruct the single top MLP projection with a small low-rank residual.",
            lowrank_groups=("proj_top1",),
            lowrank_rank=32,
        ),
        ExportPolicy(
            "fc_top4_attn_top1_qer_proj_top1_r64",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("attn_top1",),
            "Near-cap legalizer: same boundary mix, but with a medium-rank residual on the top MLP projection.",
            lowrank_groups=("proj_top1",),
            lowrank_rank=64,
        ),
        ExportPolicy(
            "fc_top4_attn_top1_qer_proj_top1_r96",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("attn_top1",),
            "Near-cap legalizer: higher-rank residual on the top MLP projection to recover more of the illegal proj+attn fp16 gain while staying legal.",
            lowrank_groups=("proj_top1",),
            lowrank_rank=96,
        ),
        ExportPolicy(
            "fc_top4_attn_top2_fp16",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("attn_top2",),
            "Moderate mini-hybrid: keep only the top-2 attention projection tensors in fp16 and preserve the rest of the sparse winner.",
        ),
        ExportPolicy(
            "fc_top4_proj_top2_fp16",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("proj_top2",),
            "Moderate mini-hybrid: keep only the top-2 MLP projection tensors in fp16 on top of the sparse winner.",
        ),
        ExportPolicy(
            "fc_top4_proj_top2_attn_top2_fp16",
            "mixed_int4_int8_packed_v2",
            ("fc_top4",),
            ("proj_top2", "attn_top2"),
            "Near-cap mini-hybrid: keep only the top-2 proj+attn pairs in fp16 while holding the rest of the checkpoint on the winning sparse exporter.",
        ),
        ExportPolicy("projhi_attnhi_fp16", "int8_clean_per_row_v1", tuple(), ("proj_hi", "attn_hi"), "Conservative fp16 keep-float on upper projections."),
        ExportPolicy(
            f"proj_top{top5 + 1}_attn_top{top5}_fp16",
            "int8_clean_per_row_v1",
            tuple(),
            (f"proj_top{top5 + 1}", f"attn_top{top5}"),
            "Near-cap extension: add one more upper mlp.proj block to the current best conservative policy.",
        ),
        ExportPolicy(
            f"proj_top{top5}_attn_top{top5 + 1}_fp16",
            "int8_clean_per_row_v1",
            tuple(),
            (f"proj_top{top5}", f"attn_top{top5 + 1}"),
            "Near-cap extension: add one more upper attn.proj block to the current best conservative policy.",
        ),
        ExportPolicy(
            f"proj_top{top5 + 1}_attn_top{top5 + 1}_fp16",
            "int8_clean_per_row_v1",
            tuple(),
            (f"proj_top{top5 + 1}", f"attn_top{top5 + 1}"),
            "Aggressive near-cap extension: add one more upper proj+attn block pair in fp16.",
        ),
        ExportPolicy("tok_attnhi_fp16", "int8_clean_per_row_v1", tuple(), ("tok", "attn_hi"), "Embedding plus upper attention recovery."),
        ExportPolicy(
            f"qer_proj_top{top6}_r32",
            "int8_clean_per_row_v1",
            tuple(),
            tuple(),
            "Diagnostic branch: only upper mlp.proj gets residual correction.",
            lowrank_groups=(f"proj_top{top6}",),
            lowrank_rank=32,
        ),
        ExportPolicy(
            f"qer_attn_top{top6}_r32",
            "int8_clean_per_row_v1",
            tuple(),
            tuple(),
            "Diagnostic branch: only upper attn.proj gets residual correction.",
            lowrank_groups=(f"attn_top{top6}",),
            lowrank_rank=32,
        ),
        ExportPolicy(
            f"qer_projattn_top{top6}_r320_attn_top1_fp16",
            "int8_clean_per_row_v1",
            tuple(),
            ("attn_top1",),
            "Near-cap hybrid: keep the single top boundary attention projection in fp16 and repair the rest with rank-320 QER.",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=320,
        ),
        ExportPolicy(
            f"qer_projattn_top{top6}_r320_proj_top1_fp16",
            "int8_clean_per_row_v1",
            tuple(),
            ("proj_top1",),
            "Near-cap hybrid: keep the single top boundary MLP projection in fp16 and repair the rest with rank-320 QER.",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=320,
        ),
        ExportPolicy(
            f"qer_projattn_top{top6}_r288_attn_top1_fp16",
            "int8_clean_per_row_v1",
            tuple(),
            ("attn_top1",),
            "Byte-balanced hybrid: slightly lower-rank QER plus one fp16 attention boundary tensor.",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=288,
        ),
        ExportPolicy(
            f"qer_projattn_top{top6}_r288_proj_top1_fp16",
            "int8_clean_per_row_v1",
            tuple(),
            ("proj_top1",),
            "Byte-balanced hybrid: slightly lower-rank QER plus one fp16 MLP projection boundary tensor.",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=288,
        ),
        ExportPolicy(
            f"codebook_projattn_top{top6}",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Orthogonal exporter: nonuniform codebook int4 on upper proj+attn without residual repair.",
        ),
        ExportPolicy(
            f"codebook_qer_projattn_top{top6}_r64",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Nonuniform 4-bit base plus low-rank residual repair on the same upper proj+attn corridor.",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=64,
        ),
        ExportPolicy(
            f"codebook_qer_projattn_top{top6}_r128",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Higher-capacity nonuniform 4-bit base plus residual repair.",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=128,
        ),
        ExportPolicy(
            f"codebook_qer_projattn_top{top6}_r192",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Aggressive nonuniform 4-bit base plus residual repair, still meant to stay comfortably legal.",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=192,
        ),
        ExportPolicy(
            f"quantile_codebook_projattn_top{top6}",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Tensor-adaptive nonuniform 4-bit exporter using an empirical 16-level codebook per tensor.",
            codebook_name="quantile16",
        ),
        ExportPolicy(
            f"quantile_codebook_qer_projattn_top{top6}_r128",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Tensor-adaptive codebook base plus moderate residual repair.",
            codebook_name="quantile16",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=128,
        ),
        ExportPolicy(
            f"quantile_codebook_qer_projattn_top{top6}_r160",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Tensor-adaptive codebook base plus a slightly wider residual branch.",
            codebook_name="quantile16",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=160,
        ),
        ExportPolicy(
            f"quantile_codebook_qer_projattn_top{top6}_r192",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Tensor-adaptive codebook base plus aggressive residual repair.",
            codebook_name="quantile16",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=192,
        ),
        ExportPolicy(
            f"quantile_codebook_qer_projattn_top{top6}_r224",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Near-cap tensor-adaptive codebook base plus very high-rank residual repair.",
            codebook_name="quantile16",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=224,
        ),
        ExportPolicy(
            f"lloyd_codebook_projattn_top{top6}",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Learned 16-level Lloyd-Max codebook per tensor on upper proj+attn without residual repair.",
            codebook_name="lloyd16",
        ),
        ExportPolicy(
            f"lloyd_codebook_qer_projattn_top{top6}_r192",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Learned Lloyd-Max codebook base plus aggressive residual repair on the active upper corridor.",
            codebook_name="lloyd16",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=192,
        ),
        ExportPolicy(
            f"lloyd_codebook_qer_projattn_top{top6}_r224",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "Near-cap Lloyd-Max codebook base plus higher-rank residual repair.",
            codebook_name="lloyd16",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=224,
        ),
        ExportPolicy(
            f"mulaw_codebook_qer_projattn_top{top6}_r192",
            "mixed_codebook_int4_int8_packed_v1",
            (f"proj_top{top6}", f"attn_top{top6}"),
            tuple(),
            "mu-law companded 16-level codebook plus aggressive residual repair on the active upper corridor.",
            codebook_name="mulaw16",
            lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
            lowrank_rank=192,
        ),
        ExportPolicy(
            "proj_top5_attn_top6_fc5_int4_fp16",
            "mixed_int4_int8_packed_v2",
            tuple(),
            tuple(),
            "Legalize the slightly-over-cap `proj_top5_attn_top6` corridor by paying with one upper-boundary fc int4 tensor.",
            direct_int4_patterns=("blocks.5.mlp.fc.weight",),
            direct_fp16_patterns=(
                "blocks.4.attn.proj.weight",
                "blocks.5.attn.proj.weight",
                "blocks.6.attn.proj.weight",
                "blocks.7.attn.proj.weight",
                "blocks.8.attn.proj.weight",
                "blocks.9.attn.proj.weight",
                "blocks.5.mlp.proj.weight",
                "blocks.6.mlp.proj.weight",
                "blocks.7.mlp.proj.weight",
                "blocks.8.mlp.proj.weight",
                "blocks.9.mlp.proj.weight",
            ),
        ),
        ExportPolicy(
            "proj_top5_attn_top6_fc9_int4_fp16",
            "mixed_int4_int8_packed_v2",
            tuple(),
            tuple(),
            "Legalize the slightly-over-cap `proj_top5_attn_top6` corridor by paying with one highest-layer fc int4 tensor.",
            direct_int4_patterns=("blocks.9.mlp.fc.weight",),
            direct_fp16_patterns=(
                "blocks.4.attn.proj.weight",
                "blocks.5.attn.proj.weight",
                "blocks.6.attn.proj.weight",
                "blocks.7.attn.proj.weight",
                "blocks.8.attn.proj.weight",
                "blocks.9.attn.proj.weight",
                "blocks.5.mlp.proj.weight",
                "blocks.6.mlp.proj.weight",
                "blocks.7.mlp.proj.weight",
                "blocks.8.mlp.proj.weight",
                "blocks.9.mlp.proj.weight",
            ),
        ),
        ExportPolicy(
            "proj_top5_attn_top6_fc5_fc9_int4_fp16",
            "mixed_int4_int8_packed_v2",
            tuple(),
            tuple(),
            "More conservative legality hedge for the same near-cap corridor using two sparse fc int4 tensors.",
            direct_int4_patterns=("blocks.5.mlp.fc.weight", "blocks.9.mlp.fc.weight"),
            direct_fp16_patterns=(
                "blocks.4.attn.proj.weight",
                "blocks.5.attn.proj.weight",
                "blocks.6.attn.proj.weight",
                "blocks.7.attn.proj.weight",
                "blocks.8.attn.proj.weight",
                "blocks.9.attn.proj.weight",
                "blocks.5.mlp.proj.weight",
                "blocks.6.mlp.proj.weight",
                "blocks.7.mlp.proj.weight",
                "blocks.8.mlp.proj.weight",
                "blocks.9.mlp.proj.weight",
            ),
        ),
    ]
    for extra_block in range(top4_start):
        policies.append(
            ExportPolicy(
                f"fc_top4_plus_block{extra_block}_int4",
                "mixed_int4_int8_packed_v2",
                ("fc_top4",),
                tuple(),
                "Sparse boundary extension: keep the winning top-4 `mlp.fc` int4 corridor and add one extra measured block below it.",
                direct_int4_patterns=(f"blocks.{extra_block}.mlp.fc.weight",),
            )
        )
    qer_notes = {
        8: "Int8-all base with low-rank residual correction on upper proj+attn tensors.",
        16: "Int8-all base with medium-rank residual correction on upper proj+attn tensors.",
        32: "Int8-all base with stronger residual correction on upper proj+attn tensors.",
        64: "High-capacity residual correction branch for near-lossless projection recovery.",
        128: "Extended residual branch once the rank-64 curve proves monotonic but still leaves headroom.",
        192: "High-rank residual branch that spends more bytes to chase fp16-level recovery while staying legal.",
        256: "Aggressive residual branch for the same near-lossless corridor with ample legal headroom.",
        320: "Near-cap residual branch intended to test whether low-rank reconstruction can replace full fp16 keeps.",
    }
    for rank, note in qer_notes.items():
        policies.append(
            ExportPolicy(
                f"qer_projattn_top{top6}_r{rank}",
                "int8_clean_per_row_v1",
                tuple(),
                tuple(),
                note,
                lowrank_groups=(f"proj_top{top6}", f"attn_top{top6}"),
                lowrank_rank=rank,
            )
        )
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    unique: list[ExportPolicy] = []
    for policy in policies:
        key = (
            policy.quant_format,
            policy.codebook_name,
            policy.int4_groups,
            policy.fp16_groups,
            policy.lowrank_groups,
            (policy.lowrank_rank,),
            policy.direct_int4_patterns,
            policy.direct_fp16_patterns,
            policy.direct_lowrank_patterns,
        )
        if key not in seen:
            seen.add(key)
            unique.append(policy)
    return unique


def build_policy_patterns(policy: ExportPolicy, num_block_layers: int) -> tuple[list[str], list[str], list[str]]:
    int4_patterns: list[str] = []
    for group in policy.int4_groups:
        int4_patterns.extend(build_group_patterns(num_block_layers, (group,)))
    int4_patterns.extend(policy.direct_int4_patterns)
    fp16_patterns = list(build_fp16_patterns(policy.fp16_groups, num_block_layers))
    fp16_patterns.extend(policy.direct_fp16_patterns)
    lowrank_patterns = list(build_group_patterns(num_block_layers, policy.lowrank_groups))
    lowrank_patterns.extend(policy.direct_lowrank_patterns)
    return sorted(set(int4_patterns)), sorted(set(fp16_patterns)), sorted(set(lowrank_patterns))


def parse_log_metrics(log_path: Path) -> tuple[float, float, float, float, int]:
    text = log_path.read_text(encoding="utf-8")
    clean_matches = CLEAN_RE.findall(text)
    roundtrip_matches = ROUNDTRIP_RE.findall(text)
    byte_matches = BYTES_RE.findall(text)
    if not clean_matches or not roundtrip_matches or not byte_matches:
        raise ValueError(f"missing metrics in {log_path}")
    clean_loss, clean_bpb = clean_matches[-1]
    shipped_loss, shipped_bpb = roundtrip_matches[-1]
    return float(clean_loss), float(clean_bpb), float(shipped_loss), float(shipped_bpb), int(byte_matches[-1])


def build_eval_env(
    cfg: EvalConfig,
    policy: ExportPolicy,
    int4_patterns: list[str],
    fp16_patterns: list[str],
    lowrank_patterns: list[str],
    run_id: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_id,
            "DATA_PATH": cfg.data_path,
            "TOKENIZER_PATH": cfg.tokenizer_path,
            "VOCAB_SIZE": str(cfg.vocab_size),
            "TRAIN_BATCH_TOKENS": str(cfg.train_batch_tokens),
            "VAL_BATCH_SIZE": str(cfg.val_batch_size),
            "VAL_MAX_TOKENS": str(cfg.val_max_tokens),
            "VAL_LOSS_EVERY": "0",
            "TRAIN_LOG_EVERY": "0",
            "MAX_WALLCLOCK_SECONDS": "0",
            "ITERATIONS": "0",
            "WARMUP_STEPS": "0",
            "NUM_LAYERS": str(cfg.num_layers),
            "NUM_UNIQUE_LAYERS": str(cfg.num_unique_layers),
            "MODEL_DIM": str(cfg.model_dim),
            "NUM_HEADS": str(cfg.num_heads),
            "NUM_KV_HEADS": str(cfg.num_kv_heads),
            "MLP_MULT": str(cfg.mlp_mult),
            "TRAIN_SEQ_LEN": str(cfg.train_seq_len),
            "QK_GAIN_INIT": str(cfg.qk_gain_init),
            "ROPE_BASE": str(cfg.rope_base),
            "LOGIT_SOFTCAP": str(cfg.logit_softcap),
            "LR_SCHEDULE": cfg.lr_schedule,
            "LR_WARMUP_ITERS": str(cfg.lr_warmup_iters),
            "MIN_LR_SCALE": str(cfg.min_lr_scale),
            "TIED_EMBED_LR": str(cfg.tied_embed_lr),
            "MATRIX_LR": str(cfg.matrix_lr),
            "SCALAR_LR": str(cfg.scalar_lr),
            "INIT_MODEL_PATH": cfg.checkpoint,
            "OUT_DIR": cfg.logs_dir,
            "QUANT_FORMAT": policy.quant_format,
            "INT4_CODEBOOK_NAME": policy.codebook_name,
            "INT4_NAME_PATTERNS": ",".join(int4_patterns),
            "INT4_BLOCK_SIZE": str(cfg.int4_block_size),
            "TRAIN_QAT_NAME_PATTERNS": "",
            "TRAIN_COMPRESSION_AWARE_NAME_PATTERNS": "",
            "INT8_KEEP_FLOAT_FP16_NAME_PATTERNS": ",".join(fp16_patterns),
            "LOWRANK_ERROR_NAME_PATTERNS": ",".join(lowrank_patterns),
            "LOWRANK_ERROR_RANK": str(policy.lowrank_rank),
        }
    )
    env.update(checkpoint_feature_flags(cfg.checkpoint))
    return env


def run_policy_eval(
    cfg: EvalConfig,
    policy: ExportPolicy,
    legal_budget_bytes: int,
    gap_weight: float,
) -> PolicyResult:
    logs_dir = (ROOT / cfg.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, cfg.num_unique_layers)
    run_id = f"{cfg.name}__{checkpoint_fingerprint(cfg.checkpoint)}__{sanitize_slug(policy.name)}"
    log_path = logs_dir / f"{run_id}.txt"

    def execute_eval() -> None:
        env = build_eval_env(cfg, policy, int4_patterns, fp16_patterns, lowrank_patterns, run_id)
        cmd = [cfg.python_exe, cfg.train_script]
        stdout_log_path = log_path.with_suffix(".stdout.tmp")
        log_path.unlink(missing_ok=True)
        stdout_log_path.unlink(missing_ok=True)
        with stdout_log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"policy eval failed for {policy.name}: see {stdout_log_path}")
        try:
            parse_log_metrics(log_path)
        except (ValueError, FileNotFoundError):
            if stdout_log_path.exists():
                try:
                    parse_log_metrics(stdout_log_path)
                except ValueError as exc:
                    raise RuntimeError(
                        f"policy eval produced no parseable metrics for {policy.name}: expected {log_path}"
                    ) from exc
                stdout_log_path.replace(log_path)
            else:
                raise RuntimeError(f"policy eval produced no log for {policy.name}: expected {log_path}")
        else:
            stdout_log_path.unlink(missing_ok=True)

    if not log_path.exists():
        execute_eval()
    else:
        try:
            parse_log_metrics(log_path)
        except ValueError:
            # Partial or corrupted cached logs should not poison future sweeps.
            log_path.unlink(missing_ok=True)
            execute_eval()

    clean_loss, clean_bpb, shipped_loss, shipped_bpb, compressed_bytes = parse_log_metrics(log_path)
    counted_code_bytes = sum((ROOT / rel.strip()).stat().st_size for rel in cfg.counted_code_path.split(",") if rel.strip())
    total_bytes = compressed_bytes + counted_code_bytes
    export_gap_bpb = shipped_bpb - clean_bpb
    over_bytes = max(total_bytes - legal_budget_bytes, 0)
    score = shipped_bpb + gap_weight * export_gap_bpb + (10.0 * over_bytes / max(legal_budget_bytes, 1))
    return PolicyResult(
        name=policy.name,
        quant_format=policy.quant_format,
        codebook_name=policy.codebook_name,
        int4_groups=list(policy.int4_groups),
        fp16_groups=list(policy.fp16_groups),
        lowrank_groups=list(policy.lowrank_groups),
        lowrank_rank=policy.lowrank_rank,
        int4_patterns=int4_patterns,
        fp16_patterns=fp16_patterns,
        lowrank_patterns=lowrank_patterns,
        log_path=str(log_path),
        compressed_bytes=compressed_bytes,
        counted_code_bytes=counted_code_bytes,
        total_bytes=total_bytes,
        clean_val_loss=clean_loss,
        clean_val_bpb=clean_bpb,
        shipped_val_loss=shipped_loss,
        shipped_val_bpb=shipped_bpb,
        export_gap_bpb=export_gap_bpb,
        legal=total_bytes <= legal_budget_bytes,
        score=score,
        notes=policy.notes,
    )


def sort_key(result: PolicyResult) -> tuple[float, float, int]:
    penalty = 0 if result.legal else 1
    return (penalty, result.score, result.total_bytes)


def select_policies(policies: list[ExportPolicy], policy_names: list[str] | None) -> list[ExportPolicy]:
    if not policy_names:
        return policies
    by_name = {policy.name: policy for policy in policies}
    missing = [name for name in policy_names if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise ValueError(f"unknown policy names: {', '.join(missing)}; available: {available}")
    ordered_unique_names = list(dict.fromkeys(policy_names))
    return [by_name[name] for name in ordered_unique_names]


def write_outputs(out_json: Path, out_md: Path, payload: dict[str, Any]) -> None:
    reference = payload.get("reference_policy")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Fixed Export Gap Search",
        "",
        f"- Preset: `{payload['config']['name']}`",
        f"- Checkpoint: `{payload['config']['checkpoint']}`",
        f"- Counted code path: `{payload['config']['counted_code_path']}`",
        f"- Legal budget: `{payload['legal_budget_bytes']}`",
        "",
        "## Best Legal Policy",
        "",
        f"- Name: `{payload['best_legal']['name']}`",
        f"- Clean `val_bpb`: `{payload['best_legal']['clean_val_bpb']:.8f}`",
        f"- Shipped `val_bpb`: `{payload['best_legal']['shipped_val_bpb']:.8f}`",
        f"- Export gap: `{payload['best_legal']['export_gap_bpb']:.8f}`",
        f"- Total bytes: `{payload['best_legal']['total_bytes']}`",
        f"- Int4 groups: `{','.join(payload['best_legal']['int4_groups']) or '-'}`",
        f"- FP16 groups: `{','.join(payload['best_legal']['fp16_groups']) or '-'}`",
        f"- Low-rank groups: `{','.join(payload['best_legal']['lowrank_groups']) or '-'}`",
        f"- Low-rank rank: `{payload['best_legal']['lowrank_rank']}`",
        "",
    ]
    if reference is not None:
        lines.extend(
            [
                "## Reference Policy Comparison",
                "",
                f"- Reference policy: `{reference['name']}`",
                f"- Reference shipped `val_bpb`: `{reference['shipped_val_bpb']:.8f}`",
                f"- Reference export gap: `{reference['export_gap_bpb']:.8f}`",
                f"- Shipped delta vs best: `{reference['shipped_val_bpb'] - payload['best_legal']['shipped_val_bpb']:.8f}`",
                f"- Gap delta vs best: `{reference['export_gap_bpb'] - payload['best_legal']['export_gap_bpb']:.8f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Policies",
            "",
            "| Name | Shipped bpb | Gap | Total bytes | Legal |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["top_results"]:
        lines.append(
            f"| `{row['name']}` | `{row['shipped_val_bpb']:.8f}` | `{row['export_gap_bpb']:.8f}` | `{row['total_bytes']}` | `{row['legal']}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)
    checkpoint = (ROOT / cfg.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    policies = select_policies(policy_space(cfg.num_unique_layers), args.policy_names)
    results = [
        run_policy_eval(cfg, policy, args.legal_budget_bytes, args.gap_weight)
        for policy in policies
    ]
    results.sort(key=sort_key)
    best_legal = next((r for r in results if r.legal), results[0])
    reference_policy = next((r for r in results if r.name == cfg.reference_policy_name), None)
    out_json = args.out_json or DEFAULT_OUT_DIR / f"{sanitize_slug(cfg.name)}_export_gap.json"
    out_md = args.out_md or DEFAULT_OUT_DIR / f"{sanitize_slug(cfg.name)}_export_gap.md"
    payload = {
        "config": asdict(cfg),
        "legal_budget_bytes": args.legal_budget_bytes,
        "best_legal": asdict(best_legal),
        "reference_policy": asdict(reference_policy) if reference_policy is not None else None,
        "top_results": [asdict(result) for result in results[: args.top_k]],
    }
    write_outputs(out_json, out_md, payload)
    print(out_json)
    print(out_md)
    print(json.dumps(asdict(best_legal), indent=2))


if __name__ == "__main__":
    main()
