from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from research.search_export_gap import (
    CLEAN_RE,
    EvalConfig,
    PRESETS,
    ROUNDTRIP_RE,
    build_eval_env,
    build_policy_patterns,
    checkpoint_log_path,
    checkpoint_feature_flags,
    checkpoint_fingerprint,
    merge_config,
    parse_checkpoint_runtime_flags,
    parse_log_metrics,
    policy_space,
    run_policy_eval,
    select_policies,
    sanitize_slug,
)


class SearchExportGapTests(unittest.TestCase):
    def test_policy_space_includes_current_family(self) -> None:
        names = {policy.name for policy in policy_space(10)}
        self.assertIn("fc_top1_int4", names)
        self.assertIn("fc_top2_int4", names)
        self.assertIn("fc_top3_int4", names)
        self.assertIn("fc_top4_int4", names)
        self.assertIn("fc_top4_plus_block0_int4", names)
        self.assertIn("fc_top4_plus_block5_int4", names)
        self.assertIn("fchi_top4_attn_top4_fp16", names)
        self.assertIn("fc_top4_proj_top1_attn_top1_fp16", names)
        self.assertIn("fc_top4_attn_top1_qer_proj_top1_r64", names)
        self.assertIn("fc_top4_proj_top2_attn_top2_fp16", names)
        self.assertIn("fcproj_attnhi_fp16", names)
        self.assertIn("proj_top6_attn_top5_fp16", names)
        self.assertIn("proj_top5_attn_top6_fp16", names)
        self.assertIn("proj_top6_attn_top6_fp16", names)
        self.assertIn("qer_projattn_top6_r16", names)
        self.assertIn("qer_projattn_top6_r64", names)
        self.assertIn("qer_projattn_top6_r320_attn_top1_fp16", names)
        self.assertIn("codebook_projattn_top6", names)
        self.assertIn("codebook_qer_projattn_top6_r128", names)
        self.assertIn("lloyd_codebook_qer_projattn_top6_r192", names)
        self.assertIn("mulaw_codebook_qer_projattn_top6_r192", names)
        self.assertIn("quantile_codebook_projattn_top6", names)
        self.assertIn("quantile_codebook_qer_projattn_top6_r224", names)
        self.assertIn("proj_top5_attn_top6_fc5_int4_fp16", names)
        self.assertIn("proj_top5_attn_top6_fc9_int4_fp16", names)

    def test_build_policy_patterns_resolves_top_groups(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "fchi_top4_attn_top4_fp16")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertIn("blocks.6.mlp.fc.weight", int4_patterns)
        self.assertIn("blocks.9.attn.proj.weight", fp16_patterns)
        self.assertNotIn("blocks.5.mlp.fc.weight", int4_patterns)
        self.assertEqual(lowrank_patterns, [])

    def test_build_policy_patterns_supports_sparse_fc_legalizer(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "fc_top2_int4")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertEqual(int4_patterns, ["blocks.8.mlp.fc.weight", "blocks.9.mlp.fc.weight"])
        self.assertEqual(fp16_patterns, [])
        self.assertEqual(lowrank_patterns, [])

    def test_build_policy_patterns_supports_top4_fc_legalizer(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "fc_top4_int4")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertEqual(
            int4_patterns,
            [
                "blocks.6.mlp.fc.weight",
                "blocks.7.mlp.fc.weight",
                "blocks.8.mlp.fc.weight",
                "blocks.9.mlp.fc.weight",
            ],
        )
        self.assertEqual(fp16_patterns, [])
        self.assertEqual(lowrank_patterns, [])

    def test_build_policy_patterns_supports_top4_plus_one_sparse_block(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "fc_top4_plus_block5_int4")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertEqual(
            int4_patterns,
            [
                "blocks.5.mlp.fc.weight",
                "blocks.6.mlp.fc.weight",
                "blocks.7.mlp.fc.weight",
                "blocks.8.mlp.fc.weight",
                "blocks.9.mlp.fc.weight",
            ],
        )
        self.assertEqual(fp16_patterns, [])
        self.assertEqual(lowrank_patterns, [])

    def test_build_policy_patterns_supports_mini_fp16_hybrid(self) -> None:
        policy = next(p for p in policy_space(12) if p.name == "fc_top4_proj_top2_attn_top2_fp16")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 12)
        self.assertEqual(
            int4_patterns,
            [
                "blocks.10.mlp.fc.weight",
                "blocks.11.mlp.fc.weight",
                "blocks.8.mlp.fc.weight",
                "blocks.9.mlp.fc.weight",
            ],
        )
        self.assertIn("blocks.10.mlp.proj.weight", fp16_patterns)
        self.assertIn("blocks.11.mlp.proj.weight", fp16_patterns)
        self.assertIn("blocks.10.attn.proj.weight", fp16_patterns)
        self.assertIn("blocks.11.attn.proj.weight", fp16_patterns)
        self.assertEqual(lowrank_patterns, [])

    def test_build_policy_patterns_supports_sparse_qer_legalizer(self) -> None:
        policy = next(p for p in policy_space(12) if p.name == "fc_top4_attn_top1_qer_proj_top1_r64")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 12)
        self.assertEqual(
            int4_patterns,
            [
                "blocks.10.mlp.fc.weight",
                "blocks.11.mlp.fc.weight",
                "blocks.8.mlp.fc.weight",
                "blocks.9.mlp.fc.weight",
            ],
        )
        self.assertEqual(fp16_patterns, ["blocks.11.attn.proj.weight"])
        self.assertEqual(lowrank_patterns, ["blocks.11.mlp.proj.weight"])

    def test_build_policy_patterns_supports_near_cap_proj_extension(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "proj_top6_attn_top5_fp16")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertEqual(int4_patterns, [])
        self.assertIn("blocks.4.mlp.proj.weight", fp16_patterns)
        self.assertIn("blocks.5.attn.proj.weight", fp16_patterns)
        self.assertNotIn("blocks.4.attn.proj.weight", fp16_patterns)
        self.assertEqual(lowrank_patterns, [])

    def test_build_policy_patterns_supports_direct_boundary_legalizer(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "proj_top5_attn_top6_fc9_int4_fp16")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertEqual(int4_patterns, ["blocks.9.mlp.fc.weight"])
        self.assertIn("blocks.4.attn.proj.weight", fp16_patterns)
        self.assertIn("blocks.9.attn.proj.weight", fp16_patterns)
        self.assertIn("blocks.9.mlp.proj.weight", fp16_patterns)
        self.assertEqual(lowrank_patterns, [])

    def test_build_policy_patterns_supports_lowrank_groups(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "qer_projattn_top6_r16")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertEqual(int4_patterns, [])
        self.assertEqual(fp16_patterns, [])
        self.assertIn("blocks.4.attn.proj.weight", lowrank_patterns)
        self.assertIn("blocks.5.mlp.proj.weight", lowrank_patterns)
        self.assertIn("blocks.9.attn.proj.weight", lowrank_patterns)

    def test_build_policy_patterns_supports_qer_hybrid_fp16_boundary(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "qer_projattn_top6_r320_attn_top1_fp16")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertEqual(int4_patterns, [])
        self.assertEqual(fp16_patterns, ["blocks.9.attn.proj.weight"])
        self.assertIn("blocks.4.mlp.proj.weight", lowrank_patterns)
        self.assertIn("blocks.9.attn.proj.weight", lowrank_patterns)

    def test_build_policy_patterns_supports_codebook_qer_combo(self) -> None:
        policy = next(p for p in policy_space(10) if p.name == "codebook_qer_projattn_top6_r128")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 10)
        self.assertIn("blocks.4.attn.proj.weight", int4_patterns)
        self.assertIn("blocks.9.mlp.proj.weight", int4_patterns)
        self.assertEqual(fp16_patterns, [])
        self.assertEqual(int4_patterns, lowrank_patterns)

    def test_build_policy_patterns_supports_quantile_codebook_combo(self) -> None:
        policy = next(p for p in policy_space(6) if p.name == "quantile_codebook_qer_projattn_top6_r192")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, 6)
        self.assertEqual(fp16_patterns, [])
        self.assertIn("blocks.0.attn.proj.weight", int4_patterns)
        self.assertIn("blocks.5.mlp.proj.weight", int4_patterns)
        self.assertEqual(int4_patterns, lowrank_patterns)

    def test_select_policies_preserves_requested_order(self) -> None:
        policies = policy_space(6)
        selected = select_policies(policies, ["proj_top6_attn_top6_fp16", "int8_all"])
        self.assertEqual([policy.name for policy in selected], ["proj_top6_attn_top6_fp16", "int8_all"])

    def test_select_policies_rejects_unknown_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown policy names"):
            select_policies(policy_space(6), ["does_not_exist"])

    def test_parse_log_metrics_extracts_clean_and_roundtrip(self) -> None:
        sample = "\n".join(
            [
                "step:300/300 val_loss:4.3666 val_bpb:2.5858 train_time:792539ms step_avg:2641.80ms",
                "serialized_model_int8_zlib:7561794 bytes (payload:21225248 raw_pickle:21234283 payload_ratio:3.90x)",
                "final_int8_zlib_roundtrip_exact val_loss:4.69873142 val_bpb:2.78245296",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.txt"
            path.write_text(sample, encoding="utf-8")
            clean_loss, clean_bpb, shipped_loss, shipped_bpb, model_bytes = parse_log_metrics(path)
        self.assertAlmostEqual(clean_loss, 4.3666)
        self.assertAlmostEqual(clean_bpb, 2.5858)
        self.assertAlmostEqual(shipped_loss, 4.69873142)
        self.assertAlmostEqual(shipped_bpb, 2.78245296)
        self.assertEqual(model_bytes, 7561794)

    def test_parse_checkpoint_runtime_flags_supports_pair_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "sample_mlx_model.npz"
            checkpoint.write_bytes(b"npz")
            log = checkpoint_log_path(str(checkpoint))
            log.write_text(
                "\n".join(
                    [
                        "smear_gate_enabled:True smear_gate_init:0.000 bigram_hash_enabled:True bigram_hash_buckets:2048 bigram_hash_dim:64 bigram_hash_init_std:0.0050",
                        "cross_skip_router_enabled:True cross_skip_router_match_init:4.000 cross_skip_router_other_init:-4.000 cross_skip_router_decoder_layers:10,11 cross_skip_router_source_layers:0,1,2,3,4,5",
                    ]
                ),
                encoding="utf-8",
            )
            flags = parse_checkpoint_runtime_flags(str(checkpoint))
        self.assertEqual(flags["SMEAR_GATE_ENABLED"], "1")
        self.assertEqual(flags["SMEAR_GATE_INIT"], "0.000")
        self.assertEqual(flags["BIGRAM_HASH_ENABLED"], "1")
        self.assertEqual(flags["BIGRAM_HASH_BUCKETS"], "2048")
        self.assertEqual(flags["BIGRAM_HASH_DIM"], "64")
        self.assertEqual(flags["CROSS_SKIP_ROUTER_ENABLED"], "1")
        self.assertEqual(flags["CROSS_SKIP_ROUTER_DECODER_LAYERS"], "10,11")
        self.assertEqual(flags["CROSS_SKIP_ROUTER_SOURCE_LAYERS"], "0,1,2,3,4,5")

    def test_checkpoint_feature_flags_supports_pair_feature_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "sample_pairfeat_mlx_model.npz"
            np.savez(
                checkpoint,
                **{
                    "bigram_hash_emb.weight": np.zeros((512, 32), dtype=np.float32),
                    "bigram_hash_proj.weight": np.zeros((64, 32), dtype=np.float32),
                    "smear_gate": np.zeros((64,), dtype=np.float32),
                },
            )
            checkpoint_feature_flags.cache_clear()
            flags = checkpoint_feature_flags(str(checkpoint))
        self.assertEqual(flags["SMEAR_GATE_ENABLED"], "1")
        self.assertEqual(flags["BIGRAM_HASH_ENABLED"], "1")
        self.assertEqual(flags["BIGRAM_HASH_BUCKETS"], "512")
        self.assertEqual(flags["BIGRAM_HASH_DIM"], "32")

    def test_regexes_match_expected_lines(self) -> None:
        self.assertIsNotNone(CLEAN_RE.search("step:0/0 val_loss:6.9427 val_bpb:4.1113 train_time:10357ms"))
        self.assertIsNotNone(
            ROUNDTRIP_RE.search("final_int8_zlib_roundtrip_exact val_loss:4.69873142 val_bpb:2.78245296")
        )

    def test_eval_env_keeps_fixed_model_settings(self) -> None:
        cfg = PRESETS["local_sp1024_10x560_m4"]
        policy = next(p for p in policy_space(cfg.num_layers) if p.name == cfg.reference_policy_name)
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, cfg.num_layers)
        env = build_eval_env(cfg, policy, int4_patterns, fp16_patterns, lowrank_patterns, "demo_run")
        self.assertEqual(env["TRAIN_SEQ_LEN"], "896")
        self.assertEqual(env["QK_GAIN_INIT"], "1.45")
        self.assertEqual(env["ROPE_BASE"], "10000.0")
        self.assertEqual(env["LOGIT_SOFTCAP"], "28.8")
        self.assertEqual(env["TRAIN_QAT_NAME_PATTERNS"], "")
        self.assertIn("blocks.6.mlp.fc.weight", env["INT4_NAME_PATTERNS"])
        self.assertIn("blocks.9.attn.proj.weight", env["INT8_KEEP_FLOAT_FP16_NAME_PATTERNS"])
        self.assertEqual(env["LOWRANK_ERROR_NAME_PATTERNS"], "")
        self.assertEqual(env["LOWRANK_ERROR_RANK"], "0")

    def test_merge_config_sets_custom_name_from_checkpoint(self) -> None:
        ns = argparse.Namespace(
            preset="local_sp1024_10x560_m4",
            checkpoint="./logs/lab_u4k_custom_pairfeat_mlx_model.npz",
            data_path=None,
            tokenizer_path=None,
            vocab_size=None,
            num_layers=None,
            num_unique_layers=None,
            model_dim=None,
            num_heads=None,
            num_kv_heads=None,
            mlp_mult=None,
            train_seq_len=None,
            qk_gain_init=None,
            rope_base=None,
            logit_softcap=None,
            python_exe=None,
            train_script=None,
            counted_code_path=None,
            train_batch_tokens=None,
            val_batch_size=None,
            val_max_tokens=None,
            lr_schedule=None,
            lr_warmup_iters=None,
            min_lr_scale=None,
            tied_embed_lr=None,
            matrix_lr=None,
            scalar_lr=None,
            int4_block_size=None,
            logs_dir=None,
            legal_budget_bytes=16_000_000,
            gap_weight=0.35,
            top_k=12,
            policy_names=None,
            out_json=None,
            out_md=None,
        )
        cfg = merge_config(ns)
        self.assertEqual(cfg.name, "lab_u4k_custom_pairfeat")

    def test_u5k_exactcopy_preset_matches_alive_branch(self) -> None:
        cfg = PRESETS["local_u5k_12x608_seq896_exactcopy_m4"]
        policy = next(p for p in policy_space(cfg.num_unique_layers) if p.name == cfg.reference_policy_name)
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, cfg.num_unique_layers)
        env = build_eval_env(cfg, policy, int4_patterns, fp16_patterns, lowrank_patterns, "demo_run")
        self.assertEqual(env["VOCAB_SIZE"], "5120")
        self.assertEqual(env["TRAIN_SEQ_LEN"], "896")
        self.assertEqual(env["LR_SCHEDULE"], "constant")
        self.assertIn("blocks.6.mlp.fc.weight", env["INT4_NAME_PATTERNS"])
        self.assertIn("blocks.11.mlp.proj.weight", env["INT4_NAME_PATTERNS"])
        self.assertEqual(env["INT8_KEEP_FLOAT_FP16_NAME_PATTERNS"], "")

    def test_eval_env_passes_lowrank_settings(self) -> None:
        cfg = PRESETS["local_sp1024_12x576_share6_m4"]
        policy = next(p for p in policy_space(cfg.num_unique_layers) if p.name == "qer_projattn_top6_r16")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, cfg.num_unique_layers)
        env = build_eval_env(cfg, policy, int4_patterns, fp16_patterns, lowrank_patterns, "demo_run")
        self.assertEqual(env["LOWRANK_ERROR_RANK"], "16")
        self.assertIn("blocks.0.attn.proj.weight", env["LOWRANK_ERROR_NAME_PATTERNS"])
        self.assertIn("blocks.5.mlp.proj.weight", env["LOWRANK_ERROR_NAME_PATTERNS"])

    def test_eval_env_passes_codebook_quant_format(self) -> None:
        cfg = PRESETS["local_sp1024_12x576_share6_m4"]
        policy = next(p for p in policy_space(cfg.num_unique_layers) if p.name == "codebook_qer_projattn_top6_r64")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, cfg.num_unique_layers)
        env = build_eval_env(cfg, policy, int4_patterns, fp16_patterns, lowrank_patterns, "demo_run")
        self.assertEqual(env["QUANT_FORMAT"], "mixed_codebook_int4_int8_packed_v1")
        self.assertEqual(env["INT4_CODEBOOK_NAME"], "normal16")
        self.assertIn("blocks.0.attn.proj.weight", env["INT4_NAME_PATTERNS"])
        self.assertIn("blocks.5.mlp.proj.weight", env["LOWRANK_ERROR_NAME_PATTERNS"])

    def test_eval_env_passes_quantile_codebook_name(self) -> None:
        cfg = PRESETS["local_sp1024_12x576_share6_m4"]
        policy = next(p for p in policy_space(cfg.num_unique_layers) if p.name == "quantile_codebook_qer_projattn_top6_r160")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, cfg.num_unique_layers)
        env = build_eval_env(cfg, policy, int4_patterns, fp16_patterns, lowrank_patterns, "demo_run")
        self.assertEqual(env["QUANT_FORMAT"], "mixed_codebook_int4_int8_packed_v1")
        self.assertEqual(env["INT4_CODEBOOK_NAME"], "quantile16")
        self.assertEqual(env["LOWRANK_ERROR_RANK"], "160")

    def test_eval_env_passes_lloyd_codebook_name(self) -> None:
        cfg = PRESETS["local_sp1024_12x576_share6_m4"]
        policy = next(p for p in policy_space(cfg.num_unique_layers) if p.name == "lloyd_codebook_qer_projattn_top6_r192")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, cfg.num_unique_layers)
        env = build_eval_env(cfg, policy, int4_patterns, fp16_patterns, lowrank_patterns, "demo_run")
        self.assertEqual(env["QUANT_FORMAT"], "mixed_codebook_int4_int8_packed_v1")
        self.assertEqual(env["INT4_CODEBOOK_NAME"], "lloyd16")
        self.assertEqual(env["LOWRANK_ERROR_RANK"], "192")

    def test_eval_env_passes_mulaw_codebook_name(self) -> None:
        cfg = PRESETS["local_sp1024_12x576_share6_m4"]
        policy = next(p for p in policy_space(cfg.num_unique_layers) if p.name == "mulaw_codebook_qer_projattn_top6_r192")
        int4_patterns, fp16_patterns, lowrank_patterns = build_policy_patterns(policy, cfg.num_unique_layers)
        env = build_eval_env(cfg, policy, int4_patterns, fp16_patterns, lowrank_patterns, "demo_run")
        self.assertEqual(env["QUANT_FORMAT"], "mixed_codebook_int4_int8_packed_v1")
        self.assertEqual(env["INT4_CODEBOOK_NAME"], "mulaw16")
        self.assertEqual(env["LOWRANK_ERROR_RANK"], "192")

    def test_checkpoint_feature_flags_enable_global_bus(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "bus_checkpoint.npz"
            np.savez(
                ckpt,
                global_bus_read_scales=np.zeros((12, 16), dtype=np.float32),
                global_bus_write_scales=np.zeros((12, 16), dtype=np.float32),
                global_bus_decay_logits=np.zeros((12,), dtype=np.float32),
            )
            with patch("research.search_export_gap.ROOT", Path(td)):
                checkpoint_feature_flags.cache_clear()
                flags = checkpoint_feature_flags("./bus_checkpoint.npz")
                checkpoint_feature_flags.cache_clear()
        self.assertEqual(flags["GLOBAL_BUS_ENABLED"], "1")

    def test_checkpoint_feature_flags_enable_second_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "second_pass_checkpoint.npz"
            np.savez(
                ckpt,
                second_pass_gates=np.zeros((6, 16), dtype=np.float32),
                second_pass_skip_scales=np.zeros((6, 16), dtype=np.float32),
            )
            with patch("research.search_export_gap.ROOT", Path(td)):
                checkpoint_feature_flags.cache_clear()
                flags = checkpoint_feature_flags("./second_pass_checkpoint.npz")
                checkpoint_feature_flags.cache_clear()
        self.assertEqual(flags["SECOND_PASS_ENABLED"], "1")
        self.assertEqual(flags["ALLOW_INIT_MISSING_KEYS"], "1")

    def test_checkpoint_feature_flags_enable_cross_skip_router(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "cross_skip_checkpoint.npz"
            np.savez(
                ckpt,
                cross_skip_router_logits=np.zeros((6, 6), dtype=np.float32),
            )
            with patch("research.search_export_gap.ROOT", Path(td)):
                checkpoint_feature_flags.cache_clear()
                flags = checkpoint_feature_flags("./cross_skip_checkpoint.npz")
                checkpoint_feature_flags.cache_clear()
        self.assertEqual(flags["CROSS_SKIP_ROUTER_ENABLED"], "1")
        self.assertEqual(flags["ALLOW_INIT_MISSING_KEYS"], "1")

    def test_parse_checkpoint_runtime_flags_recovers_bus_and_second_pass_settings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "demo_mlx_model.npz"
            ckpt.write_bytes(b"x")
            log_path = checkpoint_log_path(str(ckpt))
            log_path.write_text(
                "\n".join(
                    [
                        "global_bus_enabled:True global_bus_read_layers:8,9,10,11 global_bus_write_layers:8,9,10,11 global_bus_summary_mode:mean_tail global_bus_tail_weight:0.650 allow_init_missing_keys:True",
                        "logical_block_order:0,1,2,3,4,5,6,7,8,9,8,9",
                        "second_pass_enabled:True second_pass_layers:8,9 second_pass_gate_init:0.000 second_pass_skip_init:0.050",
                        "cross_skip_router_enabled:True cross_skip_router_match_init:4.000 cross_skip_router_other_init:-4.000",
                    ]
                ),
                encoding="utf-8",
            )
            with patch("research.search_export_gap.ROOT", Path(td)):
                flags = parse_checkpoint_runtime_flags("./demo_mlx_model.npz")
        self.assertEqual(flags["GLOBAL_BUS_ENABLED"], "1")
        self.assertEqual(flags["GLOBAL_BUS_READ_LAYERS"], "8,9,10,11")
        self.assertEqual(flags["GLOBAL_BUS_SUMMARY_MODE"], "mean_tail")
        self.assertEqual(flags["LOGICAL_BLOCK_ORDER"], "0,1,2,3,4,5,6,7,8,9,8,9")
        self.assertEqual(flags["SECOND_PASS_ENABLED"], "1")
        self.assertEqual(flags["SECOND_PASS_LAYERS"], "8,9")
        self.assertEqual(flags["CROSS_SKIP_ROUTER_ENABLED"], "1")
        self.assertEqual(flags["CROSS_SKIP_ROUTER_MATCH_INIT"], "4.000")

    def test_checkpoint_fingerprint_changes_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "demo_model.npz"
            checkpoint.write_bytes(b"v1")
            fp1 = checkpoint_fingerprint(str(checkpoint))
            checkpoint.write_bytes(b"v2-with-different-size")
            fp2 = checkpoint_fingerprint(str(checkpoint))
        self.assertNotEqual(fp1, fp2)

    def test_run_policy_eval_recovers_from_partial_cached_log(self) -> None:
        sample = "\n".join(
            [
                "step:0/0 val_loss:3.6984 val_bpb:2.1901 train_time:329019ms step_avg:4112.74ms",
                "serialized_model_int8_zlib:14241788 bytes (payload:18595360 raw_pickle:18600316 payload_ratio:2.70x)",
                "final_int8_zlib_roundtrip_exact val_loss:3.70870399 val_bpb:2.19618733",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            checkpoint = tmpdir / "model.npz"
            checkpoint.write_bytes(b"checkpoint")
            code_path = tmpdir / "counted.py"
            code_path.write_text("print('x')\n", encoding="utf-8")
            logs_dir = tmpdir / "logs"
            cfg = EvalConfig(
                name="unit_test_gap",
                checkpoint=str(checkpoint),
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
                python_exe="python3",
                train_script="train_gpt_mlx.py",
                counted_code_path=str(code_path),
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
                logs_dir=str(logs_dir),
                reference_policy_name="proj_top6_attn_top6_fp16",
            )
            policy = next(p for p in policy_space(cfg.num_layers) if p.name == "proj_top6_attn_top6_fp16")
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"unit_test_gap__placeholder__{sanitize_slug(policy.name)}.txt"
            # Ensure the expected cached path exists but is incomplete.
            log_path.touch()

            with patch("research.search_export_gap.checkpoint_fingerprint", return_value="placeholder"):
                with patch("research.search_export_gap.subprocess.run") as mock_run:
                    def fake_run(*_args, **kwargs):
                        kwargs["stdout"].write(sample)
                        return SimpleNamespace(returncode=0)

                    mock_run.side_effect = fake_run
                    result = run_policy_eval(cfg, policy, 16_000_000, 0.35)

            self.assertEqual(mock_run.call_count, 1)
            self.assertAlmostEqual(result.clean_val_bpb, 2.1901)
            self.assertAlmostEqual(result.shipped_val_bpb, 2.19618733)
            self.assertEqual(result.compressed_bytes, 14241788)
            self.assertTrue(log_path.exists())
            self.assertEqual(log_path.read_text(encoding="utf-8"), sample)


if __name__ == "__main__":
    unittest.main()
