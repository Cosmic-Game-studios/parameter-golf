from __future__ import annotations

import unittest
from pathlib import Path

from research.autoresearch_loop import ResultEntry, build_loop, load_json
from research.plan_sota_search import parse_experiment_table


ROOT = Path(__file__).resolve().parents[1]


class AutoresearchLoopTests(unittest.TestCase):
    def test_build_loop_emits_export_first_item(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")

        payload = build_loop(experiments, seeker, export_gap, [], mode="local-first")

        self.assertIn("loop", payload)
        self.assertGreaterEqual(len(payload["loop"]), 3)
        first = payload["loop"][0]
        self.assertEqual(first["name"], "sp1024_10x560_export_first")
        self.assertIn("projhi_attnhi_fp16", first["command"])
        self.assertIn("INT4_NAME_PATTERNS=", first["command"])
        self.assertIn("scripts/local/train_seeker_sp1024_best_m4.sh", first["command"])

    def test_loop_uses_measured_best_export(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")

        payload = build_loop(experiments, seeker, export_gap, [], mode="local-first")
        self.assertEqual(payload["best_measured_export_gap"]["name"], "projhi_attnhi_fp16")
        self.assertTrue(payload["best_measured_export_gap"]["legal"])

    def test_local_first_candidate_commands_use_local_launcher(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")

        payload = build_loop(experiments, seeker, export_gap, [], mode="local-first")
        second = payload["loop"][1]
        self.assertIn("scripts/local/train_seeker_sp1024_best_m4.sh", second["command"])
        self.assertIn("VAL_MAX_TOKENS=262144", second["command"])
        self.assertIn("MLX_MAX_MICROBATCH_TOKENS=8192", second["command"])

    def test_resolved_results_are_filtered_out_of_loop(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")
        results = [
            ResultEntry(
                name="sp1024_10x560_export_first",
                run_id="autoresearch_sp1024_10x560_projhi_attnhi_fp16",
                tier="local",
                status="kept",
                clean_bpb=2.5858,
                shipped_bpb=2.6799,
                export_gap_bpb=0.0941,
                note="shipping improved on same 10x560 path",
            ),
            ResultEntry(
                name="seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16",
                run_id="local_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16",
                tier="local",
                status="discarded",
                clean_bpb=2.5875,
                shipped_bpb=2.7738,
                export_gap_bpb=0.1863,
                note="worse shipped than 10x560 export-first",
            ),
        ]

        payload = build_loop(experiments, seeker, export_gap, results, mode="local-first")

        names = [item["name"] for item in payload["loop"]]
        self.assertNotIn("sp1024_10x560_export_first", names)
        self.assertNotIn(
            "seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16",
            names,
        )
        self.assertEqual(len(payload["resolved_results"]), 2)

    def test_unresolved_sibling_uses_export_only_eval_when_checkpoint_exists(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")
        results = [
            ResultEntry(
                name="seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16",
                run_id="local_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16",
                tier="local",
                status="discarded",
                clean_bpb=2.5875,
                shipped_bpb=2.7738,
                export_gap_bpb=0.1863,
                note="existing raw checkpoint",
            ),
        ]

        payload = build_loop(experiments, seeker, export_gap, results, mode="local-first")
        sibling = payload["loop"][1]
        self.assertIn("INIT_MODEL_PATH=", sibling["command"])
        self.assertIn("ITERATIONS=0", sibling["command"])
        self.assertIn("WARMUP_STEPS=0", sibling["command"])
        self.assertIn("_export_eval", sibling["command"])

    def test_local_anchor_prefers_measured_autoresearch_results(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")
        results = [
            ResultEntry(
                name="sp1024_10x560_export_first_continue120_microtail",
                run_id="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail",
                tier="local",
                status="kept",
                clean_bpb=2.1611,
                shipped_bpb=2.2821,
                export_gap_bpb=0.1210,
                note="current best local autoresearch anchor",
            ),
        ]

        payload = build_loop(experiments, seeker, export_gap, results, mode="local-first")

        self.assertEqual(payload["best_measured_local_result"]["run_id"], "autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail")
        self.assertEqual(payload["best_measured_local_result"]["shipped_bpb"], 2.2821)

    def test_loop_proposes_continuation_after_multiple_keeps(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")
        results = [
            ResultEntry(
                name="sp1024_10x560_export_first_continue120_microtail",
                run_id="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail",
                tier="local",
                status="kept",
                clean_bpb=2.1611,
                shipped_bpb=2.2821,
                export_gap_bpb=0.1210,
                note="fourth continuation tail keep",
            ),
        ]

        payload = build_loop(experiments, seeker, export_gap, results, mode="local-first")

        first = payload["loop"][0]
        self.assertEqual(first["source"], "measured_local_continuation")
        self.assertIn("INIT_MODEL_PATH=./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_mlx_model.npz", first["command"])
        self.assertIn("TIED_EMBED_LR=0.0005", first["command"])
        self.assertIn("MATRIX_LR=0.00035", first["command"])
        self.assertIn("continue120_microtail", first["command"])

    def test_kept_anchor_beats_nearly_tied_discarded_result(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")
        results = [
            ResultEntry(
                name="sp1024_10x560_export_first_continue120_microtail",
                run_id="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail",
                tier="local",
                status="kept",
                clean_bpb=2.1611,
                shipped_bpb=2.28212032,
                export_gap_bpb=0.12102032,
                note="kept anchor",
            ),
            ResultEntry(
                name="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail",
                run_id="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail",
                tier="local",
                status="discarded",
                clean_bpb=2.1267,
                shipped_bpb=2.28205890,
                export_gap_bpb=0.15535890,
                note="noise-level shipped gain but worse gap",
            ),
        ]

        payload = build_loop(experiments, seeker, export_gap, results, mode="local-first")

        self.assertEqual(
            payload["best_measured_local_result"]["run_id"],
            "autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail",
        )

    def test_gap_recovery_run_is_proposed_before_repeat_tail(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_continue120_microtail_m4_export_gap.json")
        results = [
            ResultEntry(
                name="sp1024_10x560_export_first_continue120_microtail",
                run_id="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail",
                tier="local",
                status="kept",
                clean_bpb=2.1611,
                shipped_bpb=2.28212032,
                export_gap_bpb=0.12102032,
                note="kept anchor",
            ),
            ResultEntry(
                name="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail",
                run_id="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail",
                tier="local",
                status="discarded",
                clean_bpb=2.1267,
                shipped_bpb=2.28205890,
                export_gap_bpb=0.15535890,
                note="noise-level shipped gain but worse gap",
            ),
        ]

        payload = build_loop(experiments, seeker, export_gap, results, mode="local-first")

        first = payload["loop"][0]
        self.assertEqual(first["source"], "measured_gap_recovery")
        self.assertIn(
            "INIT_MODEL_PATH=./logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_mlx_model.npz",
            first["command"],
        )
        self.assertIn("TRAIN_GRAD_ONLY_NAME_PATTERNS=blocks.5.attn.proj.weight", first["command"])
        self.assertIn("MATRIX_LR=0.0002", first["command"])

    def test_recovery_anchor_continues_with_keptfp16_only_tail(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_keptfp16_recovery120_m4_export_gap.json")
        results = [
            ResultEntry(
                name="sp1024_10x560_export_first_keptfp16_recovery120",
                run_id="autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120",
                tier="local",
                status="kept",
                clean_bpb=2.1175,
                shipped_bpb=2.27454972,
                export_gap_bpb=0.15704972,
                note="kept recovery anchor",
            ),
        ]

        payload = build_loop(experiments, seeker, export_gap, results, mode="local-first")

        first = payload["loop"][0]
        self.assertEqual(first["source"], "measured_gap_recovery_continuation")
        self.assertIn("continue80_keptfp16", first["command"])
        self.assertIn("TRAIN_GRAD_ONLY_NAME_PATTERNS=blocks.5.attn.proj.weight", first["command"])
        self.assertIn("MATRIX_LR=0.00015", first["command"])
        self.assertIn("VAL_LOSS_EVERY=80", first["command"])

    def test_recovery_continuation_stops_after_sub_threshold_gain(self) -> None:
        experiments = parse_experiment_table(ROOT / "research" / "experiment_table.md")
        seeker = load_json(ROOT / "research" / "sota_seeker.json")
        export_gap = load_json(ROOT / "research" / "local_sp1024_10x560_block4attn_m4_export_gap.json")
        results = [
            ResultEntry(
                name="sp1024_10x560_export_first_hybrid_int8aware_block4attn_keptfp16_80",
                run_id="autoresearch_sp1024_10x560_hybrid_int8aware_block4attn_keptfp16_80",
                tier="local",
                status="kept",
                clean_bpb=2.1093,
                shipped_bpb=2.27195757,
                export_gap_bpb=0.16265757,
                note="latest kept hybrid attn anchor with sub-threshold gain",
            ),
            ResultEntry(
                name="sp1024_10x560_export_first_hybrid_int8aware_block4proj_keptfp16_80",
                run_id="autoresearch_sp1024_10x560_hybrid_int8aware_block4proj_keptfp16_80",
                tier="local",
                status="kept",
                clean_bpb=2.1104,
                shipped_bpb=2.27221636,
                export_gap_bpb=0.16181636,
                note="previous kept hybrid proj anchor",
            ),
        ]

        payload = build_loop(experiments, seeker, export_gap, results, mode="local-first")
        names = [item["name"] for item in payload["loop"]]
        self.assertNotIn(
            "autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120_continue80_keptfp16_continue80_keptfp16",
            names,
        )


if __name__ == "__main__":
    unittest.main()
