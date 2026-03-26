from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from research.plan_sota_search import parse_experiment_table
from research.sota_seeker import search_seeker


SAMPLE_TABLE = textwrap.dedent(
    """
    # Experiment Table

    | Run | Hypothesis | Config | Runtime | Artifact bytes | Val loss / val_bpb | Conclusion |
    | --- | --- | --- | --- | --- | --- | --- |
    | `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` | Official 8xH100 anchor for the seeker winner. | Official `fineweb10B_sp1024`, `8xH100`, `10x512 KV2`, `seq=896`, `train_batch_tokens=573,440`, `GRAD_ACCUM_STEPS=1`, Muon `0.97`, `MAX_WALLCLOCK_SECONDS=600`, export=`int8_attnhi`. | `428.9s` train + `4.0s` roundtrip eval | `12,126,772 total` | `2.3365 / 1.3838` clean at stop, `2.5505 / 1.5106` post-quant | First official 8xH100 seeker run. |
    | `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` | Official anchor. | Official `fineweb10B_spu4096_docs`, `1xH100`, `12x608 KV2`, legal export=`fc_hi + proj_hi`. | `4801.1s` train + `38.6s` roundtrip eval | `10,842,824 total` | `3.0683 / 1.3329` clean at stop, `3.7062 / 1.6101` post-quant | Strongest official anchor. |
    | `dense_u4k_14x576_kv2_boot_from_12x576_adapt150` | Near-cap scale-up. | Expand `12x576` -> `14x576`, legal export=`INT4(blocks.7-13.mlp.fc.weight + blocks.7-13.mlp.proj.weight)`. | `522.3s` train + `54.7s` roundtrip eval | `15,180,590 total` | `4.5541 / 1.9862` post-quant | Near-tied local scale-up. |
    | `baseline_public_10min` | Public baseline defines the target. | `sp1024`, `layers=9`, `unique_layers=9`, `dim=512`, `kv=4`, `mlp=2`, tied embeddings. | `600038ms` on `8xH100` | `15,863,489 total` | `2.0727 / 1.2244` post-quant | Trustworthy leaderboard anchor. |
    | `baseline_public_4h` | Longer training improves full-precision loss but exposes a large post-quant penalty. | Same architecture as public baseline, 4-hour wallclock. | `14400039ms` on `8xH100` | `15,810,161 total` | `2.0386 / 1.2074` post-quant | Quantization penalty is a first-class bottleneck. |
    """
)


class SotaSeekerTests(unittest.TestCase):
    def _rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.md"
            path.write_text(SAMPLE_TABLE, encoding="utf-8")
            return parse_experiment_table(path)

    def test_search_includes_both_tokenizer_families(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        families = {candidate.tokenizer for candidate in candidates}
        self.assertIn("sp1024_bpe", families)
        self.assertIn("u4k_unigram", families)

    def test_best_official_candidate_prefers_sp1024_branch(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        official = [candidate for candidate in candidates if candidate.official_eligible]
        self.assertTrue(official)
        self.assertEqual(official[0].tokenizer, "sp1024_bpe")

    def test_best_candidate_tracks_token_flow(self) -> None:
        candidate = search_seeker(self._rows(), 1.1)[0]
        self.assertNotEqual(candidate.token_flow, "")
        self.assertGreater(candidate.estimated_total_bytes, 0)

    def test_search_space_is_large(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        self.assertGreater(len(candidates), 1000)

    def test_single_gpu_screen_command_downshifts_grad_accum_for_large_batches(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        oversized = [candidate for candidate in candidates if "TRAIN_BATCH_TOKENS=573440" in candidate.screen_command]
        self.assertTrue(oversized)
        self.assertTrue(all("GRAD_ACCUM_STEPS=2" in candidate.screen_command for candidate in oversized))
        self.assertTrue(all("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in candidate.screen_command for candidate in oversized))

    def test_exact_8xh100_anchor_overrides_old_offline_estimate(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        exact = next(candidate for candidate in candidates if candidate.name == "seeker_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi")
        self.assertAlmostEqual(exact.estimated_clean_bpb, 1.3838, places=4)
        self.assertAlmostEqual(exact.estimated_shipped_bpb, 1.5106, places=4)
        self.assertEqual(exact.estimated_total_bytes, 12_126_772)

    def test_search_includes_fullbudget_training_recipe(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        names = {candidate.training for candidate in candidates if candidate.tokenizer == "sp1024_bpe"}
        self.assertIn("stable_muon97_fullbudget", names)
        self.assertIn("stable_muon97_compiled_fullbudget", names)
        self.assertIn("stable_muon97_compiled_throughput", names)
        self.assertIn("stable_muon97_compiled_sparseeval", names)
        self.assertIn("stable_muon97_compiled_maxtrain", names)
        self.assertIn("stable_muon97_compiled_longtail", names)

    def test_compiled_int8_recipe_emits_empty_qat_patterns(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        candidate = next(
            c
            for c in candidates
            if c.training == "stable_muon97_compiled_fullbudget"
            and c.export_policy == "int8_attnhi"
            and c.model == "sp_10x512_kv2"
            and c.token_flow == "context_896_dense"
        )
        self.assertIn("TRAIN_QAT_NAME_PATTERNS=", candidate.final_8xh100_command)
        self.assertNotIn("TRAIN_QAT_NAME_PATTERNS=__never__", candidate.final_8xh100_command)
        self.assertIn("LFQAT_KL_WEIGHT=0.0", candidate.final_8xh100_command)

    def test_search_space_contains_new_width_and_throughput_levers(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        model_names = {candidate.model for candidate in candidates if candidate.tokenizer == "sp1024_bpe"}
        flow_names = {candidate.token_flow for candidate in candidates if candidate.tokenizer == "sp1024_bpe"}
        self.assertIn("sp_10x528_kv2", model_names)
        self.assertIn("sp_10x560_kv2", model_names)
        self.assertIn("sp_10x592_kv2", model_names)
        self.assertIn("sp_12x528_kv2", model_names)
        self.assertIn("throughput_896k_896ctx", flow_names)
        self.assertIn("throughput_816k_896ctx", flow_names)
        self.assertIn("throughput_832k_896ctx", flow_names)
        self.assertIn("context_960_balanced", flow_names)

    def test_compiled_mixed_export_recipe_emits_empty_qat_patterns(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        candidate = next(
            c
            for c in candidates
            if c.training == "stable_muon97_compiled_throughput"
            and c.export_policy == "fcproj_attnhi_fp16"
            and c.model == "sp_10x544_kv2"
            and c.token_flow == "throughput_896k_896ctx"
        )
        self.assertIn("TRAIN_QAT_NAME_PATTERNS=", candidate.final_8xh100_command)
        self.assertNotIn("TRAIN_QAT_NAME_PATTERNS=__never__", candidate.final_8xh100_command)
        self.assertIn("TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=", candidate.final_8xh100_command)
        self.assertNotIn("TRAIN_COMPRESSION_AWARE_NAME_PATTERNS=blocks.", candidate.final_8xh100_command)

    def test_top4_export_policy_emits_only_last_four_block_patterns(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        candidate = next(
            c
            for c in candidates
            if c.export_policy == "fcproj_top4_attn_top4_fp16"
            and c.model == "sp_10x560_kv2"
            and c.training == "stable_muon97_compiled_throughput"
        )
        self.assertIn("blocks.6.mlp.fc.weight", candidate.final_8xh100_command)
        self.assertIn("blocks.9.attn.proj.weight", candidate.final_8xh100_command)
        self.assertNotIn("blocks.5.mlp.fc.weight", candidate.final_8xh100_command)
        self.assertNotIn("blocks.5.attn.proj.weight", candidate.final_8xh100_command)

    def test_top5_export_policy_emits_five_block_patterns(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        candidate = next(
            c
            for c in candidates
            if c.export_policy == "fcproj_top5_attn_top5_fp16"
            and c.model == "sp_10x544_kv2"
            and c.training == "stable_muon97_compiled_maxtrain"
        )
        self.assertIn("blocks.5.mlp.fc.weight", candidate.final_8xh100_command)
        self.assertIn("blocks.9.attn.proj.weight", candidate.final_8xh100_command)
        self.assertNotIn("blocks.4.mlp.fc.weight", candidate.final_8xh100_command)

    def test_projhi_attnhi_export_policy_uses_int8_plus_upper_fp16_only(self) -> None:
        candidates = search_seeker(self._rows(), 1.1)
        candidate = next(
            c
            for c in candidates
            if c.export_policy == "projhi_attnhi_fp16"
            and c.model == "sp_10x560_kv2"
            and c.token_flow == "throughput_816k_896ctx"
        )
        self.assertIn("QUANT_FORMAT=int8_clean_per_row_v1", candidate.final_8xh100_command)
        self.assertIn("TARGET_EXPORT_NAME_PATTERNS=", candidate.final_8xh100_command)
        self.assertIn("INT4_NAME_PATTERNS=", candidate.final_8xh100_command)
        self.assertIn("blocks.5.mlp.proj.weight", candidate.final_8xh100_command)
        self.assertIn("blocks.9.attn.proj.weight", candidate.final_8xh100_command)
        self.assertNotIn("blocks.6.mlp.fc.weight", candidate.final_8xh100_command)
        self.assertLessEqual(candidate.estimated_gap, 0.1005)
        self.assertGreaterEqual(candidate.estimated_total_bytes, 15_036_838)


if __name__ == "__main__":
    unittest.main()
