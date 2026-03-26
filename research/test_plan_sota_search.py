from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from research.plan_sota_search import (
    SHAPE_SPECS,
    build_candidates,
    build_command,
    build_group_patterns,
    parse_experiment_table,
)


SAMPLE_TABLE = textwrap.dedent(
    """
    # Experiment Table

    | Run | Hypothesis | Config | Runtime | Artifact bytes | Val loss / val_bpb | Conclusion |
    | --- | --- | --- | --- | --- | --- | --- |
    | `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` | Official anchor. | Official `fineweb10B_spu4096_docs`, `1xH100`, `12x608 KV2`, legal export=`fc_hi + proj_hi`. | `4801.1s` train + `38.6s` roundtrip eval | `10,842,824 total` | `3.0683 / 1.3329` clean at stop, `3.7062 / 1.6101` post-quant | Strongest official anchor. |
    | `dense_u4k_14x576_kv2_boot_from_12x576_adapt150` | Near-cap scale-up. | Expand `12x576` -> `14x576`, legal export=`INT4(blocks.7-13.mlp.fc.weight + blocks.7-13.mlp.proj.weight)`. | `522.3s` train + `54.7s` roundtrip eval | `15,180,590 total` | `4.5541 / 1.9862` post-quant | Near-tied local scale-up. |
    | `dense_u4k_13x576_boot_continue60_fcproj_lr5` | Weaker shape. | Expand `12x576` -> `13x576`, legal export=`INT4(blocks.6-12.mlp.fc.weight + blocks.6-12.mlp.proj.weight)`. | `346.1s` train + `18.6s` roundtrip eval | `13,921,399 total` | `4.6304 / 2.0195` post-quant | Still weaker than 12x608. |
    | `baseline_public_10min` | Public baseline defines the target. | `sp1024`, `layers=9`, `unique_layers=9`, `dim=512`, `kv=4`, `mlp=2`, tied embeddings. | `600038ms` on `8xH100` | `15,863,489 total` | `2.0727 / 1.2244` post-quant | Trustworthy leaderboard anchor. |
    """
)


class PlanSotaSearchTests(unittest.TestCase):
    def test_parse_experiment_table_extracts_shape_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.md"
            path.write_text(SAMPLE_TABLE, encoding="utf-8")
            rows = parse_experiment_table(path)
        self.assertEqual(len(rows), 4)
        official = rows[0]
        self.assertEqual((official.layers, official.model_dim), (12, 608))
        self.assertAlmostEqual(official.clean_bpb or 0.0, 1.3329, places=4)
        self.assertAlmostEqual(official.shipped_bpb or 0.0, 1.6101, places=4)

    def test_candidate_ranking_prefers_official_12x608_over_weaker_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.md"
            path.write_text(SAMPLE_TABLE, encoding="utf-8")
            rows = parse_experiment_table(path)
        plans = build_candidates(rows, 1.1)
        top_shape = (plans[0].layers, plans[0].model_dim)
        self.assertEqual(top_shape, (12, 608))
        self.assertLess(plans[0].search_score, next(plan for plan in plans if (plan.layers, plan.model_dim) == (13, 576)).search_score)

    def test_scratch_command_keeps_empty_init_model_path(self) -> None:
        shape = next(spec for spec in SHAPE_SPECS if spec.key == (14, 576))
        plan_cmd = build_command("planner_dense_u4k_14x576_fcproj", shape, next(policy for policy in __import__("research.plan_sota_search", fromlist=["POLICIES"]).POLICIES if policy.name == "fcproj_hi_int4"))
        self.assertIn("INIT_MODEL_PATH=", plan_cmd)
        self.assertIn("NUM_LAYERS=14", plan_cmd)
        self.assertIn("MODEL_DIM=576", plan_cmd)

    def test_build_group_patterns_supports_top_suffixes(self) -> None:
        pats = build_group_patterns(10, ("fc_top4", "attn_top3"))
        self.assertEqual(
            pats,
            (
                "blocks.6.mlp.fc.weight",
                "blocks.7.mlp.fc.weight",
                "blocks.8.mlp.fc.weight",
                "blocks.9.mlp.fc.weight",
                "blocks.7.attn.proj.weight",
                "blocks.8.attn.proj.weight",
                "blocks.9.attn.proj.weight",
            ),
        )


if __name__ == "__main__":
    unittest.main()
