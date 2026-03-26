from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from research.search_sota_trials import build_trial_command, search_trials
from research.plan_sota_search import POLICIES, SHAPE_SPECS, parse_experiment_table


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


class SearchSotaTrialsTests(unittest.TestCase):
    def _rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.md"
            path.write_text(SAMPLE_TABLE, encoding="utf-8")
            return parse_experiment_table(path)

    def test_search_trials_prefers_continue_export_first_path(self) -> None:
        trials = search_trials(self._rows(), 1.1)
        self.assertEqual((trials[0].layers, trials[0].model_dim), (12, 608))
        self.assertEqual(trials[0].init_mode, "continue")

    def test_moonshot_shape_is_present(self) -> None:
        trials = search_trials(self._rows(), 1.1)
        self.assertTrue(any((trial.layers, trial.model_dim) == (14, 576) for trial in trials))

    def test_best_trial_is_still_remote_from_11_target(self) -> None:
        trials = search_trials(self._rows(), 1.1)
        self.assertEqual(trials[0].target_band, "target-remote")
        self.assertGreater(trials[0].clean_gain_needed_after_best_export, 0.25)

    def test_build_trial_command_uses_empty_init_for_scratch(self) -> None:
        shape = next(spec for spec in SHAPE_SPECS if spec.key == (14, 576))
        policy = next(policy for policy in POLICIES if policy.name == "fc_hi_int4")
        from research.search_sota_trials import RECIPE_LIBRARY

        recipe = next(recipe for recipe in RECIPE_LIBRARY if recipe.name == "scratch_cosine_stable")
        cmd = build_trial_command("trial_14x576", shape, recipe, policy, nproc=8)
        self.assertIn("INIT_MODEL_PATH=", cmd)
        self.assertIn("NPROC_PER_NODE=8", cmd)
        self.assertIn("NUM_LAYERS=14", cmd)


if __name__ == "__main__":
    unittest.main()
