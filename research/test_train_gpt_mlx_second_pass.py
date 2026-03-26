from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import unittest

import numpy as np


class TrainGptMlxSecondPassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [
            name
            for name in ("sentencepiece", "mlx", "mlx.core")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise unittest.SkipTest(f"train_gpt_mlx dependencies missing: {', '.join(missing)}")

    def _reload_module(self):
        sys.modules.pop("train_gpt_mlx", None)
        return importlib.import_module("train_gpt_mlx")

    def test_second_pass_mask_targets_selected_decoder_layers(self) -> None:
        saved = {
            key: os.environ.get(key)
            for key in ("SECOND_PASS_ENABLED", "SECOND_PASS_LAYERS", "GLOBAL_BUS_ENABLED")
        }
        try:
            os.environ["SECOND_PASS_ENABLED"] = "1"
            os.environ["SECOND_PASS_LAYERS"] = "8,9"
            os.environ["GLOBAL_BUS_ENABLED"] = "0"
            module = self._reload_module()
            model = module.GPT(
                vocab_size=32,
                num_layers=12,
                num_unique_layers=10,
                dim=32,
                num_heads=4,
                num_kv_heads=2,
                mlp_mult=2,
                logit_chunk_tokens=0,
                logit_softcap=30.0,
                rope_base=10000.0,
                tied_embed_init_std=0.005,
                qk_gain_init=1.0,
            )
            self.assertEqual(
                module.second_pass_decoder_mask(model.num_layers, model.num_encoder_layers, module.SECOND_PASS_LAYERS_SPEC),
                (False, False, False, False, False, False, False, False, True, True, False, False),
            )
            self.assertEqual(tuple(model.second_pass_gates.shape), (6, 32))
            self.assertEqual(tuple(model.second_pass_skip_scales.shape), (6, 32))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)

    def test_second_pass_zero_gate_is_exact_noop(self) -> None:
        saved = {
            key: os.environ.get(key)
            for key in (
                "SECOND_PASS_ENABLED",
                "SECOND_PASS_LAYERS",
                "SECOND_PASS_GATE_INIT",
                "SECOND_PASS_SKIP_INIT",
                "GLOBAL_BUS_ENABLED",
            )
        }
        try:
            os.environ["SECOND_PASS_ENABLED"] = "1"
            os.environ["SECOND_PASS_LAYERS"] = "8,9"
            os.environ["SECOND_PASS_GATE_INIT"] = "0.0"
            os.environ["SECOND_PASS_SKIP_INIT"] = "0.05"
            os.environ["GLOBAL_BUS_ENABLED"] = "0"
            module = self._reload_module()
            module.mx.random.seed(0)
            model = module.GPT(
                vocab_size=32,
                num_layers=12,
                num_unique_layers=10,
                dim=32,
                num_heads=4,
                num_kv_heads=2,
                mlp_mult=2,
                logit_chunk_tokens=0,
                logit_softcap=30.0,
                rope_base=10000.0,
                tied_embed_init_std=0.005,
                qk_gain_init=1.0,
            )
            input_ids = module.mx.array(np.random.default_rng(0).integers(0, 32, size=(2, 8)), dtype=module.mx.int32)
            with_second_pass = module._np_float32(model(input_ids))
            model.second_pass_enabled = False
            without_second_pass = module._np_float32(model(input_ids))
            self.assertTrue(np.allclose(with_second_pass, without_second_pass, atol=1e-6))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)


if __name__ == "__main__":
    unittest.main()
