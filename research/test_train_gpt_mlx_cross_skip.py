from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import unittest

import numpy as np


class TrainGptMlxCrossSkipTests(unittest.TestCase):
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

    def test_cross_skip_router_shapes(self) -> None:
        saved = {key: os.environ.get(key) for key in ("CROSS_SKIP_ROUTER_ENABLED", "GLOBAL_BUS_ENABLED")}
        try:
            os.environ["CROSS_SKIP_ROUTER_ENABLED"] = "1"
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
            self.assertEqual(tuple(model.cross_skip_router_logits.shape), (6, 6))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)

    def test_cross_skip_router_diag_init_is_near_noop(self) -> None:
        saved = {
            key: os.environ.get(key)
            for key in (
                "CROSS_SKIP_ROUTER_ENABLED",
                "CROSS_SKIP_ROUTER_MATCH_INIT",
                "CROSS_SKIP_ROUTER_OTHER_INIT",
                "GLOBAL_BUS_ENABLED",
            )
        }
        try:
            os.environ["CROSS_SKIP_ROUTER_ENABLED"] = "1"
            os.environ["CROSS_SKIP_ROUTER_MATCH_INIT"] = "20.0"
            os.environ["CROSS_SKIP_ROUTER_OTHER_INIT"] = "-20.0"
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
            with_router = module._np_float32(model(input_ids))
            model.cross_skip_router_enabled = False
            without_router = module._np_float32(model(input_ids))
            self.assertTrue(np.allclose(with_router, without_router, atol=1e-5))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)

    def test_cross_skip_router_masks_can_select_sparse_decoder_and_source_sets(self) -> None:
        saved = {
            key: os.environ.get(key)
            for key in (
                "CROSS_SKIP_ROUTER_ENABLED",
                "CROSS_SKIP_ROUTER_DECODER_LAYERS",
                "CROSS_SKIP_ROUTER_SOURCE_LAYERS",
                "GLOBAL_BUS_ENABLED",
            )
        }
        try:
            os.environ["CROSS_SKIP_ROUTER_ENABLED"] = "1"
            os.environ["CROSS_SKIP_ROUTER_DECODER_LAYERS"] = "10,11"
            os.environ["CROSS_SKIP_ROUTER_SOURCE_LAYERS"] = "2:6"
            os.environ["GLOBAL_BUS_ENABLED"] = "0"
            module = self._reload_module()
            decoder_mask = module.cross_skip_decoder_mask(12, 6, module.CROSS_SKIP_ROUTER_DECODER_LAYERS_SPEC)
            source_mask = module.cross_skip_source_mask(12, 6, module.CROSS_SKIP_ROUTER_SOURCE_LAYERS_SPEC)
            self.assertEqual(
                decoder_mask,
                (False, False, False, False, False, False, False, False, False, False, True, True),
            )
            self.assertEqual(
                source_mask,
                (False, False, True, True, True, True, False, False, False, False, False, False),
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)


if __name__ == "__main__":
    unittest.main()
