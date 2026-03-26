from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import unittest

import numpy as np


class TrainGptMlxPairFeatureTests(unittest.TestCase):
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

    def test_pair_features_are_initially_near_noop(self) -> None:
        saved = {
            key: os.environ.get(key)
            for key in (
                "SMEAR_GATE_ENABLED",
                "SMEAR_GATE_INIT",
                "BIGRAM_HASH_ENABLED",
                "BIGRAM_HASH_BUCKETS",
                "BIGRAM_HASH_DIM",
                "BIGRAM_HASH_INIT_STD",
                "GLOBAL_BUS_ENABLED",
                "CROSS_SKIP_ROUTER_ENABLED",
            )
        }
        try:
            os.environ["SMEAR_GATE_ENABLED"] = "1"
            os.environ["SMEAR_GATE_INIT"] = "0.0"
            os.environ["BIGRAM_HASH_ENABLED"] = "1"
            os.environ["BIGRAM_HASH_BUCKETS"] = "128"
            os.environ["BIGRAM_HASH_DIM"] = "16"
            os.environ["BIGRAM_HASH_INIT_STD"] = "0.005"
            os.environ["GLOBAL_BUS_ENABLED"] = "0"
            os.environ["CROSS_SKIP_ROUTER_ENABLED"] = "0"
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
            self.assertEqual(tuple(model.bigram_hash_emb.weight.shape), (128, 16))
            self.assertEqual(tuple(model.bigram_hash_proj.weight.shape), (32, 16))
            self.assertTrue(np.allclose(module._np_float32(model.bigram_hash_proj.weight), 0.0, atol=1e-8))

            input_ids = module.mx.array(np.random.default_rng(0).integers(0, 32, size=(2, 8)), dtype=module.mx.int32)
            with_pair_features = module._np_float32(model(input_ids))
            model.smear_gate_enabled = False
            model.bigram_hash_enabled = False
            without_pair_features = module._np_float32(model(input_ids))
            self.assertTrue(np.allclose(with_pair_features, without_pair_features, atol=1e-5))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)


if __name__ == "__main__":
    unittest.main()
