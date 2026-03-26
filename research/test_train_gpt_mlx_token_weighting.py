from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import unittest

import numpy as np


class TrainGptMlxTokenWeightingTests(unittest.TestCase):
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

    def test_train_token_weights_matches_eval_byte_formula(self) -> None:
        saved = {k: os.environ.get(k) for k in ("TRAIN_TOKEN_WEIGHT_MODE", "TRAIN_TOKEN_WEIGHT_POWER")}
        try:
            os.environ["TRAIN_TOKEN_WEIGHT_MODE"] = "bytes"
            os.environ["TRAIN_TOKEN_WEIGHT_POWER"] = "1.0"
            module = self._reload_module()
            module.configure_train_token_weight_luts(
                np.array([1, 2, 3, 4], dtype=np.int16),
                np.array([0, 1, 0, 0], dtype=np.bool_),
                np.array([1, 0, 0, 1], dtype=np.bool_),
            )
            x = module.mx.array([[0, 0, 2]], dtype=module.mx.int32)
            y = module.mx.array([[1, 2, 3]], dtype=module.mx.int32)
            weights = module.train_token_weights(x, y)
            self.assertEqual(weights.tolist(), [2.0, 3.0, 4.0])
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)

    def test_train_token_weights_uniform_mode_disables_weighting(self) -> None:
        saved = {k: os.environ.get(k) for k in ("TRAIN_TOKEN_WEIGHT_MODE", "TRAIN_TOKEN_WEIGHT_POWER")}
        try:
            os.environ["TRAIN_TOKEN_WEIGHT_MODE"] = "uniform"
            module = self._reload_module()
            x = module.mx.array([[0, 1]], dtype=module.mx.int32)
            y = module.mx.array([[1, 2]], dtype=module.mx.int32)
            self.assertIsNone(module.train_token_weights(x, y))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)

    def test_cross_entropy_loss_respects_unit_weights(self) -> None:
        module = self._reload_module()
        logits = module.mx.array([[2.0, 0.0], [0.0, 2.0]], dtype=module.mx.float32)
        targets = module.mx.array([0, 1], dtype=module.mx.int32)
        weighted = module.cross_entropy_loss(logits, targets, module.mx.array([1.0, 1.0], dtype=module.mx.float32))
        unweighted = module.cross_entropy_loss(logits, targets, None)
        self.assertAlmostEqual(float(weighted.item()), float(unweighted.item()), places=6)


if __name__ == "__main__":
    unittest.main()
