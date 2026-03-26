from __future__ import annotations

import importlib
import os
import sys
import unittest

import torch


class OptimizerWeightDecayTests(unittest.TestCase):
    def test_muon_applies_decoupled_weight_decay_without_gradients(self) -> None:
        from train_gpt import Muon

        param = torch.nn.Parameter(torch.tensor([[2.0, -4.0]], dtype=torch.float32))
        param.grad = torch.zeros_like(param)
        optimizer = Muon([param], lr=0.1, momentum=0.0, backend_steps=1, weight_decay=0.2)
        optimizer.step()
        self.assertTrue(torch.allclose(param.detach(), torch.tensor([[1.96, -3.92]]), atol=1e-5))

    def test_hyperparameters_default_decay_overrides(self) -> None:
        keys = ("WEIGHT_DECAY", "ADAM_WEIGHT_DECAY", "MUON_WEIGHT_DECAY")
        saved_env = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["WEIGHT_DECAY"] = "0.04"
            os.environ["ADAM_WEIGHT_DECAY"] = "0.03"
            os.environ.pop("MUON_WEIGHT_DECAY", None)
            sys.modules.pop("train_gpt", None)
            train_gpt = importlib.import_module("train_gpt")
            self.assertAlmostEqual(train_gpt.Hyperparameters.weight_decay, 0.04)
            self.assertAlmostEqual(train_gpt.Hyperparameters.adam_weight_decay, 0.03)
            self.assertAlmostEqual(train_gpt.Hyperparameters.muon_weight_decay, 0.04)
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt", None)


if __name__ == "__main__":
    unittest.main()
