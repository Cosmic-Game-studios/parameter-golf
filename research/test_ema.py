from __future__ import annotations

import importlib
import os
import sys
import unittest

import torch


class ExponentialMovingAverageTests(unittest.TestCase):
    def test_hyperparameters_pick_up_ema_env(self) -> None:
        keys = ("EMA_DECAY", "EMA_START_STEP", "EMA_UPDATE_EVERY")
        saved_env = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["EMA_DECAY"] = "0.997"
            os.environ["EMA_START_STEP"] = "7"
            os.environ["EMA_UPDATE_EVERY"] = "3"
            sys.modules.pop("train_gpt", None)
            train_gpt = importlib.import_module("train_gpt")
            self.assertAlmostEqual(train_gpt.Hyperparameters.ema_decay, 0.997)
            self.assertEqual(train_gpt.Hyperparameters.ema_start_step, 7)
            self.assertEqual(train_gpt.Hyperparameters.ema_update_every, 3)
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt", None)

    def test_ema_copies_first_eligible_step_then_smooths(self) -> None:
        from train_gpt import ExponentialMovingAverage

        weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
        named_params = [("linear.weight", weight)]
        ema = ExponentialMovingAverage(named_params, decay=0.5, start_step=2, update_every=1)

        weight.data.fill_(3.0)
        ema.update(named_params, step=1)
        self.assertFalse(ema.active)
        self.assertTrue(torch.allclose(ema.shadow["linear.weight"], torch.tensor([[1.0, 2.0]])))

        ema.update(named_params, step=2)
        self.assertTrue(ema.active)
        self.assertTrue(torch.allclose(ema.shadow["linear.weight"], torch.full((1, 2), 3.0)))

        weight.data.fill_(5.0)
        ema.update(named_params, step=3)
        self.assertTrue(torch.allclose(ema.shadow["linear.weight"], torch.full((1, 2), 4.0)))

    def test_ema_materializes_cpu_state_dict(self) -> None:
        from train_gpt import ExponentialMovingAverage, clone_state_dict_cpu

        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        named_params = list(model.named_parameters())
        ema = ExponentialMovingAverage(named_params, decay=0.0)
        ema.update(named_params, step=0)
        with torch.no_grad():
            model.weight.fill_(9.0)
        base_state = clone_state_dict_cpu(model.state_dict())
        ema_state = ema.state_dict(base_state)
        self.assertTrue(torch.allclose(base_state["weight"], torch.full((2, 2), 9.0)))
        self.assertTrue(torch.allclose(ema_state["weight"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))


if __name__ == "__main__":
    unittest.main()
