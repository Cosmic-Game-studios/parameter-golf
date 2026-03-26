from __future__ import annotations

import importlib
import os
import sys
import unittest

import torch


class TrainGptCompileTests(unittest.TestCase):
    def test_single_rank_torchrun_env_does_not_enable_ddp(self) -> None:
        saved_env = {k: os.environ.get(k) for k in ("RANK", "WORLD_SIZE")}
        try:
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            sys.modules.pop("train_gpt", None)
            train_gpt = importlib.import_module("train_gpt")
            self.assertFalse(train_gpt.should_enable_ddp(1))
            self.assertTrue(train_gpt.should_enable_ddp(2))
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt", None)

    def test_fullgraph_compile_handles_rotary_cache_path(self) -> None:
        saved_env = {k: os.environ.get(k) for k in ("TRAIN_QAT_NAME_PATTERNS", "INT4_NAME_PATTERNS", "DISABLE_COMPILE", "TORCHDYNAMO_DISABLE")}
        try:
            os.environ["TRAIN_QAT_NAME_PATTERNS"] = "__never__"
            os.environ["INT4_NAME_PATTERNS"] = "__never__"
            os.environ.pop("DISABLE_COMPILE", None)
            os.environ.pop("TORCHDYNAMO_DISABLE", None)
            sys.modules.pop("train_gpt", None)
            train_gpt = importlib.import_module("train_gpt")

            model = train_gpt.GPT(
                vocab_size=128,
                num_layers=2,
                num_unique_layers=2,
                model_dim=64,
                num_heads=4,
                num_kv_heads=2,
                mlp_mult=2,
                tie_embeddings=True,
                tied_embed_init_std=0.01,
                logit_softcap=30.0,
                rope_base=10000.0,
                qk_gain_init=1.0,
            )
            model.train()
            compiled = torch.compile(model, dynamic=False, fullgraph=True)
            x = torch.randint(0, 128, (2, 16), dtype=torch.long)
            y = torch.randint(0, 128, (2, 16), dtype=torch.long)
            loss = compiled(x, y)
            self.assertTrue(torch.isfinite(loss).item())
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt", None)


if __name__ == "__main__":
    unittest.main()
