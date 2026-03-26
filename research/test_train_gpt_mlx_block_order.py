from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import unittest


class TrainGptMlxBlockOrderTests(unittest.TestCase):
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

    def test_logical_block_order_defaults_to_modulo(self) -> None:
        module = self._reload_module()
        self.assertEqual(module.logical_block_order(12, 10, ""), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1))
        self.assertEqual(module.compact_logical_block_order((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1), 10), "modulo")
        sys.modules.pop("train_gpt_mlx", None)

    def test_logical_block_order_accepts_explicit_remap(self) -> None:
        module = self._reload_module()
        order = module.logical_block_order(12, 10, "0,1,2,3,4,5,6,7,8,9,8,9")
        self.assertEqual(order, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9))
        self.assertEqual(module.compact_logical_block_order(order, 10), "0,1,2,3,4,5,6,7,8,9,8,9")
        sys.modules.pop("train_gpt_mlx", None)

    def test_logical_block_order_rejects_wrong_length(self) -> None:
        module = self._reload_module()
        with self.assertRaisesRegex(ValueError, "must provide exactly 12 indices"):
            module.logical_block_order(12, 10, "0,1,2")
        sys.modules.pop("train_gpt_mlx", None)

    def test_logical_block_order_rejects_out_of_range(self) -> None:
        module = self._reload_module()
        with self.assertRaisesRegex(ValueError, "out of range"):
            module.logical_block_order(12, 10, "0,1,2,3,4,5,6,7,8,9,10,11")
        sys.modules.pop("train_gpt_mlx", None)


if __name__ == "__main__":
    unittest.main()
