from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import unittest

import numpy as np


class TrainGptMlxBusTests(unittest.TestCase):
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

    def test_parse_layer_selector_supports_top_and_ranges(self) -> None:
        saved = {k: os.environ.get(k) for k in ("GLOBAL_BUS_READ_LAYERS", "GLOBAL_BUS_WRITE_LAYERS", "GLOBAL_BUS_SUMMARY_MODE", "GLOBAL_BUS_TAIL_WEIGHT")}
        try:
            os.environ["GLOBAL_BUS_READ_LAYERS"] = "all"
            os.environ["GLOBAL_BUS_WRITE_LAYERS"] = "all"
            module = self._reload_module()
            self.assertEqual(module.parse_layer_selector("top4", 12), (False, False, False, False, False, False, False, False, True, True, True, True))
            self.assertEqual(module.parse_layer_selector("8:12", 12), (False, False, False, False, False, False, False, False, True, True, True, True))
            self.assertEqual(module.parse_layer_selector("0,2,11", 12), (True, False, True, False, False, False, False, False, False, False, False, True))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)

    def test_compact_layer_selector_formats_all_and_sparse(self) -> None:
        saved = {k: os.environ.get(k) for k in ("GLOBAL_BUS_READ_LAYERS", "GLOBAL_BUS_WRITE_LAYERS", "GLOBAL_BUS_SUMMARY_MODE", "GLOBAL_BUS_TAIL_WEIGHT")}
        try:
            module = self._reload_module()
            self.assertEqual(module.compact_layer_selector((True, True, True)), "all")
            self.assertEqual(module.compact_layer_selector((False, True, False, True)), "1,3")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)

    def test_configure_global_bus_boundary_lut_marks_leading_space_and_special_tokens(self) -> None:
        saved = {k: os.environ.get(k) for k in ("GLOBAL_BUS_READ_LAYERS", "GLOBAL_BUS_WRITE_LAYERS", "GLOBAL_BUS_SUMMARY_MODE", "GLOBAL_BUS_TAIL_WEIGHT")}
        try:
            module = self._reload_module()
            module.configure_global_bus_boundary_lut(
                np.array([0, 1, 0, 0], dtype=np.bool_),
                np.array([1, 0, 0, 1], dtype=np.bool_),
            )
            lut = module.GLOBAL_BUS_BOUNDARY_LUT
            self.assertIsNotNone(lut)
            self.assertEqual(lut.tolist(), [True, True, False, True])
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("train_gpt_mlx", None)


if __name__ == "__main__":
    unittest.main()
