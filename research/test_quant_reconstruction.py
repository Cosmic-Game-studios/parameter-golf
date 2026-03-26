from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from quant_reconstruction import (
    apply_lowrank_residual,
    compute_lowrank_residual,
    dequantize_codebook_int4_blockwise,
    lloyd_codebook_4bit,
    mulaw_codebook_4bit,
    normal_codebook_4bit,
    quantile_codebook_4bit,
    quantize_codebook_int4_blockwise,
    submission_code_text,
    uniform_codebook_4bit,
)


class QuantReconstructionTests(unittest.TestCase):
    def test_lowrank_residual_reduces_error(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.normal(size=(16, 12)).astype(np.float32)
        approx = (target * 0.85).astype(np.float32)
        left, right, kept_rank = compute_lowrank_residual(target, approx, rank=4)
        repaired = apply_lowrank_residual(approx, left, right)
        self.assertEqual(kept_rank, 4)
        self.assertLess(np.square(target - repaired).mean(), np.square(target - approx).mean())

    def test_lowrank_rank_zero_is_noop(self) -> None:
        target = np.eye(4, dtype=np.float32)
        approx = np.zeros((4, 4), dtype=np.float32)
        left, right, kept_rank = compute_lowrank_residual(target, approx, rank=0)
        repaired = apply_lowrank_residual(approx, left, right)
        self.assertEqual(kept_rank, 0)
        self.assertTrue(np.array_equal(repaired, approx))

    def test_submission_code_text_concatenates_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.py"
            b = Path(td) / "b.py"
            a.write_text("print('a')\n", encoding="utf-8")
            b.write_text("print('b')\n", encoding="utf-8")
            text = submission_code_text(a, b)
        self.assertIn("print('a')", text)
        self.assertIn("print('b')", text)

    def test_normal_codebook_is_sorted_and_bounded(self) -> None:
        codebook = normal_codebook_4bit()
        self.assertEqual(codebook.shape, (16,))
        self.assertTrue(np.all(np.diff(codebook) > 0))
        self.assertAlmostEqual(float(np.max(np.abs(codebook))), 1.0)

    def test_uniform_codebook_is_sorted_and_bounded(self) -> None:
        codebook = uniform_codebook_4bit()
        self.assertEqual(codebook.shape, (16,))
        self.assertTrue(np.all(np.diff(codebook) > 0))
        self.assertAlmostEqual(float(codebook[0]), -1.0)
        self.assertAlmostEqual(float(codebook[-1]), 1.0)

    def test_mulaw_codebook_is_sorted_and_bounded(self) -> None:
        codebook = mulaw_codebook_4bit()
        self.assertEqual(codebook.shape, (16,))
        self.assertTrue(np.all(np.diff(codebook) > 0))
        self.assertAlmostEqual(float(codebook[0]), -1.0)
        self.assertAlmostEqual(float(codebook[-1]), 1.0)
        self.assertLess(float(abs(codebook[8] - codebook[7])), float(abs(codebook[-1] - codebook[-2])))

    def test_quantile_codebook_tracks_empirical_distribution(self) -> None:
        values = np.concatenate(
            (
                np.full((200,), -0.75, dtype=np.float32),
                np.full((400,), 0.05, dtype=np.float32),
                np.full((200,), 0.65, dtype=np.float32),
            )
        )
        codebook = quantile_codebook_4bit(values)
        self.assertEqual(codebook.shape, (16,))
        self.assertTrue(np.all(np.diff(codebook) >= 0))
        self.assertGreater(float(codebook[8]), -0.1)
        self.assertLess(float(codebook[8]), 0.2)

    def test_lloyd_codebook_reduces_squared_error_vs_quantile(self) -> None:
        rng = np.random.default_rng(0)
        values = np.clip(rng.normal(loc=0.05, scale=0.28, size=8192).astype(np.float32), -1.0, 1.0)
        quantile = quantile_codebook_4bit(values)
        lloyd = lloyd_codebook_4bit(values)
        q_err = np.square(values[:, None] - quantile[None, :]).min(axis=1).mean()
        l_err = np.square(values[:, None] - lloyd[None, :]).min(axis=1).mean()
        self.assertLessEqual(float(l_err), float(q_err) + 1e-7)

    def test_codebook_blockwise_roundtrip_reduces_error_vs_zero(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.normal(size=(12, 20)).astype(np.float32)
        signed, scale, meta = quantize_codebook_int4_blockwise(target, block_size=8)
        repaired = dequantize_codebook_int4_blockwise(signed, scale, meta)
        self.assertLess(np.square(target - repaired).mean(), np.square(target).mean())
        self.assertEqual(meta["scheme"], "per_row_block_codebook_int4")

    def test_quantile_codebook_roundtrip_stores_tensor_specific_values(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.normal(loc=0.15, scale=0.35, size=(10, 24)).astype(np.float32)
        signed, scale, meta = quantize_codebook_int4_blockwise(target, block_size=8, codebook_name="quantile16")
        repaired = dequantize_codebook_int4_blockwise(signed, scale, meta)
        self.assertIn("codebook_values", meta)
        self.assertEqual(np.asarray(meta["codebook_values"]).shape, (16,))
        self.assertLess(np.square(target - repaired).mean(), np.square(target).mean())

    def test_lloyd_codebook_roundtrip_stores_tensor_specific_values(self) -> None:
        rng = np.random.default_rng(0)
        target = rng.normal(loc=-0.05, scale=0.22, size=(10, 24)).astype(np.float32)
        signed, scale, meta = quantize_codebook_int4_blockwise(target, block_size=8, codebook_name="lloyd16")
        repaired = dequantize_codebook_int4_blockwise(signed, scale, meta)
        self.assertIn("codebook_values", meta)
        self.assertEqual(np.asarray(meta["codebook_values"]).shape, (16,))
        self.assertLess(np.square(target - repaired).mean(), np.square(target).mean())


if __name__ == "__main__":
    unittest.main()
