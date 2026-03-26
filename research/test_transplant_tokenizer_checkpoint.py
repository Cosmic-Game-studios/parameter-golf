from __future__ import annotations

import unittest

import numpy as np

from research.transplant_tokenizer_checkpoint import build_transplanted_embedding


class FakeSP:
    def __init__(self, decode_map: dict[int, str], encode_map: dict[str, list[int]], vocab_size: int):
        self._decode_map = decode_map
        self._encode_map = encode_map
        self._vocab_size = vocab_size

    def vocab_size(self) -> int:
        return self._vocab_size

    def decode(self, ids: list[int]) -> str:
        if len(ids) != 1:
            raise AssertionError("fake only supports single-token decode")
        return self._decode_map.get(ids[0], "")

    def encode(self, text: str, out_type=int) -> list[int]:
        _ = out_type
        return list(self._encode_map.get(text, []))

    def id_to_piece(self, token_id: int) -> str:
        return self._decode_map.get(token_id, "")


class TransplantTokenizerCheckpointTests(unittest.TestCase):
    def test_embedding_transplant_copies_specials_and_maps_rows(self) -> None:
        source_emb = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
        source_sp = FakeSP(
            decode_map={4: "hello", 5: " world"},
            encode_map={
                " world": [5],
                "hello world": [4, 5],
            },
            vocab_size=6,
        )
        target_sp = FakeSP(
            decode_map={4: "hello", 5: " world", 6: "hello world", 7: ""},
            encode_map={},
            vocab_size=8,
        )

        out, summary = build_transplanted_embedding(
            source_emb=source_emb,
            source_sp=source_sp,
            target_sp=target_sp,
            init_std=0.01,
            seed=7,
        )

        np.testing.assert_allclose(out[0], source_emb[0])
        np.testing.assert_allclose(out[1], source_emb[1])
        np.testing.assert_allclose(out[2], source_emb[2])
        np.testing.assert_allclose(out[3], source_emb[3])
        np.testing.assert_allclose(out[4], source_emb[4])
        np.testing.assert_allclose(out[5], source_emb[5])
        np.testing.assert_allclose(out[6], (source_emb[4] + source_emb[5]) / 2.0)
        self.assertEqual(summary["copied_special"], 4)
        self.assertEqual(summary["mapped_rows"], 3)
        self.assertEqual(summary["exact_piece_rows"], 2)
        self.assertEqual(summary["averaged_rows"], 1)
        self.assertEqual(summary["random_rows"], 1)


if __name__ == "__main__":
    unittest.main()
