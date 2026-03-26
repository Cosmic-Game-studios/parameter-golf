from __future__ import annotations

import unittest

from train_gpt import sliding_window_segments


def covered_positions(total_tokens: int, seq_len: int, stride: int) -> list[int]:
    covered: list[int] = []
    for window_start, local_start, local_end in sliding_window_segments(total_tokens, seq_len, stride):
        covered.extend(range(window_start + local_start, window_start + local_end))
    return covered


class SlidingEvalTests(unittest.TestCase):
    def test_segments_cover_each_token_once(self) -> None:
        for total_tokens, seq_len, stride in ((10, 4, 2), (9, 4, 3), (1050, 1024, 64), (1024, 1024, 64)):
            self.assertEqual(covered_positions(total_tokens, seq_len, stride), list(range(total_tokens)))

    def test_invalid_stride_raises(self) -> None:
        with self.assertRaises(ValueError):
            sliding_window_segments(10, 4, 0)


if __name__ == "__main__":
    unittest.main()
