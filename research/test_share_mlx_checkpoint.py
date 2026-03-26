from __future__ import annotations

import unittest

from research.share_mlx_checkpoint import source_block_indices_for_target


class ShareMlxCheckpointTests(unittest.TestCase):
    def test_average_modulo_pairs_encoder_decoder_blocks(self) -> None:
        self.assertEqual(
            source_block_indices_for_target(
                target_idx=2,
                source_unique_layers=12,
                target_unique_layers=6,
                mode="average_modulo",
            ),
            [2, 8],
        )

    def test_first_cycle_uses_encoder_side(self) -> None:
        self.assertEqual(
            source_block_indices_for_target(
                target_idx=4,
                source_unique_layers=12,
                target_unique_layers=6,
                mode="first_cycle",
            ),
            [4],
        )

    def test_last_cycle_uses_decoder_side(self) -> None:
        self.assertEqual(
            source_block_indices_for_target(
                target_idx=1,
                source_unique_layers=12,
                target_unique_layers=6,
                mode="last_cycle",
            ),
            [7],
        )

    def test_average_modulo_allows_non_divisible_targets(self) -> None:
        self.assertEqual(
            source_block_indices_for_target(
                target_idx=3,
                source_unique_layers=12,
                target_unique_layers=8,
                mode="average_modulo",
            ),
            [3, 11],
        )

    def test_contiguous_partition_keeps_locality(self) -> None:
        self.assertEqual(
            source_block_indices_for_target(
                target_idx=1,
                source_unique_layers=12,
                target_unique_layers=8,
                mode="contiguous_partition",
            ),
            [1, 2],
        )

    def test_contiguous_partition_covers_tail(self) -> None:
        self.assertEqual(
            source_block_indices_for_target(
                target_idx=8,
                source_unique_layers=12,
                target_unique_layers=9,
                mode="contiguous_partition",
            ),
            [10, 11],
        )


if __name__ == "__main__":
    unittest.main()
