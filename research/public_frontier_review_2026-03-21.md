# Public Frontier Review (2026-03-21)

## Current public bar

- Official README leaderboard top score on `2026-03-21`: `1.1428` shipped `val_bpb`
- The repo README explicitly warns that the leaderboard can lag behind the strongest open PRs.
- Stronger open PR claims visible on `2026-03-21`:
  - `#338`: `1.1254` sliding `val_bpb`, `1.1256` three-seed mean
  - `#332`: `1.1320`
  - `#339`: `1.1364`, but explicitly over cap at `16,170,051` bytes
- Practical implication: our real public target is no longer `1.1428`; we should think in terms of `~1.125` or better for a meaningful record attempt.
- Strongest themes in the merged public records and PRs:
  - sliding-window eval at `stride=64`
  - `seq_len=2048` or `4096` when it pays for itself
  - `MLP 3x` funded by more aggressive compression
  - int5/int6 or mixed int6/int8 export with `zstd-22`
  - quantization-aware training only when throughput survives
  - `WD=0.04`, Muon `0.99`, longer warmdown, and EMA/SWA
  - lightweight token-pair features (`SmearGate`, `BigramHash`)
  - newer zero-parameter structure ideas like XSA, Partial RoPE, and LN scaling
  - in one top PR, bounded test-time training on already-scored validation tokens

## What they are doing better than us

- They are co-designing architecture, optimizer, export, and evaluation together instead of treating export as the only hard problem.
- Their absolute model quality is much higher before export.
- They are explicitly spending bytes on wider `MLP 3x` blocks and then buying those bytes back with int5/int6 plus better compression.
- They are taking the free eval win from sliding context, which our mainline stack did not support before this session.

## What we are doing better than them

- Our best modern export frontier has a much smaller clean-to-shipped gap than the older public int8-style paths.
- The active `12x576 share6` branch plus QER/codebook export now creates a stronger measured compact Pareto frontier than before:
  - absolute winner: `proj_top6_attn_top6_fp16` at `2.13867183` and `15,175,610` total bytes
  - closest compact frontier: `lloyd_codebook_qer_projattn_top6_r224` at `2.13943493` and `14,409,932` total bytes
  - leaner compact point: `codebook_qer_projattn_top6_r192` at `2.13976149` and `13,346,786` total bytes
- That means our exporter is still a real asset on the shared-depth branch, but it is a branch-local asset until proven elsewhere.

## Scaling judgment

- The current `12x576 share6` QER line still looks close to a **local plateau** on raw model quality.
- The exporter advantage is real on that branch, but it did **not** transfer to the stronger alive `u5k 12x608` branch:
  - fixed transfer sweep: [local_u5k_12x608_seq896_codebook_transfer.md](/Users/ronaldschmidt/openai/research/local_u5k_12x608_seq896_codebook_transfer.md)
  - `fcproj_hi` stayed best at `2.10282351`
  - every tested codebook/QER variant was worse and over cap
- So the exporter win is not yet a universal law. It currently looks **share6-specific** rather than model-agnostic.
- The branch is also too far behind the public absolute frontier to expect a jump to `~1.0` from more compute alone.
- The next H100 bet should therefore still be a **hybrid**:
  - keep our export/byte advantage where it is actually measured
  - add the strongest public training/eval ideas we were still missing
  - avoid assuming the `share6` exporter automatically drops into `12x608`-style families

## This session's concrete action

- Added `sliding eval` support to our Torch stack in [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py).
- Added proper optimizer weight decay to [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py):
  - `AdamW` on the Adam paths
  - decoupled decay in `Muon`
  - `WEIGHT_DECAY`, `ADAM_WEIGHT_DECAY`, `MUON_WEIGHT_DECAY`
- Added two H100 launchers:
  - [train_breakthrough_sp1024_conservative_slide64_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_conservative_slide64_8xh100.sh)
  - [train_breakthrough_sp1024_wildcard_mlp3_slide64_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_wildcard_mlp3_slide64_8xh100.sh)
  - both now default to `WEIGHT_DECAY=0.04`

## Decision

- Stop spending local cycles on tiny `12x576 share6` microtails by default.
- Keep the exporter as a **conditional** transferable advantage:
  - proven on `12x576 share6`
  - not yet proven on `12x608` unique-depth families
- External literature scan in [web_breakthrough_scan_2026-03-21.md](/Users/ronaldschmidt/openai/research/web_breakthrough_scan_2026-03-21.md) strengthens the same conclusion: we are still missing several large public-frontier ingredients, especially `int5/int6 + zstd` and `EMA/SWA`.
- For local work, reopen only:
  - the alive `u5k` token-flow branch
  - one hybrid branch that combines our exporter ideas with stronger public training/eval structure
- For H100, prefer:
  1. one more integration pass on the missing public levers
  2. then conservative `10x512` + `slide64`
  3. then wildcard `11x512` + `MLP3x` + `2048` + `slide64`
