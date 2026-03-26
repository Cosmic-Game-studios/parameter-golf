# Share6 H100 Prep

## Goal
Prepare the strongest locally validated `sp1024` modern branch for real `8xH100 / 600s` validation instead of keeping it trapped in M4-only evidence.

## Local evidence anchor
- Architecture anchor:
  - `12` layers
  - `6` unique layers (`share6`)
  - `dim=576`
  - `kv_heads=2`
  - `seq_len=896`
- Best local absolute exporter:
  - checkpoint family: `lab_sp1024_12x576_kv2_share6_codebook_qer192_tail20_lowlr_m4`
  - legal exporter: `proj_top6_attn_top6_fp16`
  - clean `val_bpb`: `2.1312`
  - shipped `val_bpb`: `2.13867183`
  - total bytes: `15,169,291`
- Best local compact exporter:
  - `codebook_qer_projattn_top6_r192`
  - shipped `val_bpb`: `2.13976149`
  - total bytes: `13,340,467`

## Prepared H100 launchers

### 1. Absolute-quality mainline
- Script:
  - [train_breakthrough_sp1024_share6_fp16_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_share6_fp16_8xh100.sh)
- Why:
  - closest official-budget translation of the best local absolute exporter
  - keeps the proven `proj_top6_attn_top6_fp16` shipping policy
  - folds in the already integrated H100 levers: `slide64`, `weight_decay=0.04`, `EMA=0.997`

### 2. Compact codebook/QER frontier
- Script:
  - [train_breakthrough_sp1024_share6_codebook_qer_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_share6_codebook_qer_8xh100.sh)
- Why:
  - carries the strongest local compact frontier into the official Torch/H100 stack
  - tests whether the local Pareto win survives under real H100 budget
  - uses `mixed_codebook_int4_int8_packed_v1` plus `LOWRANK_ERROR_RANK=192`

## Order of operations
1. Finish the live `1xH100` conservative anchor run and record the shipped result.
2. Run the absolute-quality `share6` H100 launcher first.
3. Run the compact codebook/QER launcher second only if budget remains or if the mainline run shows the family is healthy.

## Important caveat
- These launchers transfer the best **local family and exporter logic**, not the MLX checkpoint itself.
- We still do not have a compatible official Torch checkpoint for this `share6` line, so the H100 validation is a fresh official-budget training run on the locally strongest modern structure.
