# Current Frontier Handoff

## 2026-03-22 mission reset + first bounded cycle
- The mission is now broader than “shave the current local U4K point”: the active research program is a two-stage path toward a credible `0.8-0.9` shipped `val_bpb` LM, with local M4 falsification first and official H100 promotion only for branches that earn it.
- New runtime policy artifacts now exist:
  - [research/brief/MISSION_INTAKE.md](/Users/ronaldschmidt/openai/research/brief/MISSION_INTAKE.md)
  - [research/brief/IMPROVEMENT_PROTOCOL.md](/Users/ronaldschmidt/openai/research/brief/IMPROVEMENT_PROTOCOL.md)
  - [research/brief/AUTONOMY_POLICY.md](/Users/ronaldschmidt/openai/research/brief/AUTONOMY_POLICY.md)
- First post-reset bounded cycle is complete and negative:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt)
  - same donor, same sparse exporter family, but `cross_skip` only on logical decoder layers `10,11`
  - clean `4.5561 / 1.9871`
  - shipped `4.55026770 / 1.98455752`
  - compressed model `14,842,763`
  - estimated total `14,937,113`
  - decision: `revert`
- Interpretation:
  - the reproduced `cross_skip` donor remains the best local bar
  - the first sparse/asymmetric mask tweak is already a loser
  - the next local cycle should now be materially more orthogonal and more scaling-oriented than another tiny control tweak on the same donor

## 2026-03-22 local `u4k share10 dualrole + bus` token-pair audit
- Direct `SmearGate + BigramHash` on the reproduced `cross_skip` donor is negative:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_pairfeat40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_pairfeat40_fc_top4_m4.txt)
  - clean `4.5568 / 1.9874`
  - shipped `4.55077076 / 1.98477693`
- `SmearGate` alone is also negative:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_smear40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_smear40_fc_top4_m4.txt)
  - clean `4.5568 / 1.9874`
  - shipped `4.55083466 / 1.98480479`
- `BigramHash` alone is the only near-tie:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_bigram40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_bigram40_fc_top4_m4.txt)
  - clean `4.5562 / 1.9871`
  - training-run shipped `4.55021381 / 1.98453402`
  - targeted exporter check: [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_bigram40_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_bigram40_targeted_export_check.md)
  - best fixed policy still `fc_top4_int4` at `1.98453381`, total `15,044,799`
- Important tooling fix:
  - [search_export_gap.py](/Users/ronaldschmidt/openai/research/search_export_gap.py) no longer closes NPZ checkpoints before reading pair-feature shapes; the new regression lives in [test_search_export_gap.py](/Users/ronaldschmidt/openai/research/test_search_export_gap.py)
- Current interpretation:
  - the first token-pair frontier stack is not the breakthrough on this donor
  - `BigramHash` is the only part worth remembering, but only as a near-tie and only if we can revisit it on a stronger donor or a smaller cheaper hash path
  - the best local bar is still the reproduced tiny `cross_skip` keep, not token-pair features

## 2026-03-22 local `u4k share10 dualrole + bus` cross-skip reproduction + busjoint update
- The tiny `cross_skip` keep is now reproduced exactly:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt)
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt)
  - both landed at clean `4.5562 / 1.9871` and training-run shipped `4.55020714 / 1.98453111`
- The strongest fixed exporter on that reproduced checkpoint remains:
  - [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md)
  - `fc_top4_int4`
  - shipped `1.98453131`
  - total `14,937,204`
- New orthogonal refutation:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_busjoint40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_busjoint40_fc_top4_m4.txt)
  - same donor, jointly train `global_bus*` and `cross_skip*`
  - clean `4.5562 / 1.9871`
  - training-run shipped `4.55024624 / 1.98454816`
- New launcher:
  - [train_lab_u4k_12x608_share10_crossskip_busjoint_branch_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_12x608_share10_crossskip_busjoint_branch_m4.sh)
- Current interpretation:
  - `cross_skip` is a real and reproduced tiny control-tensor keep
  - the best fixed exporter on that checkpoint is plain `fc_top4_int4`
  - jointly moving the bus again is slightly harmful on this donor
  - the next local move should stay inside tiny communication controls, but not by reopening joint `cross_skip + global_bus`

## 2026-03-22 local `u4k share10 dualrole + bus` remap + cross-skip update
- New best measured local point:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt)
  - clean `4.5562 / 1.9871`
  - training-run shipped `4.55020714 / 1.98453111`
  - total under fixed exporter: `14,937,204`
- Promotion check on the same checkpoint:
  - [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md)
  - best fixed exporter is now just `fc_top4_int4`
  - shipped `1.98453131`
  - total `14,937,204`
  - the old near-cap `QER r64` legalizer fell back to `1.98454109`
- New structural refutation:
  - the bounded logical block-order remap bracket lost in every tested geometry:
    - `...8,9,8,9` -> `1.98614203`
    - `...9,9,8,8` -> `1.98605052`
    - `...6,7` -> `1.98580117`
    - `...4,5` -> `1.98499612`
- Important tooling fixes:
  - [train_gpt_mlx.py](/Users/ronaldschmidt/openai/train_gpt_mlx.py) now supports `LOGICAL_BLOCK_ORDER` and a tiny all-encoder `cross_skip` router
  - [search_export_gap.py](/Users/ronaldschmidt/openai/research/search_export_gap.py) now recovers `cross_skip` runtime flags and anchors all metadata matches to real log lines so embedded source text cannot poison reevaluation
- Current interpretation:
  - explicit logical block remap is not the breakthrough on this donor
  - a tiny all-encoder skip router is a real next communication lever
  - the gain is still microscopic, but it has now reproduced exactly
  - the joint `cross_skip + global_bus` follow-up already lost, so the next local step should stay inside `cross_skip`-style tiny communication controls, not a broader exporter sweep

## 2026-03-22 local `u4k share10 dualrole + bus` token-flow + second-pass update
- The active local legal anchor is unchanged:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_fc_top4_m4.txt)
  - clean `4.5561 / 1.9871`
  - training-run shipped `4.55021858 / 1.98453610`
  - total `14,937,014`
- Exact fixed-export reevaluation on the same donor still favors:
  - [local_u4k_12x608_share10_dualrole01_bus20_qer_legalizer_export_gap_v2.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_qer_legalizer_export_gap_v2.md)
  - `fc_top4_attn_top1_qer_proj_top1_r64`
  - shipped `1.98453631`
  - total `15,637,017`
- New orthogonal refutations on the same donor:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_byteweight40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_byteweight40_fc_top4_m4.txt)
    - clean `4.7447 / 2.0694`
    - shipped `4.68318987 / 2.04253031`
    - best fixed exporter in [local_u4k_12x608_share10_dualrole01_bus20_byteweight40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_byteweight40_export_gap.md) only reached `2.04248476`
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_secondpass89_gate40_fc_top4_m4_retry1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_secondpass89_gate40_fc_top4_m4_retry1.txt)
    - clean `4.5841 / 1.9993`
    - shipped `4.56823444 / 1.99239355`
    - best fixed exporter in [local_u4k_12x608_share10_dualrole01_bus20_secondpass89_export_gap_v2.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_secondpass89_export_gap_v2.md) only reached `1.99238898`
- Important tooling fix:
  - [search_export_gap.py](/Users/ronaldschmidt/openai/research/search_export_gap.py) now recovers `global_bus` / `second_pass` runtime flags from the source training log before exporter reevaluation.
- Current interpretation:
  - the donor is still real and still the local bar
  - naive global token-byte weighting is not the next win
  - the first cheap gated decoder `second_pass` design is also not the next win
  - the next high-EV local branch should be a structural remap / reuse change, not another bus-only tweak, not another pure loss-weighting branch, and not a replay of the same second-pass design

## 2026-03-22 local `u4k share10 dualrole + bus` legalization update
- The active research mission is still local-first `u4k`, but the working donor has moved past the old `share6` branch.
- Current overall local legal anchor:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_fc_top4_m4.txt)
  - clean `4.5561 / 1.9871`
  - training-run shipped `4.55021858 / 1.98453610`
  - fixed-export re-eval under `fc_top4_int4`: `1.98454462`
  - total bytes: `14,937,014`
- Best “make the too-expensive thing smaller” exporter on the same checkpoint:
  - [local_u4k_12x608_share10_dualrole01_bus20_qer_legalizer_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_qer_legalizer_export_gap.md)
  - `fc_top4_attn_top1_qer_proj_top1_r64`
  - shipped `1.98453631`
  - total `15,637,017`
- Best tiny-fp16 exporter keep on the same checkpoint:
  - [local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md)
  - `fc_top4_attn_top1_fp16`
  - shipped `1.98453797`
  - total `15,420,393`
  - reproduced exactly in [local_u4k_12x608_lfqat60_m4__4dfe288810__fc_top4_attn_top1_fp16_repro1.txt](/Users/ronaldschmidt/openai/logs/export_gap_local_u4k_12x608_lfqat60_seq1024/local_u4k_12x608_lfqat60_m4__4dfe288810__fc_top4_attn_top1_fp16_repro1.txt)
- Important refutations on this donor:
  - [local_u4k_12x608_share10_dualrole01_qer_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_qer_export_gap.md): medium-rank `QER`/codebook-QER still loses to `fc_top4_int4`
  - [local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md): `fc_top3_int4` regresses to `1.98545449`
  - [local_u4k_12x608_share10_dualrole01_bus20_top2hybrid_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_top2hybrid_export_gap.md): `fc_top4_attn_top2_fp16` is slightly worse than top1, and proj-heavy top2 variants are over cap
  - [local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md): `fc_top4_proj_top1_attn_top1_fp16` is slightly better in raw shipped score but still illegal at `16,435,203`
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_boundary_top4r_top2w20_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_boundary_top4r_top2w20_fc_top4_m4.txt): boundary-aware bus v2 stayed flat at `1.98455440`
- Current interpretation:
  - the useful expensive corridor is real, but it pays off best as a near-cap legalizer (`QER r64`) or as a tiny fp16 keep, not as a broader sweep
  - the next high-EV gain is likely a pure token-flow checkpoint lever on this donor, not another communication-only bus tweak or a broader exporter sweep
  - keep the old `share6` notes only as historical fallback, not as the default next branch

## 2026-03-22 local `u4k share6` reset
- The active research mission is local again. The stale `sp1024`/H100-prep framing is no longer the right default for the next loop.
- The locked local legal winner is still:
  - [dense_u4k_12x608_kv2_lfqat60_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj.txt)
  - clean `4.5577 / 1.9878`
  - shipped `4.5523 / 1.98544845`
- New important structural branch:
  - [lab_u4k_12x608_kv2_share6_first_cycle_continue120_int8all_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share6_first_cycle_continue120_int8all_m4.txt)
  - best legal exporter from [local_u4k_12x608_share6_first_cycle_continue120_int8all_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share6_first_cycle_continue120_int8all_export_gap.md):
    - `projhi_attnhi_fp16`
    - shipped `2.00187835`
    - total `15,423,066`
- Important refutations:
  - [lab_u4k_12x608_kv2_share6_average_modulo_eval0_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share6_average_modulo_eval0_m4.txt) is dead
  - [lab_u4k_12x608_kv2_share6_last_cycle_eval0_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share6_last_cycle_eval0_m4.txt) is dead
  - [lab_u4k_12x640_kv2_share6_first_cycle_expand_continue120_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x640_kv2_share6_first_cycle_expand_continue120_m4.txt) regressed to `2.01324506`
  - [lab_u4k_12x704_kv2_share6_first_cycle_expand_continue120_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x704_kv2_share6_first_cycle_expand_continue120_m4.txt) regressed to `2.02064519`
  - [lab_u4k_12x608_kv2_share6_first_cycle_projhi_attnhi_recovery80_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share6_first_cycle_projhi_attnhi_recovery80_m4.txt) regressed to `2.00507337`
- New reusable local tooling:
  - [research/share_mlx_checkpoint.py](/Users/ronaldschmidt/openai/research/share_mlx_checkpoint.py)
  - [scripts/local/train_lab_u4k_12xshare6_expand_continue_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_12xshare6_expand_continue_m4.sh)
- Current interpretation:
  - `u4k share6 first_cycle` is a real branch, but it is not yet the champion
  - simple width-up is not the breakthrough on this family
  - the next high-EV local move should be orthogonal: partial sharing or token-processing, not another same-family width-up or kept-tensor recovery tail

## 2026-03-21 share6 H100 prep update
- The strongest locally validated modern family now has dedicated `8xH100` launchers instead of only M4 scripts:
  - [train_breakthrough_sp1024_share6_fp16_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_share6_fp16_8xh100.sh)
  - [train_breakthrough_sp1024_share6_codebook_qer_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_share6_codebook_qer_8xh100.sh)
- Prep memo:
  - [SHARE6_H100_PREP.md](/Users/ronaldschmidt/openai/research/brief/SHARE6_H100_PREP.md)
- Intent:
  - stop treating the best `12x576 share6` local line as M4-only evidence
  - validate it directly on the official Torch/H100 stack after the current conservative anchor run finishes
- These launchers intentionally combine:
  - the local `12x576 / share6 / seq896 / kv2` family
  - `slide64`
  - `weight_decay=0.04`
  - `EMA=0.997`
  - either the absolute local winner `proj_top6_attn_top6_fp16` or the compact `codebook+QER r192` frontier
- Not yet measured:
  - no H100 score exists yet for either new `share6` launcher
  - they are launch-ready, not validated

## 2026-03-21 EMA integration update
- The official Torch stack now has opt-in model averaging:
  - [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) supports `EMA_DECAY`, `EMA_START_STEP`, and `EMA_UPDATE_EVERY`
  - EMA updates happen during training on the official path, not only in local MLX experiments
  - the final export path now compares raw vs EMA on the real shipped objective and automatically keeps the lower roundtrip `val_bpb`
- Both breakthrough H100 launchers now default to `EMA_DECAY=0.997`:
  - [train_breakthrough_sp1024_conservative_slide64_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_conservative_slide64_8xh100.sh)
  - [train_breakthrough_sp1024_wildcard_mlp3_slide64_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_wildcard_mlp3_slide64_8xh100.sh)
- Verification completed:
  - `python3 -m py_compile train_gpt.py research/test_ema.py research/test_optimizer_weight_decay.py research/test_train_gpt_compile.py`
  - `python3 -m unittest research.test_ema research.test_optimizer_weight_decay research.test_train_gpt_compile research.test_sliding_eval research.test_quant_reconstruction research.test_search_export_gap`
  - `bash -n` on both H100 launchers
- Interpretation:
  - this closes another obvious public-frontier gap before the next real H100 run
  - there is still **no measured H100 score** on the EMA path yet
  - the next highest-EV integration remains `zstd + int5/int6`, then a conservative real H100 run

## 2026-03-21 web-scan update
- New memo:
  - [web_breakthrough_scan_2026-03-21.md](/Users/ronaldschmidt/openai/research/web_breakthrough_scan_2026-03-21.md)
- Main conclusion:
  - we are **not** at a hard wall
  - the next biggest missing public-frontier levers are still `int5/int6 + zstd` and a measured verdict on `EMA` vs raw
  - therefore the best next move is **not** an immediate H100 launch yet if the goal is a real breakthrough shot
- Public/open evidence now points to a practical target closer to `~1.125` than the stale README `1.1428` line.
- Immediate recommendation:
  1. integrate mixed `int5/int6` plus stronger compression
  2. integrate one averaging path (`EMA` or `SWA`)
  3. then run the conservative H100 branch

## 2026-03-21 H100 launch-hardening update
- The H100 stack is now missing one fewer public-frontier lever:
  - [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) now supports real optimizer weight decay
  - Adam paths use decoupled `AdamW`
  - the Muon path now applies decoupled weight decay too
- New env knobs:
  - `WEIGHT_DECAY`
  - `ADAM_WEIGHT_DECAY`
  - `MUON_WEIGHT_DECAY`
- Both prepared H100 breakthrough launchers now default to `WEIGHT_DECAY=0.04`:
  - [train_breakthrough_sp1024_conservative_slide64_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_conservative_slide64_8xh100.sh)
  - [train_breakthrough_sp1024_wildcard_mlp3_slide64_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_wildcard_mlp3_slide64_8xh100.sh)
- Verification completed:
  - `py_compile` on `train_gpt.py` and the new WD regression test
  - `python3 -m unittest research.test_optimizer_weight_decay research.test_train_gpt_compile research.test_sliding_eval`
  - `bash -n` on both H100 launchers
- Interpretation:
  - the codebase is now materially closer to the public frontier recipe
  - we still do **not** have measured H100 evidence for the new WD branch yet
  - first real run should still be the conservative launcher, then wildcard only if budget remains

## 2026-03-21 smarter-codebook + transfer update
- New focused reports:
  - [local_sp1024_12x576_share6_lloyd_mulaw_frontier.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_lloyd_mulaw_frontier.md)
  - [local_sp1024_12x576_share6_compact_frontier_2026-03-21.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_compact_frontier_2026-03-21.md)
  - [local_u5k_12x608_seq896_codebook_transfer.md](/Users/ronaldschmidt/openai/research/local_u5k_12x608_seq896_codebook_transfer.md)
- Active `12x576 share6` compact frontier with current counted code bytes:
  - absolute winner: `proj_top6_attn_top6_fp16` at `2.13867183`, total `15,175,610`
  - new closest compact frontier: `lloyd_codebook_qer_projattn_top6_r224` at `2.13943493`, total `14,409,932`
  - leaner compact point: `codebook_qer_projattn_top6_r192` at `2.13976149`, total `13,346,786`
- Important negative:
  - the smarter codebook/QER exporter does **not** transfer automatically to the alive `u5k 12x608` branch
  - on that branch, old `fcproj_hi` remains best at `2.10282351`
  - all tested codebook/QER transfer variants were worse and over cap
- Practical public target also moved:
  - README top is still `1.1428`
  - strongest visible open PR claim on `2026-03-21` is `1.1254` with `1.1256` three-seed mean
- Interpretation:
  - our exporter breakthrough is real, but currently `share6`-specific
  - next H100 work should use it only where the structure matches, not as a blind drop-in exporter for every stronger family

## 2026-03-21 codebook+QER breakthrough update
- The active local modern mission is no longer tokenizer-first. It is now the `12x576 share6` codebook+QER line.
- New strongest modern checkpoint:
  - [lab_sp1024_12x576_kv2_share6_codebook_qer192_tail20_lowlr_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x576_kv2_share6_codebook_qer192_tail20_lowlr_m4.txt)
  - clean `3.5989 / 2.1312`
  - shipped under its native compact exporter `3.61341763 / 2.13976149`
  - compressed model `13,257,688` bytes
- Fixed export check on the same checkpoint:
  - [local_sp1024_12x576_share6_codebook_qer_tail20_lowlr_export_check.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_codebook_qer_tail20_lowlr_export_check.md)
  - best absolute legal exporter: `proj_top6_attn_top6_fp16` at `2.13867183`, total `15,169,291`
  - best compact legal exporter: `codebook_qer_projattn_top6_r192` at `2.13976149`, total `13,340,467`
  - strongest smaller middle point: `codebook_qer_projattn_top6_r128` at `2.14009313`, total `11,302,158`
- Important orthogonal negative:
  - [local_sp1024_12x576_share6_quantile_codebook_frontier.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_quantile_codebook_frontier.md)
  - naive tensor-adaptive `quantile16` codebooks are **not** the breakthrough; best measured point `2.14020368` still loses to the normal-codebook frontier
- Important same-line negative:
  - [local_sp1024_12x576_share6_codebook_qer_tail20_ultralow_probe.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_codebook_qer_tail20_ultralow_probe.md)
  - third ultralow continuation regressed to `2.14029037`, so blindly lowering LR again is not the default next move
- Interpretation:
  - the real breakthrough is **normal-codebook int4 + QER + short targeted tails**
  - the compact frontier is now within `~0.0011` shipped `bpb` of the fp16 winner while saving about `1.83MB`
  - this is the most promising local byte-efficiency path in the repo right now

## 2026-03-21 tokenizer sideways + token-flow update
- New focused report: [local_tokenizer_sideways_probe.md](/Users/ronaldschmidt/openai/research/local_tokenizer_sideways_probe.md)
- Sideways tokenizer search is now narrower and cleaner:
  - `u6k` unigram really does save more tokens than `u5k`
    - train `68,350,121`
    - val `42,394,997`
    - about `-6.79%` vs `u4k`
  - but its exact-copy eval-only gate is worse than `u5k`
    - [debug_u6k_12x608_transplant_eval0_exactcopy.txt](/Users/ronaldschmidt/openai/logs/debug_u6k_12x608_transplant_eval0_exactcopy.txt)
    - shipped `2.19966345`
  - local `b4k` BPE is clearly bad on this frozen corpus
    - train `73,388,814`
    - val `45,529,579`
    - slightly *more* tokens than `u4k`
    - [debug_b4k_12x608_transplant_eval0_exactcopy.txt](/Users/ronaldschmidt/openai/logs/debug_b4k_12x608_transplant_eval0_exactcopy.txt)
    - shipped `2.73211769`
- New best tokenizer/token-flow point is now:
  - [lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy.txt](/Users/ronaldschmidt/openai/logs/lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy.txt)
  - clean `4.9851 / 2.1087`
  - shipped `4.97108459 / 2.10282311`
  - compressed model `15,302,484` bytes
- Interpretation:
  - `u5k` exact-copy transfer is still the only alive tokenizer family in this branch
  - `u6k` and `b4k` should not be the next default experiments
  - the next high-EV local token-processing move is on the alive `u5k` branch itself, not another raw tokenizer family

## 2026-03-21 tokenizer-transfer breakthrough update
- New focused report: [local_u5k_exactcopy_tokenizer_probe.md](/Users/ronaldschmidt/openai/research/local_u5k_exactcopy_tokenizer_probe.md)
- The first `u5k` branch looked dead only because the tokenizer transplant was wrong.
- Real token economics on the frozen local corpus:
  - `u4k` train tokens `73,332,949` -> `u5k` train tokens `70,489,732`
  - `u4k` val tokens `45,480,154` -> `u5k` val tokens `43,715,916`
  - about `-3.88%` tokens on both splits
- Real transfer breakthrough: [transplant_tokenizer_checkpoint.py](/Users/ronaldschmidt/openai/research/transplant_tokenizer_checkpoint.py) now exact-copies shared SentencePiece rows before falling back to surface-text averaging.
  - new transplant summary: `3,971` exact piece rows, `1,145` averaged rows, `0` random rows
  - old eval-only gate [debug_u5k_12x608_transplant_eval0.txt](/Users/ronaldschmidt/openai/logs/debug_u5k_12x608_transplant_eval0.txt): shipped `3.46422313`
  - new eval-only gate [debug_u5k_12x608_transplant_eval0_exactcopy.txt](/Users/ronaldschmidt/openai/logs/debug_u5k_12x608_transplant_eval0_exactcopy.txt): shipped `2.17331566`
- First real continuation on the fixed init is a keep:
  - [lab_u5k_12x608_kv2_transplant_lowlr40_fcproj_exactcopy.txt](/Users/ronaldschmidt/openai/logs/lab_u5k_12x608_kv2_transplant_lowlr40_fcproj_exactcopy.txt)
  - clean `4.9819 / 2.1087`
  - shipped `4.97053623 / 2.10387409`
  - compressed model `15,304,828` bytes
- Second ultralow continuation already flattened:
  - [lab_u5k_12x608_kv2_transplant_continue40_ultralow_exactcopy.txt](/Users/ronaldschmidt/openai/logs/lab_u5k_12x608_kv2_transplant_continue40_ultralow_exactcopy.txt)
  - shipped `2.10392536`
- Interpretation:
  - `u5k` is now an alive tokenizer-transfer branch, not a dead idea
  - but it is still behind the locked local legal anchor [dense_u4k_12x608_kv2_lfqat60_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj.txt) at `1.98544845`
  - and the second tail says this exact continuation style is already close to plateau
- That sideways tokenizer pivot has now been tested once in both directions:
  - `u6k` unigram: alive enough to measure, but worse than `u5k`
  - `b4k` BPE: dead on arrival
- Current next highest-EV tokenizer move is therefore token-flow on `u5k`, not another raw tokenizer family.

## 2026-03-21 quantization breakthrough update
- The strongest modern local checkpoint is still [lab_sp1024_12x576_kv2_share6_continue80_ultralowlr_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x576_kv2_share6_continue80_ultralowlr_m4.txt): clean `3.6102 / 2.1378`, shipped `3.6230 / 2.1454`, counted total `15,155,616`.
- The old exporter story on this checkpoint was binary:
  - fp16 keep-float winner [local_sp1024_12x576_share6_continue80_ultralowlr_m4_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_continue80_ultralowlr_m4_targeted_export_check.md): `proj_top6_attn_top6_fp16` at `2.14541706`
  - plain int8 control: `2.24681880`
- New branch: low-rank reconstruction export, implemented in [quant_reconstruction.py](/Users/ronaldschmidt/openai/quant_reconstruction.py) and wired into [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py), [train_gpt_mlx.py](/Users/ronaldschmidt/openai/train_gpt_mlx.py), and [search_export_gap.py](/Users/ronaldschmidt/openai/research/search_export_gap.py).
- New measured frontier on the same checkpoint:
  - [local_sp1024_12x576_share6_qer_breakthrough.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_qer_breakthrough.md)
  - [local_sp1024_12x576_share6_qer_highrank.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_qer_highrank.md)
  - `qer_projattn_top6_r128`: `2.19538766` shipped at `9,442,835`
  - `qer_projattn_top6_r192`: `2.17947545` shipped at `11,484,305`
  - `qer_projattn_top6_r256`: `2.16664007` shipped at `13,526,942`
  - `qer_projattn_top6_r320`: `2.15800706` shipped at `15,571,747`
- Interpretation: QER does not yet beat the fp16-keep winner on absolute shipped `val_bpb`, but it creates the first strong legal middle frontier between fp16 keeps and `int8_all`. `r320` is only `0.01259` bpb behind the fp16 baseline while using a completely different, reconstruction-aware exporter.
- Important bookkeeping fix: counted code now includes the new shared helper, so export-gap totals are honest. The search/report paths now count [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) + [quant_reconstruction.py](/Users/ronaldschmidt/openai/quant_reconstruction.py).
- Current next highest-EV move is **not** another same-family training continuation. It is one near-cap hybrid exporter on top of `qer_projattn_top6_r320`, for example a tiny fp16 boundary keep to close the last `~0.01` bpb gap.

## 2026-03-21 local `u4k` update
- The best measured local legal anchor is still [dense_u4k_12x608_kv2_lfqat60_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj.txt): clean `4.5577 / 1.9878`, shipped `4.5523 / 1.9854`, counted total `14,796,847`.
- The strongest near-cap local control is still [dense_u4k_14x576_kv2_boot_from_12x576_adapt150.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_14x576_kv2_boot_from_12x576_adapt150.txt): clean `4.5379 / 1.9792`, shipped `4.5541 / 1.9862`, counted total `15,180,590`.
- The apparent MLX end-of-run “crash” on the new `u4k` continuations was not a model bug. It was a full local disk. After freeing space, the same recipes finalized cleanly and produced real metrics.
- Both new local continuation ideas are now refuted:
  - [lab_u4k_12x608_kv2_lfqat60_reramp_continue40_vb64k_retry1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_lfqat60_reramp_continue40_vb64k_retry1.txt): clean `4.6530 / 2.0294`, shipped `4.6349 / 2.0215`, total `14,807,218`.
  - [lab_u4k_14x576_kv2_ultralow_continue20_retry1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_14x576_kv2_ultralow_continue20_retry1.txt): clean `4.6226 / 2.0161`, shipped `4.6043 / 2.0081`, total `15,187,412`.
- Operationally important fix: the local launchers [train_lab_u4k_12x608_lfqat_reramp_continue60_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_12x608_lfqat_reramp_continue60_m4.sh) and [train_lab_u4k_14x576_ultralow_continue20_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_14x576_ultralow_continue20_m4.sh) now preflight free disk space before launching a long run.
- Current conclusion: naive local tails on the two strongest legal `u4k` families are low-EV. The next local move should be structural or tokenizer-aware, not another same-family continuation.

## Latest shared-depth branch
- The new best measured local autoresearch point is [lab_sp1024_12x576_kv2_share6_continue80_ultralowlr_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x576_kv2_share6_continue80_ultralowlr_m4.txt): clean `3.6102 / 2.1378`, shipped `3.6230 / 2.1454`, counted total `15,155,616`.
- The immediate predecessor [lab_sp1024_12x576_kv2_share6_continue80_lowlr_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x576_kv2_share6_continue80_lowlr_m4.txt) reached shipped `2.1770`, and the old `12x560` leader [lab_sp1024_12x560_kv2_share6_continue80_femtotail_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x560_kv2_share6_continue80_femtotail_m4.txt) reached shipped `2.1865`.
- The targeted conservative frontier checks on the new `12x576` checkpoints are [local_sp1024_12x576_share6_continue80_lowlr_m4_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_continue80_lowlr_m4_targeted_export_check.md) and [local_sp1024_12x576_share6_continue80_ultralowlr_m4_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_continue80_ultralowlr_m4_targeted_export_check.md). Both keep `proj_top6_attn_top6_fp16` on top.
- This means the local main line has now moved from `12x560 share6` to `12x576 share6`, with two straight keeps after the widened branch boot and no exporter change.
- The strongest new structural family is now [lab_sp1024_12x544_kv2_share6_continue120_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x544_kv2_share6_continue120_m4.txt): clean `4.1869 / 2.4793`, shipped `4.1907 / 2.4816`, counted total `13,507,819`.
- Its scratch parent [lab_sp1024_12x544_kv2_share6_896ctx_modern_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x544_kv2_share6_896ctx_modern_m4.txt) came in at clean `4.3357 / 2.5675`, shipped `4.3394 / 2.5697`, total `13,467,285`.
- The fixed export sweeps on both shared-depth checkpoints are:
  - [local_sp1024_12x544_share6_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x544_share6_m4_export_gap.md)
  - [local_sp1024_12x544_share6_continue120_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x544_share6_continue120_m4_export_gap.md)
- Both sweeps land on the same winner: `proj_top6_attn_top6_fp16`. The important result is that the branch's export gap is already almost zero (`~0.0022-0.0023` bpb), so shared depth does not need more export surgery first.
- The old `10x560` local anchor [lab_sp1024_10x560_joint_boundary_recovery_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_10x560_joint_boundary_recovery_m4.txt) at shipped `2.2710` has now been beaten locally.
- The next highest-leverage local move is now one bounded `12x576 share6` nanotail from the new ultralow anchor, not another exporter search.

## Latest autoresearch loop
- Added a repo-specific [program.md](/Users/ronaldschmidt/openai/program.md) that ports the useful `autoresearch` discipline into Parameter Golf: read the current measured frontier first, generate a small next-experiment loop, run only high-EV candidates on the correct hardware tier, and keep only measured improvements.
- Added [research/autoresearch_loop.py](/Users/ronaldschmidt/openai/research/autoresearch_loop.py), which combines the current experiment table, the calibrated seeker output, and the fixed-settings export-gap search into a concrete next-experiment queue with exact commands plus keep/kill rules. It writes [research/autoresearch_loop.md](/Users/ronaldschmidt/openai/research/autoresearch_loop.md) and [research/autoresearch_loop.json](/Users/ronaldschmidt/openai/research/autoresearch_loop.json).
- The current local autoresearch winner is [lab_sp1024_12x576_kv2_share6_continue80_ultralowlr_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x576_kv2_share6_continue80_ultralowlr_m4.txt): `3.6102 / 2.1378` clean and `3.6230 / 2.1454` shipped at `15,155,616` counted bytes.
- The preceding [lab_sp1024_12x576_kv2_share6_continue80_lowlr_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x576_kv2_share6_continue80_lowlr_m4.txt) also kept strongly at `2.1770` shipped, so the widened branch improved again immediately after becoming the new winner.
- The exporter on this branch is still stable: [local_sp1024_12x576_share6_continue80_lowlr_m4_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_continue80_lowlr_m4_targeted_export_check.md) and [local_sp1024_12x576_share6_continue80_ultralowlr_m4_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_sp1024_12x576_share6_continue80_ultralowlr_m4_targeted_export_check.md) both keep `proj_top6_attn_top6_fp16` on top.
- The older `10x560` recovery work below is now historical context, not the current main line.
- The raw-shape research-lab probes both failed fast: [lab_sp1024_10x576_kv2_896ctx_compiled_maxtrain_baseline_export_b128_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_10x576_kv2_896ctx_compiled_maxtrain_baseline_export_b128_m4.txt) finished at `2.6594` clean / `2.9861` shipped, and [lab_sp1024_11x560_kv2_896ctx_compiled_maxtrain_baseline_export_b128_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_11x560_kv2_896ctx_compiled_maxtrain_baseline_export_b128_m4.txt) was killed early at `step 40` because it was already slightly worse than `10x576` while slower.
- The fixed export sweep on the new best checkpoint is [local_sp1024_10x560_joint_boundary_recovery_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_joint_boundary_recovery_m4_export_gap.md). It reconfirms `projhi_attnhi_fp16` as the best legal export at `2.27104283` shipped and `15,570,847` total bytes. The tiny illegal frontier remains [local_sp1024_10x560_m4__9123b7ed3c__proj_top5_attn_top6_fp16.txt](/Users/ronaldschmidt/openai/logs/export_gap_local_sp1024_10x560_seq896/local_sp1024_10x560_m4__9123b7ed3c__proj_top5_attn_top6_fp16.txt) at `2.27174028` and `16,087,789` total.
- The new local autoresearch winner is [autoresearch_sp1024_10x560_hybrid_int8aware_block4proj_keptfp16_80.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_hybrid_int8aware_block4proj_keptfp16_80.txt): `3.5638 / 2.1104` clean and `3.8371 / 2.2722` shipped at `15,570,270` counted bytes.
- This is the first successful `int8-aware` keep on the `10x560` line. The pure block4-only int8-aware tail in [autoresearch_sp1024_10x560_keep_anchor_int8aware_block4proj120.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_keep_anchor_int8aware_block4proj120.txt) was slightly negative on shipped (`2.2734`), but the hybrid version that co-trains `blocks.4.mlp.proj.weight` with the proven fp16-kept recovery corridor turns that into a small real gain.
- The gain is tiny: only about `0.000285` shipped bpb better than the prior anchor. So it is a real keep, but another clear near-saturation point, not evidence that we should start another blind continuation chain on the same settings.
- I also expanded the near-cap export sweep around the current winner. No new legal policy beat `projhi_attnhi_fp16`: [local_sp1024_10x560_m4__c9049b3d1e__proj_top5_attn_top6_fp16.txt](/Users/ronaldschmidt/openai/logs/export_gap_local_sp1024_10x560_seq896/local_sp1024_10x560_m4__c9049b3d1e__proj_top5_attn_top6_fp16.txt) reached `2.2720` shipped but missed legality at `16,086,547` total, while [local_sp1024_10x560_m4__c9049b3d1e__proj_top6_attn_top5_fp16.txt](/Users/ronaldschmidt/openai/logs/export_gap_local_sp1024_10x560_seq896/local_sp1024_10x560_m4__c9049b3d1e__proj_top6_attn_top5_fp16.txt) was much better at `2.2626` but too far over cap at `16,629,949` total.
- To support that hybrid run, [train_gpt_mlx.py](/Users/ronaldschmidt/openai/train_gpt_mlx.py) now has a local-only `int8-aware` alignment path for targeted recovery tails. It aligns named tensors to the actual `int8_clean_per_row_v1` dequantized target, and it automatically disables the compiled training path for that special mode so the MLX run stays stable.
- This third fp16-kept recovery tail is still a real keep, but only by about `0.0010` shipped bpb. I updated the loop so it now treats sub-`0.001` shipped gains as near-saturation and stops blindly proposing the same identical continuation forever.
- The latest local autoresearch winner is [autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120_continue80_keptfp16.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120_continue80_keptfp16.txt): `3.5708 / 2.1145` clean and `3.8392 / 2.2735` shipped at `15,570,260` counted bytes.
- The second fp16-kept recovery tail also paid off. It uses the same restricted update set (`blocks.5-9.{attn.proj,mlp.proj}.weight`) with a smaller `0.00015` matrix LR and bought another shipped gain from `2.2745 -> 2.2735`.
- The new local autoresearch winner is [autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail_keptfp16_recovery120.txt): `3.5758 / 2.1175` clean and `3.8410 / 2.2745` shipped at `15,570,613` counted bytes.
- This is the first successful `gap recovery` keep on the saturated `10x560` line. Instead of another identical tail, it starts from the cleaner-but-gap-worse checkpoint and trains only the tensors already kept in fp16 (`blocks.5-9.{attn.proj,mlp.proj}.weight`). That bought back another real shipped gain from `2.2821 -> 2.2745`.
- The continuation chain is still alive after a fourth keep. [autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail.txt) reached `3.6495 / 2.1611` clean and `3.8538 / 2.2821` shipped at `15,538,191` counted bytes.
- The `10x560` continuation chain is still alive. A third continuation tail [autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr.txt) pushed the same family down to `3.7495 / 2.2203` clean and `3.9092 / 2.3149` shipped at `15,504,049` counted bytes.
- The fixed export sweep on the new microtail checkpoint is in [local_sp1024_10x560_continue120_microtail_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_continue120_microtail_m4_export_gap.md). It again reconfirms `projhi_attnhi_fp16` as the best legal policy, so the latest gain is again a real checkpoint improvement rather than a changed exporter.
- The fixed export sweep on the new recovery checkpoint is in [local_sp1024_10x560_keptfp16_recovery120_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_keptfp16_recovery120_m4_export_gap.md). It again keeps `projhi_attnhi_fp16` on top, now at `2.27454986` shipped with `0.15704986` export gap and `15,570,613` total bytes.
- The fixed export sweep on the follow-up recovery checkpoint is in [local_sp1024_10x560_keptfp16_recovery120_continue80_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_keptfp16_recovery120_continue80_m4_export_gap.md). The same policy still wins and moves to `2.27347403` shipped with `0.15897403` export gap and `15,570,260` total bytes.
- The fixed export sweep on the third recovery checkpoint is in [local_sp1024_10x560_keptfp16_recovery120_continue80b_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_keptfp16_recovery120_continue80b_m4_export_gap.md). The same policy still wins at `2.27250127` shipped with `0.16010127` export gap and `15,569,626` total bytes.
- The fixed export sweep on that new checkpoint is in [local_sp1024_10x560_continue200_ultralowlr_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_continue200_ultralowlr_m4_export_gap.md). It reconfirms `projhi_attnhi_fp16` as the best legal policy on the stronger checkpoint too, so this improvement is real model progress rather than a one-off export accident.
- The `10x560` default launchers now use the measured lower-gap export policy instead of the overly aggressive top-layer int4 default. Both [scripts/local/train_seeker_sp1024_best_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_seeker_sp1024_best_m4.sh) and [scripts/runpod/train_seeker_sp1024_best_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_seeker_sp1024_best_8xh100.sh) now default to `projhi_attnhi_fp16`: `int8` everywhere, fp16 only on upper `mlp.proj` and `attn.proj`, no default `int4`.
- [research/sota_seeker.py](/Users/ronaldschmidt/openai/research/sota_seeker.py) now ingests the measured [local_sp1024_10x560_block4attn_m4_export_gap.json](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_block4attn_m4_export_gap.json) calibration instead of the older recovery checkpoint. The `projhi_attnhi_fp16` export spec is wired into the search space and gets a measured shipping-gap override when the candidate matches the calibrated `10x560 / seq=896` regime.

## Latest compile-start crash fix
- The `8xH100` compile-start crash is now root-caused and fixed in [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py). The immediate Dynamo failure was not the export path; it was [Rotary.forward](/Users/ronaldschmidt/openai/train_gpt.py) calling `torch.is_inference_mode_enabled()` inside the compiled fullgraph path. A tiny local fullgraph repro now passes again, and the regression is covered by [research/test_train_gpt_compile.py](/Users/ronaldschmidt/openai/research/test_train_gpt_compile.py).
- The RoPE cache is now only persisted in training mode, which keeps the original safety property (eval must not poison the training cache with inference tensors) without calling unsupported runtime-state introspection from the compiled graph.
- [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) also now respects explicit compile disable flags via `DISABLE_COMPILE` / `TORCHDYNAMO_DISABLE` before compiling either the model or the Muon Newton-Schulz helper. That fixes the earlier mismatch where the “safe fallback” still logged `compile:True`.
- [scripts/runpod/train_seeker_sp1024_best_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_seeker_sp1024_best_8xh100.sh) now treats empty export overrides as real overrides instead of silently restoring the old default top-layer `int4` policy.

## Latest fixed-settings export-gap search
- Added [research/search_export_gap.py](/Users/ronaldschmidt/openai/research/search_export_gap.py), a fixed-checkpoint export-policy searcher that holds architecture, tokenizer, `seq_len`, `QK_GAIN_INIT`, `ROPE_BASE`, and `LOGIT_SOFTCAP` constant and reranks only the legal shipping policy using real `ITERATIONS=0` roundtrip evals. The fixed-settings env builder is covered by [research/test_search_export_gap.py](/Users/ronaldschmidt/openai/research/test_search_export_gap.py), so `seq_len=896` and the other model-shaping knobs cannot silently drift again.
- Fixed a real cache bug in [research/search_export_gap.py](/Users/ronaldschmidt/openai/research/search_export_gap.py): sweep logs are now checkpoint-specific instead of being keyed only by preset name and policy. Before this fix, a new checkpoint could silently reuse stale export logs from an older checkpoint. The new regression is covered in [research/test_search_export_gap.py](/Users/ronaldschmidt/openai/research/test_search_export_gap.py).
- On the current local `sp1024 10x560` checkpoint `local_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_compiled_longtail_fchi_top4_attn_top4_m4_mlx_model.npz`, the best legal policy is now `projhi_attnhi_fp16`, not the current mixed-int4 top-4 policy. It keeps the model fully `int8` except for upper `mlp.proj` and `attn.proj` weights in fp16 and reaches `val_bpb=2.67617117` shipped with only `0.09037117` export gap at `15,036,838` counted bytes.
- The current reference export `fchi_top4_attn_top4_fp16` is clearly too aggressive for this checkpoint: `2.78245296` shipped, `0.19665296` export gap, `7,629,350` counted bytes. So the fixed-settings search bought back `0.10628179` shipped bpb and the same `0.10628179` gap reduction by spending an extra `7,407,488` bytes that were still available under the `16MB` cap. Full ranked output is in [research/local_sp1024_10x560_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_m4_export_gap.md).
- On the stronger `continue200_ultralowlr` checkpoint, the same policy still wins in [research/local_sp1024_10x560_continue200_ultralowlr_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_continue200_ultralowlr_m4_export_gap.md): `2.31488601` shipped at `15,504,049` total with `0.09458601` export gap. The older aggressive top-4 int4 reference is still worse at `2.39279421`.
- On the still stronger `continue120_microtail` checkpoint, the same policy wins again in [research/local_sp1024_10x560_continue120_microtail_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_continue120_microtail_m4_export_gap.md): `2.28212046` shipped at `15,538,191` total with `0.12102046` export gap. The aggressive top-4 int4 reference is again clearly worse at `2.37250881`.

## Latest 8xH100 training-path hardening
- [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) is now tuned for better useful work under fixed `8xH100 / 600s` runs: default `step 0` full-val is skipped unless explicitly re-enabled, `__never__` QAT sentinels no longer disable compile by accident, validation batching is no longer tied to `GRAD_ACCUM_STEPS`, the wallclock-cap all-reduce can be throttled with `WALLCLOCK_SYNC_EVERY`, and the distributed loader now advances only the local rank span instead of materializing the full cross-rank chunk before slicing.
- The same path now supports opt-in `DDP_STATIC_GRAPH` and `DDP_GRADIENT_AS_BUCKET_VIEW`; the active compiled `sp1024` launcher enables both, along with `VAL_AT_STEP_ZERO=0` and `WALLCLOCK_SYNC_EVERY=16`.
- [research/sota_seeker.py](/Users/ronaldschmidt/openai/research/sota_seeker.py) and [scripts/runpod/train_seeker_sp1024_best_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_seeker_sp1024_best_8xh100.sh) were updated so the searcher and the default `8xH100` launcher emit the hardened settings automatically for the current best official-ready candidate.

## Latest official sp1024 probe
- The first cheap `1xH100` probe for the offline seeker winner OOMed immediately because the generated single-GPU command kept the full seeker global batch at `GRAD_ACCUM_STEPS=1`.
- I fixed that in [research/sota_seeker.py](/Users/ronaldschmidt/openai/research/sota_seeker.py): single-GPU probe commands now add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and automatically raise `GRAD_ACCUM_STEPS` when the requested batch exceeds the known-safe `524,288`-token anchor.
- The corrected official `1xH100` run `seeker_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi_ga2_retry1` completed on the real `fineweb10B_sp1024` path at `600.6s` train time and reached `val_loss=3.2726`, `val_bpb=1.9382` clean, then `val_loss=4.0585`, `val_bpb=2.4037` after the legal int8 roundtrip.
- That real anchor was then fed back into the seeker and used to re-rank the full `sp1024` search space. The first official `8xH100 / 600s` seeker run `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` finished at `val_loss=2.3365`, `val_bpb=1.3838` clean and `val_loss=2.5505`, `val_bpb=1.5106` post-quant, with `12,060,447` compressed-model bytes and `12,126,772` counted total bytes.
- The seeker now treats that `8xH100` result as the primary calibration anchor. After another loop focused on slightly longer compile-friendly tails, finer `896`-context throughput points, and tighter top-of-stack hybrid exports, the best objective-ranked official-ready candidate is now `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16`, estimated at `1.1614` shipped bpb, `15,997,981` total bytes, and runtime ratio `0.993`.
- The same loop also found the lowest estimated shipped official-ready candidate in the current space: `seeker_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16` at `1.1594` shipped bpb, `15,757,981` total bytes, and runtime ratio `0.994`. That is the new default Runpod launcher target because it optimizes the headline metric directly while staying legal and under budget.

## Experiment table
| Run | Hypothesis | Key config | Runtime | Artifact bytes | Val loss / val_bpb | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| `seeker_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi_ga2_retry1` | The new offline seeker winner should be re-tested on real official `sp1024` data once the `1xH100` probe command is made memory-safe. | Official `fineweb10B_sp1024`, `1xH100`, `10x512 KV2`, `seq=896`, `train_batch_tokens=573,440`, `GRAD_ACCUM_STEPS=2`, Muon `0.97`, `MAX_WALLCLOCK_SECONDS=600`, export=`int8_attnhi`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. | `600.6s` train + `29.8s` roundtrip eval | `10,566,413 total` | `3.2726 / 1.9382` clean at stop, `4.0585 / 2.4037` roundtrip | First completed official `sp1024` seeker probe. Stable and legal on `1xH100`, but still much too weak to be a serious `1.1` path. |
| `seeker_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` | The top offline seeker candidate may run as-is on a cheap `1xH100` probe. | Official `fineweb10B_sp1024`, `1xH100`, `10x512 KV2`, `seq=896`, `train_batch_tokens=573,440`, original seeker-generated `GRAD_ACCUM_STEPS=1`, export=`int8_attnhi`. | Failed on first training forward | N/A | `6.9390 / 4.1097` initial val before train, then CUDA OOM | Important negative launch result. The original single-GPU probe command was not memory-safe and forced a fix in the seeker command generator. |
| `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` | The official U4K/H100 path may simply be massively undertrained; a long continuation from the first real CUDA checkpoint could reveal whether model quality or export is the real blocker. | Official `fineweb10B_spu4096_docs`, `1xH100`, `12x608 KV2`, start from `runpod_dense_u4k_12x608_lfqat_1xh100_20260319_rerun1_final_model.pt`, `2485` steps under `MAX_WALLCLOCK_SECONDS=4800`, cosine tail `0.002/0.0015/0.0015`, full-val every `100`. | `4801.1s` train + `38.6s` roundtrip eval | `10,842,824 total` | `3.0683 / 1.3329` clean at stop, `3.7062 / 1.6101` roundtrip | This is the new decisive result. The model itself became much stronger on official data, but the quantized export still destroys about `0.2772` bpb. |
| `runpod_dense_u4k_12x608_lfqat_1xh100_20260319_rerun1` | The local `12x608` LFQAT winner may survive a first real CUDA run on the official Raw-Docs/U4K path. | Official `fineweb10B_spu4096_docs`, `1xH100`, `12x608 KV2`, `60` LFQAT steps, legal export=`fc_hi + proj_hi`. | `119.4s` train + `38.6s` roundtrip eval | `7,429,251 total` | `5.2403 / 2.2765` clean, `5.4762 / 2.3790` roundtrip | First real official-path CUDA anchor. Much weaker than leaderboard level by itself, but it proved the path runs and justified the long continuation. |
| `clean_gate_dense_u4k_12x608_lfqat60_fcproj_1m` | The new LFQAT winner must survive a meaningfully larger gate. | `12x608 KV2`, legal export=`fc_hi + proj_hi`, `VAL_MAX_TOKENS=1,048,576`, checkpoint=`dense_u4k_12x608_kv2_lfqat60_fcproj`. | `82.0s` clean eval + `84.3s` roundtrip eval | `14,796,847 total` | `4.5593 / 2.0041` | Best completed longer-gate local result. |
| `dense_u4k_12x608_kv2_lfqat60_fcproj` | Fisher- and KL-guided LFQAT on the legal export family can beat both raw continuation and earlier post-hoc-only legalizers. | Start from raw `12x608 adapt150`, `60` steps, constant LR `0.004/0.003/0.003`, legal export=`fc_hi + proj_hi`, `LFQAT_KL_WEIGHT=0.05`, `LFQAT_FISHER_WEIGHT=0.01`, `TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015`, ramp `qat_prob=0.15->1.0` by step `30`. | `330.8s` train + `14.3s` roundtrip eval | `14,796,847 total` | `4.5523 / 1.9854` | Current best legal `262k` local result and first local dense result under `1.99`. |
| `dense_u4k_13x576_boot_continue60_fcproj_lr5` | The mathematically attractive `13x576` shape may not need a heavy adaptation; a gentle low-LR continuation from the expanded boot checkpoint could already beat the boot export cleanly. | Expand `12x576` -> `13x576`, legal export=`fc_hi + proj_hi` on blocks `6-12`, then `60` constant-LR steps `0.006/0.005/0.005`. | `346.1s` train + `18.6s` roundtrip eval | `13,921,399 total` | `4.6304 / 2.0195` | Best completed `13x576` result so far. Real positive gain over the legal boot probe, but still behind `12x608`. |
| `dense_u4k_14x576_kv2_boot_from_12x576_adapt150` | If the bought-capacity thesis is real, a deeper `14x576` dense model should at least tie the `12x608` line while staying under the byte cap. | Expand `12x576` -> `14x576`, adapt `150` cosine steps, legal export=`fc_hi + proj_hi` on blocks `7-13`. | `522.3s` train + `54.7s` roundtrip eval | `15,180,590 total` | `4.5541 / 1.9862` | Important near-cap control. Legal and almost tied with the winner, so the family is not obviously dead. |
| `dense_u4k_14x576_kv2_qat60_fcproj` | If `14x576` is truly the next step up, a simple compile-stable `QAT + compression-aware` tail should push it past `12x608`. | Start from raw `14x576 adapt150`, `60` constant-LR steps `0.006/0.0045/0.0045`, legal export=`fc_hi + proj_hi`, `TRAIN_QAT=fc_hi+proj_hi`, `TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015`. | `214.8s` train + `19.8s` roundtrip eval | `15,223,725 total` | `4.8062 / 2.0962` | Clear negative result. This simple post-adapt QAT tail hurts the larger model badly. |
| `dense_u4k_13x576_kv2_lfqat60_fcproj` | If `13x576` is truly the clean byte-budget sweet spot, a better warm start plus the same LFQAT recipe as the `12x608` winner should keep it competitive. | Start from `13x576 boot_continue60`, `60` LFQAT steps, constant LR `0.004/0.003/0.003`, legal export=`fc_hi + proj_hi` on blocks `6-12`. | `377.0s` train + `17.7s` roundtrip eval | `13,954,254 total` | `4.7119 / 2.0550` | Completed negative result. Even the stronger `13x576` LFQAT continuation stays clearly behind the `12x608` LFQAT winner. |
| `dense_u4k_12x608_plain_continue60_fcproj` | A simple low-LR tail on the raw `12x608` checkpoint may still beat the old leader without any extra compression-aware objective. | Start from raw `12x608 adapt150`, `60` steps, constant LR `0.006/0.005/0.005`, no QAT, no compression-aware loss, legal export=`fc_hi + proj_hi`. | `173.0s` train + `14.4s` roundtrip eval | `14,813,203 total` | `4.6017 / 2.0070` | Strong positive control, but worse than LFQAT on the same family. |

## Current best candidate
The strongest measured shipped official result in the repo is still the official-path `8xH100 / 600s` `sp1024` run `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` at `1.5106` roundtrip. The active forward bet is now the new low-shipped seeker candidate `seeker_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16`: a `10x560 KV2` dense `sp1024` model with a compile-friendly long-tail schedule, slightly denser `896`-context throughput, `fc_top4` int4, and `attn_top4` fp16 retention. It is still estimated rather than measured, but it is the first official-ready candidate in the current search space to reach `1.1594` shipped bpb and `1.1473` best-case.

## What changed in this cycle
- Added a real official `sp1024` `1xH100` probe for the offline seeker winner and saved its full log plus both checkpoint artifacts locally.
- Found and fixed a real seeker bug in [research/sota_seeker.py](/Users/ronaldschmidt/openai/research/sota_seeker.py): the generated `screen_command` could emit memory-unsafe `1xH100` launches for large batch geometries.
- The single-GPU seeker path now automatically adds `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and upshifts `GRAD_ACCUM_STEPS` when `TRAIN_BATCH_TOKENS` exceed the documented safe `1xH100` anchor.
- Regenerated [research/sota_seeker.md](/Users/ronaldschmidt/openai/research/sota_seeker.md) and [research/sota_seeker.json](/Users/ronaldschmidt/openai/research/sota_seeker.json) with the corrected probe command, and added a regression test in [research/test_sota_seeker.py](/Users/ronaldschmidt/openai/research/test_sota_seeker.py).
- Implemented `LFQAT-lite` in [train_gpt_mlx.py](/Users/ronaldschmidt/openai/train_gpt_mlx.py): ramped fake quantization, KL-to-full-precision teacher, Fisher EMA weighting, and optional uncompiled training path when dynamic LFQAT state is active.
- Fixed the local pattern mismatch mistake by switching all serious `fc_hi + proj_hi` runs to explicit tensor-name lists instead of the invalid shorthand `blocks.6-11.*`.
- Verified on a `64k` control gate that the new LFQAT objective improves the same legal export family over the raw `12x608` baseline before spending a full `262k` run.
- Ran the first real `12x608` LFQAT continuation and established the new local leader at `1.9854` bpb.
- Ran a direct plain low-LR tail on the same raw `12x608` checkpoint and showed it is genuinely positive but weaker than LFQAT (`2.0070` vs `1.9854`).
- Ran a lower-LR full-quant LFQAT polish from the new winner and showed it is negative (`2.0232`), so the first LFQAT jump is the useful move, not indefinite all-quant continuation.
- Added a `1M` gate for the LFQAT winner and confirmed the gain survives on a larger slice (`2.0041`).
- Added a safer depth-expansion path in [research/expand_mlx_checkpoint.py](/Users/ronaldschmidt/openai/research/expand_mlx_checkpoint.py) so deeper warm-starts copy the last compatible block/control tensors instead of leaving random new-depth state.
- Probed `13x576` directly. The legal boot comparison showed `fc_hi + proj_hi` beats `uppercombo` (`2.1082` vs `2.1125`), a plain `continue60` warm start improved that line to `2.0195`, and the full `13x576` LFQAT continuation still only reached `2.0550`. That is enough evidence to kill `13x576` locally.
- Promoted `14x576` from a forgotten side run into the active evidence set: the raw adapted checkpoint is legal and nearly tied with the current leader at `1.9862`, but a straightforward `QAT + compression-aware` continuation on the same legal export family collapsed to `2.0962`.

## Current blocker
The blocker is now honest calibration of the raw-quality frontier under the real `8xH100 / 600s` budget. The first official `8xH100` seeker run proved the branch can train to `1.3838` clean and `1.5106` shipped, which is much better than the `1xH100` probe and close enough to matter. But it still misses the public `1.2244` shipped anchor by a large margin, and the run under-used the full budget because `ITERATIONS=2500` stopped it after about `428.9s`. So the next bottleneck is not blind local search any more: it is using the new `8xH100` anchor to pick a stronger official-ready `sp1024` candidate that both spends the budget better and keeps the shipped gap under control.

## Top next moves
1. Run `seeker_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16` on the real `8xH100 / 600s` path. It is now the lowest-estimated shipped official-ready candidate and the default launcher target in [scripts/runpod/train_seeker_sp1024_best_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_seeker_sp1024_best_8xh100.sh).
2. If the `10x560` run misses badly, fall back one notch to the more conservative objective-ranked `10x544` hybrids, especially `fcproj_top5_attn_top5_fp16`, `fcproj_top4_attn_top4_fp16`, and `fchi_top4_attn_top4_fp16`, instead of revisiting the clearly weak long-context line.
3. Keep the U4K `12x608` export-sweep path alive as a measured backup. It is still the strongest already-shipped official non-`sp1024` model in the repo and remains the most grounded fallback if the `sp1024` branch stalls.

## Exact files changed this cycle
- [train_gpt_mlx.py](/Users/ronaldschmidt/openai/train_gpt_mlx.py)
- [research/expand_mlx_checkpoint.py](/Users/ronaldschmidt/openai/research/expand_mlx_checkpoint.py)
- [research/experiment_table.md](/Users/ronaldschmidt/openai/research/experiment_table.md)
- [HANDOFF.md](/Users/ronaldschmidt/openai/HANDOFF.md)
- [logs/dense_u4k_13x576_boot_from_12x576_continue_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_from_12x576_continue_mlx_model.npz)
- [logs/dense_u4k_13x576_boot_eval0_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_fcproj.txt)
- [logs/dense_u4k_13x576_boot_eval0_fcproj_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_fcproj_mlx_model.npz)
- [logs/dense_u4k_13x576_boot_eval0_uppercombo.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_uppercombo.txt)
- [logs/dense_u4k_13x576_boot_eval0_uppercombo_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_uppercombo_mlx_model.npz)
- [logs/dense_u4k_13x576_boot_eval0_fchi.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_fchi.txt)
- [logs/dense_u4k_13x576_boot_eval0_fchi_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_fchi_mlx_model.npz)
- [logs/dense_u4k_13x576_boot_eval0_fchi_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_fchi_mlx_model.int8.ptz)
- [logs/dense_u4k_13x576_boot_eval0_projhi.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_projhi.txt)
- [logs/dense_u4k_13x576_boot_eval0_projhi_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_projhi_mlx_model.npz)
- [logs/dense_u4k_13x576_boot_eval0_projhi_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_eval0_projhi_mlx_model.int8.ptz)
- [logs/dense_u4k_13x576_boot_continue60_fcproj_lr5.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_continue60_fcproj_lr5.txt)
- [logs/dense_u4k_13x576_boot_continue60_fcproj_lr5_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_continue60_fcproj_lr5_mlx_model.npz)
- [logs/dense_u4k_13x576_boot_continue60_fcproj_lr5_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_boot_continue60_fcproj_lr5_mlx_model.int8.ptz)
- [logs/dense_u4k_13x576_kv2_m2_boot_from_12x576_continue_adapt150_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_kv2_m2_boot_from_12x576_continue_adapt150_fcproj.txt)
- [logs/dense_u4k_13x576_kv2_m2_boot_from_12x576_continue_adapt150_fcproj_gate262k.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_kv2_m2_boot_from_12x576_continue_adapt150_fcproj_gate262k.txt)
- [logs/dense_u4k_13x576_kv2_lfqat60_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_kv2_lfqat60_fcproj.txt)
- [logs/dense_u4k_13x576_kv2_lfqat60_fcproj_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_kv2_lfqat60_fcproj_mlx_model.npz)
- [logs/dense_u4k_13x576_kv2_lfqat60_fcproj_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_13x576_kv2_lfqat60_fcproj_mlx_model.int8.ptz)
- [logs/dense_u4k_14x576_kv2_qat60_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_14x576_kv2_qat60_fcproj.txt)
- [logs/dense_u4k_14x576_kv2_qat60_fcproj_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_14x576_kv2_qat60_fcproj_mlx_model.npz)
- [logs/dense_u4k_14x576_kv2_qat60_fcproj_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_14x576_kv2_qat60_fcproj_mlx_model.int8.ptz)
- [logs/dense_u4k_12x608_lfqat_smoke5.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_lfqat_smoke5.txt)
- [logs/dense_u4k_12x608_lfqat_smoke5_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_lfqat_smoke5_mlx_model.npz)
- [logs/dense_u4k_12x608_lfqat_smoke5_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_lfqat_smoke5_mlx_model.int8.ptz)
- [logs/dense_u4k_12x608_lfqat_smoke5_explicit.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_lfqat_smoke5_explicit.txt)
- [logs/dense_u4k_12x608_lfqat_smoke5_explicit_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_lfqat_smoke5_explicit_mlx_model.npz)
- [logs/dense_u4k_12x608_lfqat_smoke5_explicit_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_lfqat_smoke5_explicit_mlx_model.int8.ptz)
- [logs/dense_u4k_12x608_fcproj_gate64k_base.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_fcproj_gate64k_base.txt)
- [logs/dense_u4k_12x608_fcproj_gate64k_base_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_fcproj_gate64k_base_mlx_model.npz)
- [logs/dense_u4k_12x608_fcproj_gate64k_base_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_fcproj_gate64k_base_mlx_model.int8.ptz)
- [logs/dense_u4k_12x608_kv2_lfqat60_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj.txt)
- [logs/dense_u4k_12x608_kv2_lfqat60_fcproj_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj_mlx_model.npz)
- [logs/dense_u4k_12x608_kv2_lfqat60_fcproj_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj_mlx_model.int8.ptz)
- [logs/dense_u4k_12x608_kv2_lfqat60_fcproj_polish40.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj_polish40.txt)
- [logs/dense_u4k_12x608_kv2_lfqat60_fcproj_polish40_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj_polish40_mlx_model.npz)
- [logs/dense_u4k_12x608_kv2_lfqat60_fcproj_polish40_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_kv2_lfqat60_fcproj_polish40_mlx_model.int8.ptz)
- [logs/dense_u4k_12x608_plain_continue60_fcproj.txt](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_plain_continue60_fcproj.txt)
- [logs/dense_u4k_12x608_plain_continue60_fcproj_mlx_model.npz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_plain_continue60_fcproj_mlx_model.npz)
- [logs/dense_u4k_12x608_plain_continue60_fcproj_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/dense_u4k_12x608_plain_continue60_fcproj_mlx_model.int8.ptz)
- [logs/clean_gate_dense_u4k_12x608_lfqat60_fcproj_1m.txt](/Users/ronaldschmidt/openai/logs/clean_gate_dense_u4k_12x608_lfqat60_fcproj_1m.txt)
- [logs/clean_gate_dense_u4k_12x608_lfqat60_fcproj_1m_mlx_model.npz](/Users/ronaldschmidt/openai/logs/clean_gate_dense_u4k_12x608_lfqat60_fcproj_1m_mlx_model.npz)
- [logs/clean_gate_dense_u4k_12x608_lfqat60_fcproj_1m_mlx_model.int8.ptz](/Users/ronaldschmidt/openai/logs/clean_gate_dense_u4k_12x608_lfqat60_fcproj_1m_mlx_model.int8.ptz)

## One-line summary
Current local best is still the legal `12x608 KV2` LFQAT winner at `1.9854` bpb on `262k` and `2.0041` on `1M`; the new evidence says `13x576` is viable but still weaker (`2.0195` best legal, `2.0550` after copied LFQAT), and a simple `14x576` QAT tail is not good enough either, even though the raw `14x576` checkpoint is legally near-tied.

## 2026-03-20 autoresearch local loop

- I wired the repo-specific autoresearch loop to consume measured local results from [autoresearch_results.tsv](/Users/ronaldschmidt/openai/research/autoresearch_results.tsv) so already-tested winners/losers stop reappearing in [autoresearch_loop.md](/Users/ronaldschmidt/openai/research/autoresearch_loop.md).
- The first local autoresearch keep was `autoresearch_sp1024_10x560_projhi_attnhi_fp16`: clean stayed at `2.5858`, but shipped improved from the older aggressive `10x560` export (`2.7825`) to `2.6799`. This is now the best measured local `sp1024` shipping policy.
- The next local autoresearch keep improved the same `10x560` family on the training side, not just the export side:
  - [autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_fp16.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_fp16.txt)
  - clean `2.5724`, shipped `2.6540`, export gap `0.0816`
  - this beats the previous `10x560 export-first` anchor on both clean and shipped while staying under `16MB`
- I then ran a fixed-settings export sweep on that new raw checkpoint:
  - [local_sp1024_10x560_maxtrain_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_maxtrain_m4_export_gap.md)
  - the same conservative `projhi_attnhi_fp16` policy remained best, so the gain is real training-side progress rather than an export-policy fluke
- The strongest new result is now the continuation path on that same winning family:
  - [autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue120.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue120.txt)
  - clean `2.4792`, shipped `2.5205`, export gap `0.0413`
  - followed by a second lower-LR keep in [autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue80_lowlr.txt](/Users/ronaldschmidt/openai/logs/autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue80_lowlr.txt)
  - clean `2.3820`, shipped `2.4397`, export gap `0.0576`
- Both continuation checkpoints got their own fixed-settings export sweeps:
  - [local_sp1024_10x560_continue120_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_continue120_m4_export_gap.md)
  - [local_sp1024_10x560_continue80_lowlr_m4_export_gap.md](/Users/ronaldschmidt/openai/research/local_sp1024_10x560_continue80_lowlr_m4_export_gap.md)
  - in both cases `projhi_attnhi_fp16` stayed the best legal policy, so the improvement is coming from the checkpoint itself, not from a new shipping trick
- The `10x544 longtail` family was measured and effectively killed as a main local path:
  - full run `fcproj_top5_attn_top5`: `2.5875` clean, `2.7738` shipped
  - export-only `fchi_top4_attn_top4`: `2.7567` shipped
  - export-only `fcproj_top4_attn_top4`: `2.7567` shipped
- A new `10x544 maxtrain` recipe is the first real training-side improvement for that family:
  - [local_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fcproj_top5_attn_top5_fp16.txt](/Users/ronaldschmidt/openai/logs/local_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fcproj_top5_attn_top5_fp16.txt)
  - clean `2.5746`, shipped `2.7435`
- Best measured `10x544` shipped point so far comes from an export-only eval on that stronger raw checkpoint:
  - [local_sp1024_bpe_sp_10x544_kv2_export_eval_maxtrain_fchi_top4_attn_top4_fp16.txt](/Users/ronaldschmidt/openai/logs/local_sp1024_bpe_sp_10x544_kv2_export_eval_maxtrain_fchi_top4_attn_top4_fp16.txt)
  - clean `2.5746`, shipped `2.7270`
- Another conservative export on the same raw checkpoint did not help:
  - [local_sp1024_bpe_sp_10x544_kv2_export_eval_maxtrain_fchi_attnhi_fp16.txt](/Users/ronaldschmidt/openai/logs/local_sp1024_bpe_sp_10x544_kv2_export_eval_maxtrain_fchi_attnhi_fp16.txt)
  - shipped `2.7435`

## Current local frontier

- Best measured local `sp1024` path: `10x560 ... + hybrid int8-aware block4attn + joint boundary recovery + projhi_attnhi_fp16` at shipped `2.2710`
- Best measured local `10x544` path: `10x544 maxtrain + fchi_top4_attn_top4` at shipped `2.7270`
- So the `10x544` family is improving, but it is now clearly behind the continuation-and-recovery-hardened `10x560` line.
- Important stability result: every fixed-settings export sweep on the winning `10x560` continuation/recovery path kept the same winner, so `projhi_attnhi_fp16` remains the correct local default for `10x560`.

## Next local moves

1. Keep [research/autoresearch_loop.md](/Users/ronaldschmidt/openai/research/autoresearch_loop.md) empty unless a genuinely new local hypothesis appears; the latest `block4attn` keep is real, but too small to justify another same-family continuation.
2. Treat the `block4proj` and `block4attn` hybrid keeps together as evidence that the current `10x560` line is still movable, but only by sub-threshold amounts; the next local hypothesis should be structurally new, not another tiny continuation on the same exact tensor set.
3. Keep using export-only sweeps when the hypothesis is purely about shipping, but stop spending time on obviously over-cap near-cap variants once the first legal winner stays unchanged.
## 2026-03-21 - public frontier review + H100 pivot

- Decision memo: [research/public_frontier_review_2026-03-21.md](/Users/ronaldschmidt/openai/research/public_frontier_review_2026-03-21.md)
- The `12x576 share6` codebook/QER line is now treated as an exporter anchor, not the default next local tail.
- Added `slide64`-capable validation to [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) with a correctness test in [research/test_sliding_eval.py](/Users/ronaldschmidt/openai/research/test_sliding_eval.py).
- Prepared two next remote launchers:
  - [scripts/runpod/train_breakthrough_sp1024_conservative_slide64_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_conservative_slide64_8xh100.sh)
  - [scripts/runpod/train_breakthrough_sp1024_wildcard_mlp3_slide64_8xh100.sh](/Users/ronaldschmidt/openai/scripts/runpod/train_breakthrough_sp1024_wildcard_mlp3_slide64_8xh100.sh)

## 2026-03-22 - real local sp1024 challenge-data run

- A dedicated long local launcher now exists:
  - [scripts/local/train_lab_sp1024_12x560_share6_real80m_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_sp1024_12x560_share6_real80m_m4.sh)
- Purpose:
  - run a true long local experiment on the published `sp1024` challenge bins, not another short proxy-only branch
  - use the strongest immediately runnable shared-depth scratch family still available from local disk: `12x560 KV2`, `12 logical / 6 unique`, `seq=896`
- Active log:
  - [logs/lab_sp1024_12x560_kv2_share6_896ctx_real80m_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x560_kv2_share6_896ctx_real80m_m4.txt)
- Early confirmed config:
  - train path `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
  - val path `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
  - `MAX_WALLCLOCK_SECONDS=4800`
  - `ITERATIONS=5000`
  - cosine schedule, compile enabled, sparse fp16 exporter keep on `blocks.0-5.{attn.proj,mlp.proj}.weight`
- Early health check:
  - `step:0/5000 val_loss:6.9424 val_bpb:4.1111`
  - `step:1-4` are training normally at roughly `5.8k-6.3k tok/s`
- Important constraint:
  - exact official `u4k` docs are still not fully reconstructable locally because `docs_selected.jsonl` is missing from disk
  - so this `sp1024` long run is currently the closest true local challenge-data path

## 2026-03-22 - correction to the actual local champion

- The above `sp1024 share6` long run was the wrong priority relative to the user's ask and was stopped before completion.
- The correct active long run is now the reproduced local champion family:
  - launcher: [scripts/local/train_lab_u4k_12x608_share10_crossskip_real80m_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_12x608_share10_crossskip_real80m_m4.sh)
  - log: [logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_real80m_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_real80m_m4.txt)
  - init checkpoint: [logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_mlx_model.npz](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_mlx_model.npz)
- Early confirmed config:
  - local dataset `./data/local_u4k_unigram/datasets/fineweb10B_spu4096_local`
  - tokenizer `./data/local_u4k_unigram/tokenizers/fineweb_4096_unigram.model`
  - same `share10 + bus20 + cross_skip` structure as the current local champion
  - same `fc_top4_int4` shipping base
  - `MAX_WALLCLOCK_SECONDS=4800`
- Early health check:
  - `step:0/5000 val_loss:4.5562 val_bpb:1.9871`
  - `step:1/5000 train_loss:3.6903`
  - `model_params:26566080`
