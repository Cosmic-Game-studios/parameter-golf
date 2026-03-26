# Next Session

1. Start from the active corrected running experiment in [STATE.json](/Users/ronaldschmidt/openai/research/STATE.json).
2. The current live long run is now the actual local champion family:
   - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_real80m_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_real80m_m4.txt)
   - launcher: [train_lab_u4k_12x608_share10_crossskip_real80m_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_12x608_share10_crossskip_real80m_m4.sh)
   - init checkpoint: [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_mlx_model.npz](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_mlx_model.npz)
   - status: running under `MAX_WALLCLOCK_SECONDS=4800`
   - do not start a competing long local run until this one resolves or crashes
3. The earlier `sp1024 share6` long run was superseded and should stay parked unless we intentionally reopen that family later.
4. Keep the reproduced local champion as the regression bar:
   - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt)
   - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt)
   - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro2_20260322.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro2_20260322.txt)
5. Treat the first sparse/asymmetric follow-up as refuted:
   - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt)
   - shipped `1.98455752`
6. While the long run is live, the only decision-useful checks are:
   - confirm the first intermediate val point
   - confirm the run survives past warmup and into stable throughput
   - record final clean/shipped metrics plus runtime when it ends
7. Do not spend the next cycle on:
   - another tiny `cross_skip` mask tweak on the same donor
   - another `SmearGate` or unchanged token-pair replay
   - another `cross_skip + global_bus` joint retrain
8. Lowest-entropy action after the long run finishes:
   - compare its final local `u4k` clean/shipped point against the reproduced champion bar and the old `dense_u4k_12x608_kv2_lfqat60_fcproj` line
   - decide `keep`, `revert`, `crash`, or `park`
   - only then choose the next orthogonal branch or any H100 promotion candidate
9. Before any H100 promotion, require:
   - a measured local win or strong critic-cleared rationale
   - updated frontier comparison
   - explicit expected official upside story
10. If the long run is only a near-tie or changes the comparison frame, run a critic gate before calling it a `keep`.
