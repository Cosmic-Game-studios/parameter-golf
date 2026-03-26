# Evidence

## 2026-03-22 fresh local real rerun

### New bounded result
- Fresh reproduction run of the active local champion:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro2_20260322.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro2_20260322.txt)
  - launcher: [train_lab_u4k_12x608_share10_crossskip_branch_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_12x608_share10_crossskip_branch_m4.sh)
  - same donor, same sparse `fc_top4_int4` exporter family, same `40`-step control-tensor budget
- Result:
  - clean `4.5562 / 1.9871`
  - shipped `4.55020714 / 1.98453111`
  - compressed `14,842,854`
  - estimated total `14,937,204`
  - runtime `97.6s` train + `14.5s` roundtrip eval
- Verdict:
  - `keep`
  - the active local champion was reconfirmed under the same frame instead of merely inherited from older logs

## 2026-03-22 mission reset + first bounded cycle

### New bounded result
- First post-reset asymmetric `cross_skip` branch:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt)
  - launcher: [train_lab_u4k_12x608_share10_crossskip_top2_branch_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_12x608_share10_crossskip_top2_branch_m4.sh)
  - same donor, same sparse `fc_top4_int4` shipping base, same `40`-step control-tensor budget
  - only logical decoder layers `10,11` routed through `cross_skip`
- Result:
  - clean `4.5561 / 1.9871`
  - shipped `4.55026770 / 1.98455752`
  - compressed `14,842,763`
  - estimated total `14,937,113`
  - runtime `103.9s` train + `13.9s` roundtrip eval
- Verdict:
  - `revert`
  - the first sparse/asymmetric `cross_skip` variant is worse than the reproduced all-decoder `cross_skip` winner at `1.98453111`

## 2026-03-22 local `u4k share10 dualrole + bus` cycle

### New supported result
- The tiny all-encoder `cross_skip` router is now reproduced on the active donor:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt)
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt)
  - clean `4.5562 / 1.9871`
  - training-run shipped `4.55020714 / 1.98453111`
  - total under fixed exporter: `14,937,204`
- The obvious “stronger” follow-up is negative:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_busjoint40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_busjoint40_fc_top4_m4.txt)
  - jointly training `global_bus*` and `cross_skip*` on the same donor regressed to shipped `1.98454816`
- A bounded fixed-export reevaluation on the same new checkpoint confirms that the best exporter there is now just plain `fc_top4_int4`:
  - [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md)
  - `fc_top4_int4` -> `1.98453131`
  - `fc_top4_attn_top1_qer_proj_top1_r64` -> `1.98454109`
  - `fc_top4_attn_top1_fp16` -> `1.98454171`
- A token-pair follow-up did not turn into a win, but it did separate the promising half from the dead half:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_pairfeat40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_pairfeat40_fc_top4_m4.txt) regressed to `1.98477693`
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_smear40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_smear40_fc_top4_m4.txt) regressed further to `1.98480479`
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_bigram40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_bigram40_fc_top4_m4.txt) nearly tied at `1.98453402`
  - [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_bigram40_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_bigram40_targeted_export_check.md) still kept `fc_top4_int4` on top at `1.98453381`

### Proved
- The best measured local legal point is now a reproduced result: [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt) and [repro1](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt) both landed at shipped `1.98453111`.
- The `share10 dualrole` family really does improve under a tiny communication add-on:
  - previous donor frontier: [local_u4k_12x608_share10_dualrole01_qer_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_qer_export_gap.md)
  - new narrowed-bus run: [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_fc_top4_m4.txt)
- The previously useful-but-too-expensive fp16 corridor can be shrunk into a legal fixed exporter on the new donor:
  - [local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md)
  - best legal fixed policy: `fc_top4_attn_top1_fp16`
  - shipped `1.98453797`
  - total `15,420,393`
- A slightly stronger fixed legalizer exists on the same checkpoint:
  - [local_u4k_12x608_share10_dualrole01_bus20_qer_legalizer_export_gap_v2.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_qer_legalizer_export_gap_v2.md)
  - best legal fixed policy: `fc_top4_attn_top1_qer_proj_top1_r64`
  - shipped `1.98453631`
  - total `15,637,017`
- The new log-aware exporter path reproduces the same fixed-export ordering on the active donor. Recovering bus runtime flags from the source training log did not change the winner or its bytes.
- The exporter harness is now also safe for custom logical block order and `cross_skip` checkpoints; the log parser no longer confuses embedded source code with runtime metadata.
- That tiny-fp16 fixed exporter reproduces exactly:
  - [local_u4k_12x608_lfqat60_m4__4dfe288810__fc_top4_attn_top1_fp16_repro1.txt](/Users/ronaldschmidt/openai/logs/export_gap_local_u4k_12x608_lfqat60_seq1024/local_u4k_12x608_lfqat60_m4__4dfe288810__fc_top4_attn_top1_fp16_repro1.txt)
  - shipped `1.98453797`

### Refuted
- The bounded structural remap bracket on the active donor is negative across the board:
  - `...8,9,8,9` -> `1.98614203`
  - `...9,9,8,8` -> `1.98605052`
  - `...6,7` -> `1.98580117`
  - `...4,5` -> `1.98499612`
  - verdict: moving repeated computation to other decoder slots did not beat the plain modulo donor
- On the old donor, medium-rank `QER` / codebook-QER exporter extensions did not pay off:
  - [local_u4k_12x608_share10_dualrole01_qer_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_qer_export_gap.md)
  - `fc_top4_int4` stayed best at `1.98455440`
- On the new bus20 donor:
  - `fc_top3_int4` is materially worse at `1.98545449`
  - `fc_top4_attn_top2_fp16` is legal but slightly behind the top1 keep at `1.98453859`
  - `fc_top4_proj_top2_fp16` and `fc_top4_proj_top2_attn_top2_fp16` are both over cap
  - the first boundary-aware bus v2 branch [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_boundary_top4r_top2w20_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_boundary_top4r_top2w20_fc_top4_m4.txt) stayed flat at `1.98455440`
  - the first pure token-flow branch [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_byteweight40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_byteweight40_fc_top4_m4.txt) regressed sharply to `2.04253031`; even its best fixed exporter in [local_u4k_12x608_share10_dualrole01_bus20_byteweight40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_byteweight40_export_gap.md) only reached `2.04248476`
  - the first cheap structural second-pass branch [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_secondpass89_gate40_fc_top4_m4_retry1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_secondpass89_gate40_fc_top4_m4_retry1.txt) also lost at `1.99239355`; its best fixed exporter in [local_u4k_12x608_share10_dualrole01_bus20_secondpass89_export_gap_v2.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_secondpass89_export_gap_v2.md) only reached `1.99238898`
  - the first joint `cross_skip + global_bus` retrain [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_busjoint40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_busjoint40_fc_top4_m4.txt) also lost at `1.98454816`
  - the first token-pair stack on the active donor is negative:
    - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_pairfeat40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_pairfeat40_fc_top4_m4.txt) -> `1.98477693`
    - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_smear40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_smear40_fc_top4_m4.txt) -> `1.98480479`
    - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_bigram40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_bigram40_fc_top4_m4.txt) came closest but still stayed slightly behind; even its targeted exporter check in [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_bigram40_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_bigram40_targeted_export_check.md) only reached `1.98453381`

### Supported but not enough
- The `cross_skip` keep is now reproduced, but it is still only about `5e-6` shipped `bpb` better than the old donor. The next step should stay inside tiny communication controls, but not by retraining the bus together.
- The overall lesson is now sharper: expensive useful corridors should be legalized surgically, logical remap did not help, joint bus-plus-cross-skip retraining did not help, the first token-pair stack does not beat the current donor, and the next high-EV move is a sparser or more asymmetric `cross_skip`-style communication control.

### Running but not yet decision-ready
- A real long local run on actual challenge `sp1024` bins is now active:
  - [lab_sp1024_12x560_kv2_share6_896ctx_real80m_m4.txt](/Users/ronaldschmidt/openai/logs/lab_sp1024_12x560_kv2_share6_896ctx_real80m_m4.txt)
  - launcher: [train_lab_sp1024_12x560_share6_real80m_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_sp1024_12x560_share6_real80m_m4.sh)
- This is the first deliberate long local run on the published challenge `sp1024` bins under the new mission framing.
- Early evidence only:
  - `step:0/5000 val_loss:6.9424 val_bpb:4.1111`
  - first few train steps are healthy
  - run confirmed to use `./data/datasets/fineweb10B_sp1024`
- No promotion decision should be made until at least the first real intermediate val point lands.

### Superseded running branch
- The above `sp1024 share6` long run was explicitly superseded because it was not the user's requested local champion path.
- The active long run is now the actual reproduced local champion family:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_real80m_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_real80m_m4.txt)
  - launcher: [train_lab_u4k_12x608_share10_crossskip_real80m_m4.sh](/Users/ronaldschmidt/openai/scripts/local/train_lab_u4k_12x608_share10_crossskip_real80m_m4.sh)
- Early evidence on the corrected champion run:
  - `step:0/5000 val_loss:4.5562 val_bpb:1.9871`
  - `step:1/5000 train_loss:3.6903`
  - correct local champion checkpoint loaded, with both `global_bus` and `cross_skip_router` enabled
