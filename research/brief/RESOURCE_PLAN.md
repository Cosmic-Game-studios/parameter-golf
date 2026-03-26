# Resource Plan

## Local evidence required first
- Reproduced local champion:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt)
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt)
- Strongest fixed exporter on that checkpoint:
  - [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md)
- Latest negative asymmetry check:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt)

## What matters most by stage
- Stage 1 local:
  - comparable local shipped `val_bpb`
  - export gap
  - total bytes
  - cheap falsification
  - training/runtime stability
- Stage 2 official:
  - official shipped `val_bpb`
  - step speed / useful updates per `600s`
  - full counted artifact breakdown

## Web search policy
- No routine web work for local cycles.
- Reopen public-frontier refresh only when:
  - preparing an H100 promotion
  - or when local evidence suggests a new branch family that needs frontier comparison

## Default validation tools
- `py_compile`
- focused unit tests
- `bash -n` for launchers
- exact local shipped/clean metrics from logs
- exporter reevaluation only when checkpoint promotion relevance warrants it
