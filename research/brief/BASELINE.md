# Baseline

## Official anchor
- [research/reports/SCOREBOARD.md](/Users/ronaldschmidt/openai/research/reports/SCOREBOARD.md)
- best measured official shipped `val_bpb`: `1.51056383`
- run: `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi`

## Local reproduced champion
- [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt)
- [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt)
- clean `4.5562 / 1.9871`
- shipped `4.55020714 / 1.98453111`
- best fixed exporter:
  - [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md)
  - `fc_top4_int4`
  - shipped `1.98453131`
  - total `14,937,204`

## Latest bounded-cycle result
- [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt)
- same donor family, same exporter family, same `40`-step control-tensor budget
- clean `4.5561 / 1.9871`
- shipped `4.55026770 / 1.98455752`
- decision: `revert`

## Frontier gaps
- Local stretch target: `<=1.9`
- Official long-term target: `<=0.9`
- Public frontier still far below our current local and official anchors, so the mission now optimizes for scalable breakthroughs rather than more same-family polishing.
