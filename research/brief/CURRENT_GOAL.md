# Current Goal

## Mission now
Build a durable, evidence-backed research program that finds the strongest realistic path from our local M4 lab to an official `8xH100 / 600s` challenger, with a long-term north star of `0.8-0.9` shipped `val_bpb`.

## Immediate reality
- Best measured local legal point is still:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4.txt)
  - exact repro: [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt)
  - shipped `1.98453111`
  - fixed exporter `fc_top4_int4` at `1.98453131`, `14,937,204` total
- Best measured official point remains:
  - [research/reports/SCOREBOARD.md](/Users/ronaldschmidt/openai/research/reports/SCOREBOARD.md)
  - `1.51056383` shipped on the old official `sp1024` path

## What success means
- Short term: maintain a reproduced local champion and stop wasting cycles on low-signal repeats.
- Medium term: identify one conservative and one high-upside local branch that plausibly scale better than the current local champion.
- Promotion target: send only branches with clear local evidence and scaling rationale to official H100.
- Long term: converge on a credible, measurable route toward `<=0.9` shipped `val_bpb`, or prove why the current design family cannot reach it.

## Current bounded objective
- Treat the reproduced `cross_skip` point as the locked local bar.
- Record the first asymmetry test as a completed negative result:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt)
  - clean `4.5561 / 1.9871`
  - shipped `4.55026770 / 1.98455752`
- Move the next local cycle away from “same donor, slightly different `cross_skip`” unless the next branch is materially more orthogonal.
