# Scoreboard

## Locked anchors

| Tier | Run | Shipped val_bpb | Clean val_bpb | Total bytes | Status |
| --- | --- | ---: | ---: | ---: | --- |
| official | `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` | `1.5106` | `1.3838` | `12,126,772` | current best measured official |
| local overall | `lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4 + fc_top4_int4` | `1.9845` | `1.9871` | `14,937,204` | current best measured local legal; the tiny `cross_skip` keep has now reproduced exactly at `1.98453111` |
| local best fixed-export alt | `same checkpoint + fc_top4_int4` | `1.9845` | `1.9871` | `14,937,204` | strongest fixed exporter on the new cross-skip checkpoint; the old near-cap legalizer no longer wins on this donor |
| local previous fixed-export alt | `same checkpoint + fc_top4_attn_top1_qer_proj_top1_r64` | `1.9845` | `1.9871` | `15,637,181` | supported near-cap legalizer on the new cross-skip checkpoint, but slightly behind plain `fc_top4_int4` |
| local second fixed-export alt | `same checkpoint + fc_top4_attn_top1_fp16` | `1.9845` | `1.9871` | `15,420,393` | reproduced tiny-fp16 keep; slightly behind the QER legalizer |
| local previous champion | `dense_u4k_12x608_kv2_lfqat60_fcproj` | `1.9854` | `1.9878` | `14,796,847` | former locked local legal winner |
| local structural donor | `lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4 + fc_top4_int4` | `1.9845` | `1.9871` | `14,937,204` | current best measured partial-sharing + communication branch; exact repro confirms the same point |
| local structural fallback | `lab_u4k_12x608_kv2_share9_first_cycle_eval0_int8all_m4 + int8_all` | `1.9886` | `1.9874` | `15,087,368` | shipping-friendly partial-sharing fallback |
| local shared-depth legacy branch | `lab_u4k_12x608_kv2_share6_first_cycle_continue120_int8all_m4 + projhi_attnhi_fp16` | `2.0019` | `2.0004` | `15,423,066` | older living branch, now superseded |

## Active mission
- Run a two-stage mission: local M4 falsification and candidate building first, official H100 promotion only for branches that earn it.
- Keep `lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4` as the reproduced local regression bar.
- Treat the active `u4k share10 + bus20 + cross_skip` family as locally near saturation unless a materially more orthogonal branch reopens it.

## Latest decision-relevant results
- Mission-reset bounded cycle:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt)
  - same donor, same exporter family, but only logical decoder layers `10,11` use `cross_skip`
  - clean `4.5561 / 1.9871`
  - shipped `4.55026770 / 1.98455752`
  - decision: `revert`; this first sparse/asymmetric `cross_skip` branch did not beat the reproduced winner
- New focused reports:
  - [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_export_gap.md)
  - [local_u4k_12x608_share10_dualrole01_qer_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_qer_export_gap.md)
  - [local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_minihybrid_export_gap.md)
  - [local_u4k_12x608_share10_dualrole01_bus20_top2hybrid_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_top2hybrid_export_gap.md)
  - [local_u4k_12x608_share10_dualrole01_bus20_qer_legalizer_export_gap_v2.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_qer_legalizer_export_gap_v2.md)
  - [local_u4k_12x608_share10_dualrole01_bus20_boundary_top4r_top2w20_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_boundary_top4r_top2w20_export_gap.md)
  - [local_u4k_12x608_share10_dualrole01_bus20_byteweight40_export_gap.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_byteweight40_export_gap.md)
  - [local_u4k_12x608_share10_dualrole01_bus20_secondpass89_export_gap_v2.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_secondpass89_export_gap_v2.md)
- Kept:
  - `share10 dualrole01 + bus20 + cross_skip + fc_top4_int4` is the reproduced best measured local structural anchor at `1.98453111`; [repro1](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_fc_top4_m4_repro1.txt) landed on the same exact shipped score
  - the strongest fixed exporter on the same checkpoint simplified back to plain `fc_top4_int4` at `1.98453131`
  - shrinking the previously-too-expensive fp16 corridor still works: `fc_top4_attn_top1_fp16` reproduces at `1.98453797`
- Supported but below bar:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_bigram40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_bigram40_fc_top4_m4.txt) is the closest token-pair branch, but it still stayed just behind the reproduced winner at training-run shipped `1.98453402`
  - the targeted exporter check on the same checkpoint in [local_u4k_12x608_share10_dualrole01_bus20_crossskip40_bigram40_targeted_export_check.md](/Users/ronaldschmidt/openai/research/local_u4k_12x608_share10_dualrole01_bus20_crossskip40_bigram40_targeted_export_check.md) still favored plain `fc_top4_int4` at `1.98453381`, so the branch remains a near-tie loser rather than a promotion
- Refuted:
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_top2_40_fc_top4_m4.txt) at `1.98455752`; limiting `cross_skip` to the top two decoder layers did not help, so the first post-reset sparse/asymmetric router branch is a loser
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_pairfeat40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_pairfeat40_fc_top4_m4.txt) at `1.98477693`; the first direct `SmearGate + BigramHash` stack regressed clearly on the active donor
  - [lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_smear40_fc_top4_m4.txt](/Users/ronaldschmidt/openai/logs/lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip40_smear40_fc_top4_m4.txt) at `1.98480479`; `SmearGate` by itself is worse than both the winner and the combined pair-feature branch
  - `lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_crossskip_busjoint40_fc_top4_m4` at `1.98454816`; jointly retraining `global_bus*` and `cross_skip*` was slightly worse than keeping the bus frozen
  - the full bounded structural remap bracket:
    - `...8,9,8,9` -> `1.98614203`
    - `...9,9,8,8` -> `1.98605052`
    - `...6,7` -> `1.98580117`
    - `...4,5` -> `1.98499612`
  - `fc_top3_int4` on the new anchor (`1.98545449`)
  - `fc_top4_proj_top2_fp16` and `fc_top4_proj_top2_attn_top2_fp16`; both are over cap and do not buy enough quality
  - `lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_boundary_top4r_top2w20_fc_top4_m4` stayed flat at `1.98455440`, so the first boundary-aware bus v2 branch is not a keep
  - `lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_byteweight40_fc_top4_m4` regressed sharply; even the best fixed exporter only reached `2.04248476`
  - `lab_u4k_12x608_kv2_share10_first_cycle_dualrole01_bus20_secondpass89_gate40_fc_top4_m4_retry1` also lost; even the best fixed exporter only reached `1.99238898`
  - the larger direct fp16 corridor remains real but still too expensive: `fc_top4_proj_top1_attn_top1_fp16` is slightly better in raw shipped quality but illegal at `16,435,203`
- Decision: the reproduced `cross_skip` keep remains the local bar, but the first sparse/asymmetric follow-up already lost. The next local branch should now be materially more orthogonal and more scaling-oriented than another tiny control tweak on the same donor.
