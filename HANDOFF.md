# Local Dense Frontier Handoff

## Experiment table
| Run | Hypothesis | Key config | Runtime | Artifact bytes | Val loss / val_bpb | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| `clean_gate_dense_u4k_12x608_lfqat60_fcproj_1m` | The new LFQAT winner must survive a meaningfully larger gate. | `12x608 KV2`, legal export=`fc_hi + proj_hi`, `VAL_MAX_TOKENS=1,048,576`, checkpoint=`dense_u4k_12x608_kv2_lfqat60_fcproj`. | `82.0s` clean eval + `84.3s` roundtrip eval | `14,796,847 total` | `4.5593 / 2.0041` | Best completed longer-gate local result. |
| `dense_u4k_12x608_kv2_lfqat60_fcproj` | Fisher- and KL-guided LFQAT on the legal export family can beat both raw continuation and earlier post-hoc-only legalizers. | Start from raw `12x608 adapt150`, `60` steps, constant LR `0.004/0.003/0.003`, legal export=`fc_hi + proj_hi`, `LFQAT_KL_WEIGHT=0.05`, `LFQAT_FISHER_WEIGHT=0.01`, `TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015`, ramp `qat_prob=0.15->1.0` by step `30`. | `330.8s` train + `14.3s` roundtrip eval | `14,796,847 total` | `4.5523 / 1.9854` | Current best legal `262k` local result and first local dense result under `1.99`. |
| `dense_u4k_13x576_boot_continue60_fcproj_lr5` | The mathematically attractive `13x576` shape may not need a heavy adaptation; a gentle low-LR continuation from the expanded boot checkpoint could already beat the boot export cleanly. | Expand `12x576` -> `13x576`, legal export=`fc_hi + proj_hi` on blocks `6-12`, then `60` constant-LR steps `0.006/0.005/0.005`. | `346.1s` train + `18.6s` roundtrip eval | `13,921,399 total` | `4.6304 / 2.0195` | Best completed `13x576` result so far. Real positive gain over the legal boot probe, but still behind `12x608`. |
| `dense_u4k_14x576_kv2_boot_from_12x576_adapt150` | If the bought-capacity thesis is real, a deeper `14x576` dense model should at least tie the `12x608` line while staying under the byte cap. | Expand `12x576` -> `14x576`, adapt `150` cosine steps, legal export=`fc_hi + proj_hi` on blocks `7-13`. | `522.3s` train + `54.7s` roundtrip eval | `15,180,590 total` | `4.5541 / 1.9862` | Important near-cap control. Legal and almost tied with the winner, so the family is not obviously dead. |
| `dense_u4k_14x576_kv2_qat60_fcproj` | If `14x576` is truly the next step up, a simple compile-stable `QAT + compression-aware` tail should push it past `12x608`. | Start from raw `14x576 adapt150`, `60` constant-LR steps `0.006/0.0045/0.0045`, legal export=`fc_hi + proj_hi`, `TRAIN_QAT=fc_hi+proj_hi`, `TRAIN_COMPRESSION_AWARE_WEIGHT=0.0015`. | `214.8s` train + `19.8s` roundtrip eval | `15,223,725 total` | `4.8062 / 2.0962` | Clear negative result. This simple post-adapt QAT tail hurts the larger model badly. |
| `dense_u4k_13x576_kv2_lfqat60_fcproj` | If `13x576` is truly the clean byte-budget sweet spot, a better warm start plus the same LFQAT recipe as the `12x608` winner should keep it competitive. | Start from `13x576 boot_continue60`, `60` LFQAT steps, constant LR `0.004/0.003/0.003`, legal export=`fc_hi + proj_hi` on blocks `6-12`. | `377.0s` train + `17.7s` roundtrip eval | `13,954,254 total` | `4.7119 / 2.0550` | Completed negative result. Even the stronger `13x576` LFQAT continuation stays clearly behind the `12x608` LFQAT winner. |
| `dense_u4k_12x608_plain_continue60_fcproj` | A simple low-LR tail on the raw `12x608` checkpoint may still beat the old leader without any extra compression-aware objective. | Start from raw `12x608 adapt150`, `60` steps, constant LR `0.006/0.005/0.005`, no QAT, no compression-aware loss, legal export=`fc_hi + proj_hi`. | `173.0s` train + `14.4s` roundtrip eval | `14,813,203 total` | `4.6017 / 2.0070` | Strong positive control, but worse than LFQAT on the same family. |

## Current best candidate
Current local leader is a `12x608`, `KV2`, `Unigram-4096` dense model with legal mixed export `INT4(blocks.6-11.mlp.fc.weight + blocks.6-11.mlp.proj.weight)` and a `60`-step LFQAT continuation from the raw `adapt150` checkpoint. The LFQAT recipe uses ramped fake quantization, teacher-style KL against the full-precision path, and Fisher-weighted compression alignment on the same legal export family. Best measured metrics are `val_loss=4.55231047`, `val_bpb=1.98544845` on the `262k` gate and `val_loss=4.55927324`, `val_bpb=2.00412683` on the `1M` gate. Best measured compressed model size is `14,735,753` bytes; with current root [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) at `61,094` bytes, the estimated counted total is `14,796,847`. New evidence from `13x576` is mixed but decisive: a gentle legal continuation can reach `2.0195`, yet the same LFQAT tail that wins at `12x608` degrades it to `2.0550`, so the larger shape is not the current best local path.

## What changed in this cycle
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
The blocker is now raw quality gap and scaling stability, not legal export bytes. The legal `12x608` mixed `int4/int8` export is already stable and the quantization gap on the winner is tiny (`1.9878` clean vs `1.9854` roundtrip on `262k`; `2.0076` clean vs `2.0041` roundtrip on `1M`). But the first larger dense follow-ups were not free wins: `13x576` lost even after a stronger LFQAT continuation (`2.0550`), and `14x576` only tied the winner in raw form before a simple `QAT + compression-aware` tail made it much worse (`2.0962`). The remaining gap to the user’s local `1.800` target is therefore mostly a training/data/model-quality problem inside this local `Unigram-4096` path, plus a recipe-stability problem once we scale past `12x608`.

## Top next moves
1. Rebuild the current LFQAT winner on the official raw-doc retokenization path. The current local path is still the reconstructed `spu4096_local` corpus.
2. Port LFQAT from [train_gpt_mlx.py](/Users/ronaldschmidt/openai/train_gpt_mlx.py) into [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) before CUDA. Right now the winning training method is only implemented in MLX.
3. Only revisit a larger dense model if it uses a more principled recipe than the failed local continuations here. `13x576` is now killed locally, and `14x576` needs something stronger than post-hoc QAT if it is going to beat `12x608`.

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
