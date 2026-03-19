# Local Dense Frontier Handoff

## Experiment table
| Run | Hypothesis | Key config | Runtime | Artifact bytes | Val loss / val_bpb | Conclusion |
| --- | --- | --- | --- | --- | --- | --- |
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
The strongest measured candidate is no longer the local proxy winner, but the official-path `1xH100` continuation of the same `12x608 KV2` dense family. On the real `fineweb10B_spu4096_docs` path it reached `val_loss=3.0683`, `val_bpb=1.3329` before export, then degraded to `val_loss=3.7062`, `val_bpb=1.6101` after the legal mixed `int4/int8` roundtrip. The compressed model is only `10,777,005` bytes, so the counted total is `10,842,824` with current root [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py). That changes the diagnosis materially: the dense family is not obviously dead on real data anymore, but the current export policy is far too lossy relative to the available byte budget.

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
The blocker is now quantization gap on the real official-data path, not local architecture ranking. The long `1xH100` continuation proved the model itself can reach `1.3329` clean `val_bpb` on full official validation, which is dramatically better than all earlier local evidence and much closer to the public `1.2244` anchor. But the current legal export falls to `1.6101`, so we are burning `0.2772` bpb in the final shipping artifact while still sitting more than `5MB` below the byte cap. That means the highest-EV work is no longer “train longer or try another dense shape first”; it is to redesign the export frontier for this exact checkpoint family and spend the unused bytes where they buy back the most quality.

## Top next moves
1. Run a full official-val export sweep on the `runpod_dense_u4k_12x608_lfqat_continue80m_20260319` raw checkpoint. The immediate target is to recover the `0.2772` bpb roundtrip loss while spending some of the unused `~5.16MB` byte headroom.
2. Keep the real checkpoint artifacts automatically per `RUN_ID`. This is now implemented in [train_gpt.py](/Users/ronaldschmidt/openai/train_gpt.py) so future sweeps cannot accidentally destroy the raw winner.
3. Only return to larger dense shapes after the export frontier is re-measured on the real official path. Right now export policy, not model scale, is the clearest limiter.

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
