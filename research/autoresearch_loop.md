# Autoresearch Loop

- Mode: `local-first`

## Anchors

- Best measured official shipped anchor: `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` at `1.5106` bpb.
- Best measured local shipped anchor: `dense_u4k_12x608_kv2_lfqat60_fcproj` at `1.9854` bpb.
- Best measured local autoresearch anchor: `lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy` at `2.1028` bpb.
- Best fixed-settings export policy for the current `10x560` local checkpoint: `projhi_attnhi_fp16` with shipped `2.2710` and gap `0.1639`.

## Resolved Results

- `u5k_12x608_exactcopy_gate`: `kept` clean `2.1746` shipped `2.1733` gap `-0.0013` (exact-copying shared SentencePiece rows turned the same larger-tokenizer branch from clearly bad into an alive starting point)
- `u5k_12x608_exactcopy_lowlr40`: `kept` clean `2.1087` shipped `2.1039` gap `-0.0048` (first careful continuation on the corrected tokenizer transfer became the best measured tokenizer-transfer challenger so far)
- `u5k_12x608_exactcopy_continue40_ultralow`: `discarded` clean `2.1093` shipped `2.1039` gap `-0.0054` (second continuation was effectively flat to slightly worse, so this exact u5k exploit style has already plateaued)
- `sp1024_10x560_export_first_hybrid_int8aware_block4attn_keptfp16_80`: `kept` clean `2.1093` shipped `2.2720` gap `0.1627` (hybrid int8-aware block4 attn.proj plus the proven kept-fp16 recovery corridor bought a new best local shipped point, but only by a tiny sub-threshold margin; treat it as another saturation keep, not a new continuation line)
- `sp1024_10x560_export_first_hybrid_int8aware_block4proj_keptfp16_80`: `kept` clean `2.1104` shipped `2.2722` gap `0.1618` (hybrid int8-aware block4 mlp.proj plus the proven kept-fp16 recovery corridor bought a new best local shipped point, but only by a clearly sub-threshold margin; treat this as another near-saturation keep rather than a new continuation chain)
- `sp1024_10x560_export_first_int8aware_block4proj120`: `discarded` clean `2.1114` shipped `2.2734` gap `0.1620` (pure block4 mlp.proj int8-aware recovery improved clean slightly but regressed shipped relative to the current kept anchor, so isolated int8-aware tuning on that tensor is not enough on its own)
- `sp1024_10x560_export_first_keptfp16_recovery120_continue80b`: `kept` clean `2.1124` shipped `2.2725` gap `0.1601` (third fp16-kept recovery tail still improved shipped, but only by a sub-threshold amount; treat as the last meaningful point on this continuation line and avoid infinite micro-tail chaining)
- `sp1024_10x560_export_first_keptfp16_recovery120_continue80`: `kept` clean `2.1145` shipped `2.2735` gap `0.1590` (second fp16-kept recovery tail improved shipped again on the same restricted parameter set; fixed export sweep reconfirmed projhi_attnhi_fp16 as best on the new checkpoint)
- `sp1024_10x560_export_first_keptfp16_recovery120`: `kept` clean `2.1175` shipped `2.2745` gap `0.1570` (targeted recovery tail on the discarded cleaner checkpoint improved shipped again by training only the tensors already kept in fp16; fixed export sweep reconfirmed projhi_attnhi_fp16 as best on the new checkpoint)
- `autoresearch_sp1024_10x560_maxtrain_projhi_attnhi_continue200_ultralowlr_continue120_microtail_continue120_microtail`: `discarded` clean `2.1267` shipped `2.2821` gap `0.1554` (clean improved again but shipped only moved by noise-level amount while export gap widened; do not promote this over the previous microtail anchor)
- `sp1024_10x560_export_first_continue120_microtail`: `kept` clean `2.1611` shipped `2.2821` gap `0.1210` (fourth continuation tail kept improving shipped bpb; fixed export sweep again confirmed projhi_attnhi_fp16 remains best on the stronger checkpoint)
- `sp1024_10x560_export_first_continue200_ultralowlr`: `kept` clean `2.2203` shipped `2.3149` gap `0.0946` (third continuation tail on the same 10x560 winner kept compounding gains; checkpoint-specific export sweep reconfirmed projhi_attnhi_fp16 as best)
- `sp1024_10x560_export_first`: `kept` clean `2.5724` shipped `2.6540` gap `0.0816` (first real training-side improvement on the 10x560 path; export sweep confirmed projhi_attnhi_fp16 stays best)
- `sp1024_10x560_export_first_continue120`: `kept` clean `2.4792` shipped `2.5205` gap `0.0413` (first low-LR continuation tail on the new 10x560 winner; very large clean and shipped gain with the same export winner)
- `sp1024_10x560_export_first_continue80_lowlr`: `kept` clean `2.3821` shipped `2.4397` gap `0.0576` (second lower-LR continuation tail kept improving shipped bpb; export sweep again confirmed projhi_attnhi_fp16 stays best)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16`: `discarded` clean `2.5875` shipped `2.7738` gap `0.1863` (worse shipped than 10x560 export-first despite slightly faster training)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16`: `discarded` clean `2.5875` shipped `2.7567` gap `0.1692` (export-only eval on same raw 10x544 checkpoint still behind 10x560 export-first)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top4_attn_top4_fp16`: `discarded` clean `2.5875` shipped `2.7567` gap `0.1692` (export-only eval on same raw 10x544 checkpoint still behind 10x560 export-first)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fcproj_top5_attn_top5_fp16`: `kept` clean `2.5746` shipped `2.7435` gap `0.1689` (best measured 10x544 training recipe so far but still behind 10x560 export-first)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fchi_top4_attn_top4_fp16`: `discarded` clean `2.5746` shipped `2.7270` gap `0.1524` (best measured 10x544 shipped policy so far but still behind 10x560 export-first)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fchi_attnhi_fp16`: `discarded` clean `2.5746` shipped `2.7435` gap `0.1689` (fc-only int4 plus upper attn fp16 on maxtrain raw checkpoint gave no gain over fcproj_top5)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fchi_attnhi_fp16`: `discarded` clean `2.5875` shipped `2.7738` gap `0.1863` (export-only eval on same raw 10x544 longtail checkpoint matched the bad top5 policy)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_attnhi_fp16`: `discarded` clean `2.5875` shipped `2.7738` gap `0.1863` (export-only eval on same raw 10x544 longtail checkpoint gave no gain over fcproj_top5)
- `seeker_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16`: `discarded` clean `2.5858` shipped `2.7881` gap `0.2023` (top offline 10x560 export policy was worse than measured projhi_attnhi on the real 10x560 raw checkpoint)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fcproj_top4_attn_top4_fp16`: `discarded` clean `2.5746` shipped `2.7270` gap `0.1524` (matches maxtrain fchi_top4 result and still trails 10x560 export-first)
- `seeker_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16`: `discarded` clean `2.5858` shipped `2.7881` gap `0.2023` (top offline 10x560 export policy was worse than measured projhi_attnhi on the same raw checkpoint)
- `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fcproj_top4_attn_top4_fp16`: `discarded` clean `2.5746` shipped `2.7270` gap `0.1524` (maxtrain fcproj_top4 matched the existing fchi_top4 shipped point and stayed behind 10x560 export-first)
- `sp1024_12x544_share6_scratch`: `kept` clean `2.5675` shipped `2.5697` gap `0.0022` (shared-depth scratch branch is not globally competitive yet, but it established a new legal family with a near-zero export gap under proj_top6_attn_top6_fp16)
- `sp1024_12x544_share6_continue120`: `kept` clean `2.4793` shipped `2.4816` gap `0.0023` (short low-LR continuation materially improved the shared-depth branch while preserving the same nearly lossless exporter)
- `sp1024_12x560_share6_scratch`: `kept` clean `2.5628` shipped `2.5662` gap `0.0034` (wider shared-depth scratch slightly beat the smaller scratch point while preserving the same exporter, so the width-up branch earned a continuation)
- `sp1024_12x560_share6_continue120`: `kept` clean `2.4542` shipped `2.4564` gap `0.0022` (first continuation on the width-up branch clearly beat the smaller shared-depth branch)
- `sp1024_12x560_share6_continue80_lowlr`: `kept` clean `2.3384` shipped `2.3422` gap `0.0038` (second continuation on the width-up branch became the first shared-depth point to beat the old 10x560 local anchor)
- `sp1024_12x560_share6_continue80_ultralowlr`: `kept` clean `2.2596` shipped `2.2638` gap `0.0042` (new best local autoresearch point; the shared-depth branch is now the leading local line and still uses the same legal exporter)
- `sp1024_12x560_share6_continue80_nanotail`: `kept` clean `2.2146` shipped `2.2198` gap `0.0052` (material keep; the same shared-depth exporter stayed best and the branch kept descending)
- `sp1024_12x560_share6_continue80_picotail`: `kept` clean `2.1901` shipped `2.1962` gap `0.0061` (new best local autoresearch point; targeted frontier check kept proj_top6_attn_top6_fp16 on top)
- `sp1024_12x560_share6_continue80_femtotail`: `kept` clean `2.1790` shipped `2.1865` gap `0.0075` (final bounded femtotail still produced a real keep; exporter stayed fixed, so the next highest-EV move is structural rather than another identical tail)
- `sp1024_12x576_share6_boot_from_12x560_femtotail_continue120`: `supported` clean `2.2354` shipped `2.2402` gap `0.0048` (width-up boot started behind the femtotail anchor but adapted quickly enough to justify one more bounded continuation)
- `sp1024_12x576_share6_continue80_lowlr`: `kept` clean `2.1714` shipped `2.1770` gap `0.0056` (first low-LR follow-up on the widened branch became the new local winner; targeted frontier check kept proj_top6_attn_top6_fp16 on top)
- `sp1024_12x576_share6_continue80_ultralowlr`: `kept` clean `2.1378` shipped `2.1454` gap `0.0076` (second low-LR follow-up on the widened branch improved again and is now the strongest measured local model)
- `u6k_12x608_exactcopy_gate`: `discarded` clean `2.1992` shipped `2.1997` gap `0.0005` (larger unigram saved more tokens than u5k but still started from a worse gate, so just increasing vocab again is not the next winner)
- `b4k_12x608_exactcopy_gate`: `discarded` clean `2.7376` shipped `2.7321` gap `-0.0055` (sideways 4k BPE had slightly worse token counts than u4k and a clearly bad gate, so this branch is killed)
- `u5k_12x608_seq896_continue40`: `kept` clean `2.1087` shipped `2.1028` gap `-0.0059` (first token-flow retune on the alive u5k checkpoint bought a small real keep, so the next local lever remains on this branch rather than another raw tokenizer family)

## Next Experiments

### 1. `lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy_batch9216_continue40`

- Track: `M4 local`
- Source: `measured_u5k_tokenflow_continuation`
- Hypothesis: The revived `u5k` branch has already produced one small `seq_len=896` keep. A second bounded token-flow step that only increases train tokens per update may convert the same alive branch into a larger real gain without reopening raw tokenizer search.
- Keep if: Keep if shipped val_bpb beats the current best measured local autoresearch anchor (2.1028) or if clean improves by >= 0.01 with no shipping regression.
- Kill if: Kill if the first `step 20` validation is flat to worse than the current anchor or if final shipped does not improve.
- Why:
  - Current alive tokenizer/token-flow anchor: lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy at shipped 2.1028.
  - The first sideways tokenizer probes (`u6k`, `b4k`) both lost, so the next low-entropy lever stays on the alive `u5k` branch.
- Command:
```bash
RUN_ID=lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy_batch9216_continue40 INIT_MODEL_PATH=./logs/lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy_mlx_model.npz DATA_PATH=./data/local_u5k_unigram/datasets/fineweb10B_spu5120_local TOKENIZER_PATH=./data/local_u5k_unigram/tokenizers/fineweb_5120_unigram.model VOCAB_SIZE=5120 TRAIN_SEQ_LEN=896 TRAIN_BATCH_TOKENS=9216 VAL_BATCH_SIZE=131072 VAL_MAX_TOKENS=262144 LR_SCHEDULE=constant WARMUP_STEPS=0 LR_WARMUP_ITERS=0 MIN_LR_SCALE=1.0 TIED_EMBED_LR=0.0009 MATRIX_LR=0.0006 SCALAR_LR=0.0006 ITERATIONS=40 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=20 TRAIN_LOG_EVERY=10 LFQAT_FULL_STEP=5 LFQAT_MIN_PROB=0.50 LFQAT_KL_WEIGHT=0.015 LFQAT_FISHER_WEIGHT=0.003 TRAIN_COMPRESSION_AWARE_WEIGHT=0.0008 bash scripts/local/train_lab_u5k_12x608_transplant_lowlr40_m4.sh
```
