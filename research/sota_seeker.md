# SOTA Seeker

- Target shipped bpb: `1.1000`
- Counted code bytes: `67722`
- Search space: `1 combinatorial sp1024 family x 22 models x 12 token-flows x 14 training recipes x 18 export policies + 12 carried-over u4k finalists`

## Verdict

- Best official candidate by overall objective: `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16` at estimated shipped `1.1614` bpb.
- Lowest-estimated shipped official-ready candidate: `seeker_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16` at estimated shipped `1.1594` bpb.
- Highest-upside candidate: `seeker_sp1024_bpe_sp_12x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_tok_attnhi_fp16` with best-case `1.1333` bpb.
- Clean gain still needed after best-case export: `0.0333` bpb.
- Current verdict: the search space contains launchable target-stretch candidates.

## Official-Ready Candidates

| Rank | Candidate | Tokenizer | Model | Token flow | Training | Export | Est. shipped | Best-case | Bytes | Runtime | Band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1` | `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16` | `sp1024_bpe` | `10x544` | `throughput_832k_896ctx` | `stable_muon97_compiled_longtail` | `fcproj_top5_attn_top5_fp16` | `1.1614` | `1.1493` | `15999169` | `0.993` | `target-stretch` |
| `2` | `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16` | `sp1024_bpe` | `10x544` | `throughput_832k_896ctx` | `stable_muon97_compiled_longtail` | `fchi_top4_attn_top4_fp16` | `1.1624` | `1.1503` | `15569169` | `0.993` | `target-stretch` |
| `3` | `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top4_attn_top4_fp16` | `sp1024_bpe` | `10x544` | `throughput_832k_896ctx` | `stable_muon97_compiled_longtail` | `fcproj_top4_attn_top4_fp16` | `1.1624` | `1.1503` | `15919169` | `0.993` | `target-stretch` |
| `4` | `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fcproj_top5_attn_top5_fp16` | `sp1024_bpe` | `10x544` | `throughput_832k_896ctx` | `stable_muon97_compiled_maxtrain` | `fcproj_top5_attn_top5_fp16` | `1.1624` | `1.1503` | `15999169` | `0.983` | `target-stretch` |
| `5` | `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fchi_top4_attn_top4_fp16` | `sp1024_bpe` | `10x544` | `throughput_832k_896ctx` | `stable_muon97_compiled_maxtrain` | `fchi_top4_attn_top4_fp16` | `1.1634` | `1.1513` | `15569169` | `0.983` | `target-stretch` |
| `6` | `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fchi_attnhi_fp16` | `sp1024_bpe` | `10x544` | `throughput_832k_896ctx` | `stable_muon97_compiled_longtail` | `fchi_attnhi_fp16` | `1.1634` | `1.1513` | `15659169` | `0.993` | `target-stretch` |

## Official-Stretch Candidates

| Rank | Candidate | Tokenizer | Model | Est. shipped | Best-case | Bytes | Runtime | Band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `3083` | `seeker_sp1024_bpe_sp_10x560_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fchi_top4_attn_top4_fp16` | `sp1024_bpe` | `10x560` | `1.1584` | `1.1463` | `15759169` | `1.002` | `target-stretch` |
| `3317` | `seeker_sp1024_bpe_sp_10x560_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fchi_attnhi_fp16` | `sp1024_bpe` | `10x560` | `1.1594` | `1.1473` | `15849169` | `1.002` | `target-stretch` |
| `3434` | `seeker_sp1024_bpe_sp_10x560_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fcproj_attnhi_fp16` | `sp1024_bpe` | `10x560` | `1.1594` | `1.1473` | `15979169` | `1.002` | `target-stretch` |
| `3467` | `seeker_sp1024_bpe_sp_10x560_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16` | `sp1024_bpe` | `10x560` | `1.1574` | `1.1453` | `15759169` | `1.012` | `target-stretch` |
| `3514` | `seeker_sp1024_bpe_sp_10x560_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fchi_attnhi_fp16_b128` | `sp1024_bpe` | `10x560` | `1.1604` | `1.1483` | `15689169` | `1.002` | `target-stretch` |
| `3645` | `seeker_sp1024_bpe_sp_10x560_kv2_throughput_832k_896ctx_stable_muon97_compiled_maxtrain_fcproj_attnhi_fp16_b128` | `sp1024_bpe` | `10x560` | `1.1604` | `1.1483` | `15829169` | `1.002` | `target-stretch` |

## Research Recovery Candidates

| Rank | Candidate | Source | Est. shipped | Best-case | Tier |
| --- | --- | --- | --- | --- | --- |
| `34808` | `seeker_sp1024_bpe_sp_10x576_kv2_throughput_832k_896ctx_depth_push_targeted_baseline_export_b128` | `offline_joint_search` | `1.1824` | `1.1703` | `research-only` |
| `35377` | `seeker_sp1024_bpe_sp_10x576_kv2_throughput_832k_896ctx_depth_push_targeted_baseline_export` | `offline_joint_search` | `1.1854` | `1.1733` | `research-only` |
| `35641` | `seeker_sp1024_bpe_sp_10x592_kv2_throughput_816k_896ctx_depth_push_targeted_baseline_export_b128` | `offline_joint_search` | `1.1814` | `1.1693` | `research-only` |
| `35832` | `seeker_sp1024_bpe_sp_11x560_kv2_throughput_832k_896ctx_depth_push_targeted_baseline_export_b128` | `offline_joint_search` | `1.1814` | `1.1693` | `research-only` |
| `35981` | `seeker_sp1024_bpe_sp_10x576_kv2_throughput_832k_896ctx_depth_push_baseline_export_b128` | `offline_joint_search` | `1.1864` | `1.1743` | `research-only` |
| `36123` | `seeker_sp1024_bpe_sp_12x528_kv2_throughput_816k_896ctx_depth_push_targeted_baseline_export_b128` | `offline_joint_search` | `1.1814` | `1.1693` | `research-only` |

## Commands

### `seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16`

- Tokenizer: `sp1024_bpe`
- Model: `10x544` / `kv=2` / `mlp=2`
- Token flow: `throughput_832k_896ctx`
- Training: `stable_muon97_compiled_longtail`
- Export: `fcproj_top5_attn_top5_fp16`
- Estimated shipped: `1.1614`
- Best-case shipped: `1.1493`
- Estimated bytes: `15999169`
- Runtime ratio: `0.993`
- Launch tier: `official-ready`

1xH100 screen:
```bash
RUN_ID=seeker_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16 \
  NPROC_PER_NODE=1 \
  MAX_WALLCLOCK_SECONDS=600 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=10 \
  NUM_UNIQUE_LAYERS=10 \
  MODEL_DIM=544 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=860160 \
  VAL_BATCH_SIZE=860160 \
  GRAD_CLIP_NORM=0.9 \
  GRAD_ACCUM_STEPS=2 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.5 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=1 \
  DDP_STATIC_GRAPH=0 \
  DDP_GRADIENT_AS_BUCKET_VIEW=0 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.9 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.5.mlp.fc.weight,blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.5.mlp.proj.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.5.mlp.fc.weight,blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.5.mlp.proj.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.5.attn.proj.weight,blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

6xH100 compute-matched:
```bash
RUN_ID=final_6x_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16 \
  NPROC_PER_NODE=6 \
  MAX_WALLCLOCK_SECONDS=800 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=10 \
  NUM_UNIQUE_LAYERS=10 \
  MODEL_DIM=544 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=860160 \
  VAL_BATCH_SIZE=860160 \
  GRAD_CLIP_NORM=0.9 \
  GRAD_ACCUM_STEPS=1 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.5 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=16 \
  DDP_STATIC_GRAPH=1 \
  DDP_GRADIENT_AS_BUCKET_VIEW=1 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.9 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.5.mlp.fc.weight,blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.5.mlp.proj.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.5.mlp.fc.weight,blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.5.mlp.proj.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.5.attn.proj.weight,blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

8xH100 final:
```bash
RUN_ID=final_sp1024_bpe_sp_10x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_top5_attn_top5_fp16 \
  NPROC_PER_NODE=8 \
  MAX_WALLCLOCK_SECONDS=600 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=10 \
  NUM_UNIQUE_LAYERS=10 \
  MODEL_DIM=544 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=860160 \
  VAL_BATCH_SIZE=860160 \
  GRAD_CLIP_NORM=0.9 \
  GRAD_ACCUM_STEPS=1 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.5 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=16 \
  DDP_STATIC_GRAPH=1 \
  DDP_GRADIENT_AS_BUCKET_VIEW=1 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.9 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.5.mlp.fc.weight,blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.5.mlp.proj.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.5.mlp.fc.weight,blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.5.mlp.proj.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.5.attn.proj.weight,blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

Rationale:
- Public `sp1024` baseline anchors shipped quality at `1.2244` bpb.
- Longer `sp1024` training reached `1.2074` shipped bpb, proving additional raw headroom exists.
- Calibrated against real `8xH100` anchor `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` (blend `0.15`, clean shift `+0.2524`, gap shift `+0.0808`).
- Model `sp_10x544_kv2` contributes `-0.0320` clean-bpb delta with byte delta `-150,000`.
- Token flow `throughput_832k_896ctx` contributes `-0.0210` clean and `+0.0000` export-gap delta.
- Training `stable_muon97_compiled_longtail` contributes `-0.0470` clean and `+0.0030` export-gap delta.
- Export `fcproj_top5_attn_top5_fp16` targets an optimistic gap floor of `0.0420`.
- Best-case shipped estimate is `1.1493`, still `0.0493` away from `1.1`.

### `seeker_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16`

- Tokenizer: `sp1024_bpe`
- Model: `10x560` / `kv=2` / `mlp=2`
- Token flow: `throughput_816k_896ctx`
- Training: `stable_muon97_compiled_longtail`
- Export: `fchi_top4_attn_top4_fp16`
- Estimated shipped: `1.1594`
- Best-case shipped: `1.1473`
- Estimated bytes: `15759169`
- Runtime ratio: `0.994`
- Launch tier: `official-ready`

1xH100 screen:
```bash
RUN_ID=seeker_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16 \
  NPROC_PER_NODE=1 \
  MAX_WALLCLOCK_SECONDS=600 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=10 \
  NUM_UNIQUE_LAYERS=10 \
  MODEL_DIM=560 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=817152 \
  VAL_BATCH_SIZE=817152 \
  GRAD_CLIP_NORM=0.88 \
  GRAD_ACCUM_STEPS=2 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.8 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=1 \
  DDP_STATIC_GRAPH=0 \
  DDP_GRADIENT_AS_BUCKET_VIEW=0 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.9 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

6xH100 compute-matched:
```bash
RUN_ID=final_6x_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16 \
  NPROC_PER_NODE=6 \
  MAX_WALLCLOCK_SECONDS=800 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=10 \
  NUM_UNIQUE_LAYERS=10 \
  MODEL_DIM=560 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=817152 \
  VAL_BATCH_SIZE=817152 \
  GRAD_CLIP_NORM=0.88 \
  GRAD_ACCUM_STEPS=1 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.8 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=16 \
  DDP_STATIC_GRAPH=1 \
  DDP_GRADIENT_AS_BUCKET_VIEW=1 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.9 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

8xH100 final:
```bash
RUN_ID=final_sp1024_bpe_sp_10x560_kv2_throughput_816k_896ctx_stable_muon97_compiled_longtail_fchi_top4_attn_top4_fp16 \
  NPROC_PER_NODE=8 \
  MAX_WALLCLOCK_SECONDS=600 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=10 \
  NUM_UNIQUE_LAYERS=10 \
  MODEL_DIM=560 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=817152 \
  VAL_BATCH_SIZE=817152 \
  GRAD_CLIP_NORM=0.88 \
  GRAD_ACCUM_STEPS=1 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.8 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=16 \
  DDP_STATIC_GRAPH=1 \
  DDP_GRADIENT_AS_BUCKET_VIEW=1 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.9 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

Rationale:
- Public `sp1024` baseline anchors shipped quality at `1.2244` bpb.
- Longer `sp1024` training reached `1.2074` shipped bpb, proving additional raw headroom exists.
- Calibrated against real `8xH100` anchor `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` (blend `0.15`, clean shift `+0.2524`, gap shift `+0.0808`).
- Model `sp_10x560_kv2` contributes `-0.0370` clean-bpb delta with byte delta `+40,000`.
- Token flow `throughput_816k_896ctx` contributes `-0.0190` clean and `+0.0000` export-gap delta.
- Training `stable_muon97_compiled_longtail` contributes `-0.0470` clean and `+0.0030` export-gap delta.
- Export `fchi_top4_attn_top4_fp16` targets an optimistic gap floor of `0.0430`.
- Best-case shipped estimate is `1.1473`, still `0.0473` away from `1.1`.

### `seeker_sp1024_bpe_sp_12x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_tok_attnhi_fp16`

- Tokenizer: `sp1024_bpe`
- Model: `12x544` / `kv=2` / `mlp=2`
- Token flow: `throughput_832k_896ctx`
- Training: `stable_muon97_compiled_longtail`
- Export: `fcproj_tok_attnhi_fp16`
- Estimated shipped: `1.1454`
- Best-case shipped: `1.1333`
- Estimated bytes: `17389169`
- Runtime ratio: `1.059`
- Launch tier: `illegal`

1xH100 screen:
```bash
RUN_ID=seeker_sp1024_bpe_sp_12x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_tok_attnhi_fp16 \
  NPROC_PER_NODE=1 \
  MAX_WALLCLOCK_SECONDS=600 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=544 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=860160 \
  VAL_BATCH_SIZE=860160 \
  GRAD_CLIP_NORM=0.9 \
  GRAD_ACCUM_STEPS=2 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.5 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=1 \
  DDP_STATIC_GRAPH=0 \
  DDP_GRADIENT_AS_BUCKET_VIEW=0 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.7 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight,blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

6xH100 compute-matched:
```bash
RUN_ID=final_6x_sp1024_bpe_sp_12x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_tok_attnhi_fp16 \
  NPROC_PER_NODE=6 \
  MAX_WALLCLOCK_SECONDS=800 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=544 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=860160 \
  VAL_BATCH_SIZE=860160 \
  GRAD_CLIP_NORM=0.9 \
  GRAD_ACCUM_STEPS=1 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.5 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=16 \
  DDP_STATIC_GRAPH=1 \
  DDP_GRADIENT_AS_BUCKET_VIEW=1 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.7 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight,blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

8xH100 final:
```bash
RUN_ID=final_sp1024_bpe_sp_12x544_kv2_throughput_832k_896ctx_stable_muon97_compiled_longtail_fcproj_tok_attnhi_fp16 \
  NPROC_PER_NODE=8 \
  MAX_WALLCLOCK_SECONDS=600 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=12 \
  NUM_UNIQUE_LAYERS=12 \
  MODEL_DIM=544 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=2 \
  MLP_MULT=2 \
  TIE_EMBEDDINGS=1 \
  INIT_MODEL_PATH= \
  TRAIN_SEQ_LEN=896 \
  TRAIN_BATCH_TOKENS=860160 \
  VAL_BATCH_SIZE=860160 \
  GRAD_CLIP_NORM=0.9 \
  GRAD_ACCUM_STEPS=1 \
  QK_GAIN_INIT=1.45 \
  ROPE_BASE=10000.0 \
  LOGIT_SOFTCAP=28.5 \
  LR_SCHEDULE=cosine \
  WARMUP_STEPS=24 \
  WARMDOWN_ITERS=2250 \
  LR_WARMUP_ITERS=24 \
  MIN_LR_SCALE=0.14 \
  TIED_EMBED_LR=0.00195 \
  MATRIX_LR=0.0013 \
  SCALAR_LR=0.0013 \
  TIED_EMBED_INIT_STD=0.0052 \
  MUON_MOMENTUM=0.97 \
  MUON_BACKEND_STEPS=6 \
  MUON_MOMENTUM_WARMUP_START=0.9 \
  MUON_MOMENTUM_WARMUP_STEPS=900 \
  BETA1=0.9 \
  BETA2=0.96 \
  ADAM_EPS=1e-08 \
  ITERATIONS=6600 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=100 \
  VAL_AT_STEP_ZERO=0 \
  WALLCLOCK_SYNC_EVERY=16 \
  DDP_STATIC_GRAPH=1 \
  DDP_GRADIENT_AS_BUCKET_VIEW=1 \
  LFQAT_START_STEP=0 \
  LFQAT_FULL_STEP=0 \
  LFQAT_MIN_PROB=1.0 \
  LFQAT_MAX_PROB=1.0 \
  LFQAT_KL_WEIGHT=0.0 \
  LFQAT_FISHER_WEIGHT=0.0 \
  LFQAT_TEMPERATURE=2.0 \
  TRAIN_COMPRESSION_AWARE_WEIGHT=0.0 \
  TRAIN_GRAD_ONLY_NAME_PATTERNS= \
  TRAIN_GRAD_SKIP_NAME_PATTERNS= \
  INT4_BLOCK_SIZE=64 \
  INT4_CLIP_PERCENTILE=99.7 \
  TRAIN_QAT_BLOCK_SIZE=64 \
  QUANT_FORMAT=mixed_int4_int8_packed_v2 \
  TARGET_EXPORT_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight \
  INT4_NAME_PATTERNS=blocks.6.mlp.fc.weight,blocks.7.mlp.fc.weight,blocks.8.mlp.fc.weight,blocks.9.mlp.fc.weight,blocks.10.mlp.fc.weight,blocks.11.mlp.fc.weight,blocks.6.mlp.proj.weight,blocks.7.mlp.proj.weight,blocks.8.mlp.proj.weight,blocks.9.mlp.proj.weight,blocks.10.mlp.proj.weight,blocks.11.mlp.proj.weight \
  TRAIN_QAT_NAME_PATTERNS= \
  TRAIN_COMPRESSION_AWARE_NAME_PATTERNS= \
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight,blocks.6.attn.proj.weight,blocks.7.attn.proj.weight,blocks.8.attn.proj.weight,blocks.9.attn.proj.weight,blocks.10.attn.proj.weight,blocks.11.attn.proj.weight \
  scripts/runpod/train_baseline_sp1024.sh
```

Rationale:
- Public `sp1024` baseline anchors shipped quality at `1.2244` bpb.
- Longer `sp1024` training reached `1.2074` shipped bpb, proving additional raw headroom exists.
- Calibrated against real `8xH100` anchor `final_sp1024_bpe_sp_10x512_kv2_context_896_dense_stable_muon97_int8_attnhi` (blend `0.15`, clean shift `+0.2524`, gap shift `+0.0808`).
- Model `sp_12x544_kv2` contributes `-0.0460` clean-bpb delta with byte delta `+1,050,000`.
- Token flow `throughput_832k_896ctx` contributes `-0.0210` clean and `+0.0000` export-gap delta.
- Training `stable_muon97_compiled_longtail` contributes `-0.0470` clean and `+0.0030` export-gap delta.
- Export `fcproj_tok_attnhi_fp16` targets an optimistic gap floor of `0.0400`.
- Best-case shipped estimate is `1.1333`, still `0.0333` away from `1.1`.
