# Current Task

## Task name and slug
- Name: Local tokenizer and token-flow breakthrough
- Slug: `2026-03-21-local-tokenizer-tokenflow-breakthrough`

## Task type
- Model training / local fixed-budget improvement loop

## Desired outcome
- Beat or meaningfully challenge the best measured legal local anchor `dense_u4k_12x608_kv2_lfqat60_fcproj` at `1.98544845` shipped bpb by exploiting the alive `u5k` tokenizer-transfer branch through token-flow improvements with real local evidence.

## Non-goals
- No H100 runs in this cycle.
- No claims from token-count proxies alone.
- No reopening of stale same-family continuation chains without a new orthogonal lever.
- No destructive changes to existing local winners.

## Execution plan
1. Treat `dense_u4k_12x608_kv2_lfqat60_fcproj` as the legal local baseline.
2. Keep exact-copy tokenizer transplant as fixed infrastructure.
3. Reject raw tokenizer families quickly with token-count economics plus eval-only gates.
4. Spend the next bounded training cycle on token-flow tuning of the alive `u5k` branch.
5. Keep only measured wins; kill raw tokenizer families that lose the gate.
6. Reopen a new tokenizer family only if the current `u5k` token-flow line clearly plateaus.

## Memory refresh notes
- Working memory: this file
- Episodic memory: `research/autoresearch_results.tsv`, `improvement/ledger.jsonl`
- Learned memory: `improvement/patterns.md`
- Procedural memory: root `AGENTS.md`, `program.md`, and the user-provided `swe-self-improve` instructions

## Fast-loop evals
- Token-count and manifest comparison for any new tokenizer family
- Eval-only exact-copy gate before any new raw tokenizer continuation
- One bounded local MLX continuation with `VAL_MAX_TOKENS=262144` on the alive `u5k` branch
- Shipped roundtrip metric from the end-of-run exporter

## Full gates
- Fixed-settings export sweep on the exact resulting checkpoint using `research/search_export_gap.py` if the branch is alive
- `python3 -m unittest research.test_autoresearch_loop research.test_search_export_gap research.test_sota_seeker research.test_quant_reconstruction`

## Primary metric
- Shipped `val_bpb` on the local fixed-budget proxy, lower is better

## Secondary metrics
- Clean `val_bpb`
- Export gap `shipped - clean`
- Counted total bytes
- Runtime
- Token count on the same frozen local corpus
- Whether the branch preserves a plausible path to later `8xH100 / 600s`

## Evaluation commands
- Current alive branch anchor:
  - `RUN_ID=lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy INIT_MODEL_PATH=./logs/lab_u5k_12x608_kv2_transplant_lowlr40_fcproj_exactcopy_mlx_model.npz TRAIN_SEQ_LEN=896 ITERATIONS=40 TIED_EMBED_LR=0.0012 MATRIX_LR=0.0008 SCALAR_LR=0.0008 LFQAT_FULL_STEP=5 LFQAT_MIN_PROB=0.50 LFQAT_KL_WEIGHT=0.015 LFQAT_FISHER_WEIGHT=0.003 TRAIN_COMPRESSION_AWARE_WEIGHT=0.0008 bash scripts/local/train_lab_u5k_12x608_transplant_lowlr40_m4.sh`
- Quick raw-family filters:
  - `RUN_ID=debug_u6k_12x608_transplant_eval0_exactcopy DATA_PATH=./data/local_u6k_unigram/datasets/fineweb10B_spu6144_local TOKENIZER_PATH=./data/local_u6k_unigram/tokenizers/fineweb_6144_unigram.model VOCAB_SIZE=6144 INIT_MODEL_PATH=./logs/lab_u6k_12x608_kv2_transplant_init_mlx_model.npz ITERATIONS=0 VAL_LOSS_EVERY=0 bash scripts/local/train_lab_u5k_12x608_transplant_lowlr40_m4.sh`
  - `RUN_ID=debug_b4k_12x608_transplant_eval0_exactcopy DATA_PATH=./data/local_b4k_bpe/datasets/fineweb10B_spb4096_local TOKENIZER_PATH=./data/local_b4k_bpe/tokenizers/fineweb_4096_bpe.model VOCAB_SIZE=4096 INIT_MODEL_PATH=./logs/lab_b4k_12x608_kv2_transplant_init_mlx_model.npz ITERATIONS=0 VAL_LOSS_EVERY=0 bash scripts/local/train_lab_u5k_12x608_transplant_lowlr40_m4.sh`
- Fixed export sweep:
  - `python3 research/search_export_gap.py --preset local_u5k_12x608_lfqat60_m4 --checkpoint <checkpoint> --out-json <json> --out-md <md> --top-k 16`
- Regression tests:
  - `python3 -m unittest research.test_autoresearch_loop research.test_search_export_gap research.test_sota_seeker research.test_quant_reconstruction`

## Iteration budget
- Raw tokenizer families get one exact-copy eval-only gate.
- The alive `u5k` branch gets at most two more bounded token-flow continuations before another full pivot.

## Rollback / checkpoint plan
- Global baseline is `dense_u4k_12x608_kv2_lfqat60_fcproj` at `1.98544845` shipped.
- Local tokenizer/token-flow anchor is now `lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy` at `2.10282311` shipped.
- Keep only if a new `u5k` token-flow run beats that anchor or clearly earns one more bounded follow-up.
- Otherwise retain the existing winner and pivot again.

## Stop conditions
- Stop after the current `u5k` token-flow branch either beats the local anchor or clearly plateaus.
- Stop early if a raw tokenizer family is weak on token economics or eval-only shipped quality.
- Stop once another same-family `u5k` token-flow continuation becomes flat enough that a new orthogonal lever is lower entropy.
