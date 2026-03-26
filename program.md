# Parameter Golf Autoresearch

You are an autonomous AI researcher. Your task is to improve a small language model's
**shipped val_bpb** (bits per byte after quantization export) by proposing and testing
modifications to `train_gpt_mlx.py`. You run an infinite loop of experiments, keeping
improvements and reverting failures, without asking for permission.

This repo participates in the OpenAI Parameter Golf Challenge:
- Total artifact must be < 16,000,000 bytes (model weights + code)
- Training must complete in ≤ 600 seconds on 8×H100 (official) or locally on M4 (smoke)
- Metric: shipped `val_bpb` on FineWeb validation set

## Read First

Before your first experiment, read these files in order:

1. `HANDOFF.md` – current frontier, recent results, active interpretation
2. `research/experiment_table.md` – all measured experiments
3. `research/autoresearch_results.tsv` – structured results log
4. The launcher script you will use (see below)

If sources disagree, trust the newer measured evidence in `HANDOFF.md`.

## Sacred Files (never modify)

- `eval_val()` in `train_gpt_mlx.py` – the evaluation function
- `quantize_state_dict_int8()` / `dequantize_state_dict_int8()` – the export pipeline
- `prepare.py` / data loading infrastructure (if it exists)

## Mutable Files (you may edit)

- `train_gpt_mlx.py` – model architecture, optimizer, hyperparameters, training loop
- `scripts/local/*.sh` – launcher scripts (prefer creating new ones over editing existing)

## The Loop

Run forever. Never stop. Never ask for permission.

```
1. SETUP
   - Create branch: git checkout -b autoresearch/<tag>
   - Read HANDOFF.md, research/autoresearch_results.tsv
   - Identify the current best shipped val_bpb anchor

2. BASELINE
   - If no baseline exists yet, run the current best launcher unmodified
   - Record: clean val_bpb, shipped val_bpb (from final_int8_zlib_roundtrip_exact),
     artifact bytes (from serialized_model_int8_zlib), export gap
   - This is your anchor. All experiments compare against this.

3. EXPERIMENT (repeat forever)
   a. Propose ONE modification. Think about what's most likely to help:
      - Architecture: layer count, width, sharing pattern, attention config
      - Optimizer: learning rates, schedules, warmup/warmdown, momentum
      - Quantization-aware training: LFQAT parameters, compression-aware weights
      - Export policy: which tensors get int4 vs int8 vs fp16
      - Training efficiency: batch size, sequence length, gradient accumulation
   b. Git commit the change with a descriptive message
   c. Run the experiment:
      bash scripts/local/<your_launcher>.sh > logs/<run_id>.txt 2>&1
   d. Extract results from the log:
      - grep "final_int8_zlib_roundtrip_exact" logs/<run_id>.txt   → shipped val_bpb
      - grep "val_bpb:" logs/<run_id>.txt | tail -1                → clean val_bpb
      - grep "serialized_model_int8_zlib:" logs/<run_id>.txt       → artifact bytes
   e. DECIDE:
      - If shipped val_bpb IMPROVED: KEEP the commit, update the anchor
      - If shipped val_bpb WORSE or EQUAL: REVERT via git reset --hard HEAD~1
      - If CRASHED: attempt one fix, retry once, then revert and move on
   f. Log the result to research/autoresearch_results.tsv (append a row):
      name<TAB>run_id<TAB>tier<TAB>status<TAB>clean_bpb<TAB>shipped_bpb<TAB>export_gap_bpb<TAB>note
   g. NEVER stop. Go back to step 3a.
```

## Environment

- Hardware: Apple M4 (local smoke runs, MLX framework)
- Training command: `python3 train_gpt_mlx.py` (launchers wrap this with env vars)
- Time budget per local experiment: ~10-80 minutes depending on iteration count
- Checkpoint format: `.npz` (raw) and `.int8.ptz` (quantized export)
- Log format: stdout/stderr, key lines prefixed with metric names

## Extracting Results

After each run completes, extract these exact values from the log:

```bash
# Clean val_bpb (the unquantized model quality)
grep "val_bpb:" logs/<run_id>.txt | tail -1
# → step:N/M val_loss:X.XXXX val_bpb:X.XXXX ...

# Shipped val_bpb (the quantized roundtrip quality — THIS IS THE REAL METRIC)
grep "final_int8_zlib_roundtrip_exact" logs/<run_id>.txt
# → final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX

# Artifact size (must be < 16,000,000 bytes for the total submission)
grep "serialized_model_int8_zlib:" logs/<run_id>.txt
# → serialized_model_int8_zlib:NNNNN bytes (payload:... raw_pickle:... payload_ratio:...)
```

## Keep / Revert Rules

**Keep** a change only if it earns measured evidence:
- Shipped `val_bpb` improves (lower is better) vs the current anchor
- Minimum meaningful improvement: 0.001 bpb
- Artifact size stays under budget (< 16,000,000 bytes total)

**Revert** if:
- Shipped `val_bpb` is worse or within noise (< 0.001 improvement)
- Artifact size exceeds budget
- Training crashed and cannot be quickly fixed

**Export gap** = shipped_bpb − clean_bpb. Lower is better. A change that improves
clean but worsens shipped (widens the gap) is NOT a keep.

## Current Anchors

Read these from `research/autoresearch_results.tsv` and `HANDOFF.md` at the start
of each session. The anchors below may be stale — always verify from measured data.

- Best local U4K shipped: ~1.9854 bpb (dense_u4k_12x608_kv2_lfqat60_fcproj)
- Best local sp1024 shared-depth shipped: ~2.1454 bpb (sp1024_12x576_share6_continue80_ultralowlr)
- Best official shipped: ~1.5106 bpb (final_sp1024_bpe_sp_10x512 on 8×H100)

## Experiment Strategy

Prioritize high-EV directions in this order:

1. **Training efficiency**: more useful tokens processed in the time budget
2. **Export gap reduction**: quantization-aware training, better export policies
3. **Architecture improvements**: width, depth, sharing, attention patterns
4. **Optimizer tuning**: learning rates, schedules, Muon parameters
5. **Novel ideas**: but only after the above are saturated

Do NOT:
- Run the same experiment twice expecting different results
- Chain more than 3 continuation tails without a structural change
- Add new pip dependencies (everything must ship in the artifact)
- Modify the evaluation or export pipeline
- Stream training logs into context (redirect to file, grep for results)

## Launcher Template

To create a new experiment, copy an existing launcher and override env vars:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Base: copy from scripts/local/train_lab_u4k_12x608_share10_crossskip_real80m_m4.sh
# Override only the parameters you're testing:
export RUN_ID="autoresearch_<experiment_name>"
export MATRIX_LR="0.0005"      # example: lower learning rate
export ITERATIONS="80"          # example: shorter run for smoke test
# ... then source the base or exec python3 train_gpt_mlx.py
```

## Active Priorities (as of 2026-03-23)

- The U4K shared-layer + cross-skip + bus architecture is the strongest local family
- The sp1024 12×576 share6 branch has the lowest export gap but worse absolute BPB
- LFQAT (learned fake quantization-aware training) is the primary export gap reducer
- Next high-EV moves: structural changes on the U4K donor, not more micro-tails
- Do not spend H100 time until a local experiment clearly beats the current official anchor direction
