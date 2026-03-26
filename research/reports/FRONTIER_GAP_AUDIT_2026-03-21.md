# Frontier Gap Audit (2026-03-21)

## Scope

This audit inspects the current repo state for the next public-frontier levers that matter before the next real `8xH100 / 600s` run.

Primary evidence:
- exporter and training code in `train_gpt.py`, `train_gpt_mlx.py`, `quant_reconstruction.py`, and `research/search_export_gap.py`
- recent internal decision memos:
  - `research/public_frontier_review_2026-03-21.md`
  - `research/web_breakthrough_scan_2026-03-21.md`
  - `research/brief/RESOURCE_PLAN.md`
  - `research/reports/SCOREBOARD.md`
  - `research/local_u5k_12x608_seq896_codebook_transfer.md`

Status legend:
- `implemented`: materially present and usable now
- `partial`: meaningful groundwork exists, but the named frontier idea is not actually landed
- `missing`: not present in the code path that matters

## Already-landed adjacent frontier work

These are not the focus of the gap audit, but they matter for prioritization because they reduce the remaining gap:

- `slide64` validation is implemented in `train_gpt.py:62` and `train_gpt.py:247-352`.
- optimizer weight decay is implemented in `train_gpt.py:102-104` and already wired into the breakthrough H100 launchers per `research/public_frontier_review_2026-03-21.md:55-63`.
- the exporter stack is already materially richer than plain int8:
  - int4/blockwise mixed export in `train_gpt.py:792-798` and `train_gpt_mlx.py:1159-1165`
  - nonuniform/codebook int4 export in `train_gpt.py:792-793` and `train_gpt_mlx.py:1159-1160`
  - low-rank residual repair / QER-style hooks in `train_gpt.py:799-810` and `train_gpt_mlx.py:1166-1173`

## Gap matrix

| Area | Status | Repo evidence | Audit read | Why it matters |
| --- | --- | --- | --- | --- |
| `zstd` artifact compression | `missing` | Both final export paths hardcode `zlib`: `train_gpt.py:406-408`, `train_gpt.py:1703-1728`, `train_gpt_mlx.py:954-959`, `train_gpt_mlx.py:1676-1692` | The public review and web scan still call out `int5/int6 + zstd` as a major missing lever: `research/public_frontier_review_2026-03-21.md:16-20`, `research/web_breakthrough_scan_2026-03-21.md:7-13`, `research/web_breakthrough_scan_2026-03-21.md:56-67` | Compression is the cheapest way to buy architecture bytes back without retraining the whole stack. |
| mixed `int5/int6` export | `missing` | The only supported low-bit paths are int8 and int4-based formats: `train_gpt.py:421-425`, `train_gpt.py:474-479`, `train_gpt.py:792-798`, `train_gpt_mlx.py:972-980`, `train_gpt_mlx.py:1011-1016`, `train_gpt_mlx.py:1159-1165` | Still listed as missing in the scan and plan: `research/web_breakthrough_scan_2026-03-21.md:58-60`, `research/plan/TODO.md:11-13`, `research/brief/RESOURCE_PLAN.md:29-32` | The public best runs are explicitly using medium bitwidths to fund stronger models. |
| `EMA` / `SWA` checkpoint averaging | `missing` | The only `EMA` in core code is LFQAT Fisher statistics, not model averaging: `train_gpt.py:454-457`, `train_gpt_mlx.py:204-210` | Docs still flag averaging as absent: `research/web_breakthrough_scan_2026-03-21.md:60-61`, `research/brief/RESOURCE_PLAN.md:29-32` | This is the easiest train-time generalization lever that does not change inference structure or artifact layout. |
| token-pair features (`SmearGate`, `BigramHash`) | `missing` | No `BigramHash`, `SmearGate`, or token-pair feature path exists in the active training code. Repo-wide searches only hit memos. | The scan marks both as absent: `research/web_breakthrough_scan_2026-03-21.md:61-63` | Public evidence says these can move the frontier, but they require a new input feature path rather than just exporter work. |
| token-processing modernization around the tokenizer | `partial` | Tokenizer-family experimentation and embedding transplant tooling exist in `research/transplant_tokenizer_checkpoint.py:16-153` | The repo has active `u5k` / `u6k` token-flow work, but not pairwise runtime features | Good groundwork exists, but it does not yet implement the public pair-feature ideas. |
| `XSA` | `missing` | No `XSA` code path or config is present in the active Torch or MLX stack | Listed as absent in the scan: `research/web_breakthrough_scan_2026-03-21.md:64-67` | Potentially strong, but it touches the attention core and has more throughput risk than averaging/export work. |
| `Partial RoPE` | `missing` | Current attention only exposes `rope_base` and standard RoPE application: `train_gpt.py:76`, `train_gpt.py:1034-1047`, `train_gpt_mlx.py:89-90`, `train_gpt_mlx.py:657-669` | The scan calls out `Partial RoPE` as absent: `research/web_breakthrough_scan_2026-03-21.md:42-45`, `research/web_breakthrough_scan_2026-03-21.md:64-67` | This is likely the easiest of the zero-parameter attention/position tweaks. |
| `LN Scale` | `partial` | The repo already has learned residual/branch scaling controls: `train_gpt.py:410-412`, `train_gpt.py:1140-1143`, `train_gpt.py:1189-1191`, `train_gpt_mlx.py:213-217`, `train_gpt_mlx.py:760-763`, `train_gpt_mlx.py:805-807` | But there is no explicit LayerNorm/RMSNorm scale parameter added as a named frontier feature; the scan still lists `LN Scale` as missing: `research/web_breakthrough_scan_2026-03-21.md:64-67` | We already have some of the effect space, so the marginal value of a dedicated LN-scale patch is lower than a truly missing lever. |
| `Backout` | `missing` | The stack has additive skip/residual mixing, not the named backout mechanism: `train_gpt.py:1140-1143`, `train_gpt.py:1200-1207`, `train_gpt_mlx.py:760-763`, `train_gpt_mlx.py:819-825` | The scan still marks `Backout` as absent: `research/web_breakthrough_scan_2026-03-21.md:49-54`, `research/web_breakthrough_scan_2026-03-21.md:64-67` | Promising, but the semantics are less clear and the code risk is higher than EMA or exporter work. |
| exporter portability across Torch and MLX | `partial` | Torch and MLX implement very similar export formats and metadata contracts: `train_gpt.py:753-832`, `train_gpt_mlx.py:1132-1187` | Good local parity exists, but it is duplicated rather than centralized | This is workable today, but every new exporter invention must currently be landed twice. |
| exporter portability across families / tokenizers | `partial` with strong negative evidence | The `share6` exporter frontier did not transfer to the alive `u5k 12x608` branch: `research/public_frontier_review_2026-03-21.md:41-46`, `research/local_u5k_12x608_seq896_codebook_transfer.md:8-39`, `research/reports/SCOREBOARD.md:27-39` | The best legal `u5k` export remained `fcproj_hi`, while codebook/QER variants all regressed and went over cap | This is the main reason not to assume a local exporter win is universally H100-ready. |

## Feasibility notes

- `zstd` is feasible on this workstation right now via the installed CLI (`/opt/homebrew/bin/zstd`, version `1.5.7`).
- There is no `compression.zstd`, `zstandard`, or `zstd` Python module in either `python3` or `./.venv/bin/python`.
- Practical implication: the easiest first repo integration is not “hard-wire a Python zstd dependency”, but “introduce a compressor abstraction with a safe `zlib` fallback and optional `zstd` backend”.

## Main judgment

The repo is **not** missing exporter sophistication in general. It already has:
- int8/int4 export
- codebook int4
- low-rank residual repair
- exporter sweep tooling
- family-transfer probes

But it is still missing the **public-frontier medium-bit compression + averaging stack**:
- no `zstd`
- no `int5/int6`
- no `EMA/SWA`

That means the next high-EV work unit should still be a **pre-H100 integration pass**, not another blind local microtail.

## Top 3 easiest high-EV integrations before the next real H100 run

### 1. Add `EMA` to the official Torch stack

Why first:
- smallest code change with the clearest public precedent
- train-time only; no counted artifact format change
- low integration risk versus `SWA`
- can be toggled behind a single env flag and evaluated on the exact current H100 recipes

Suggested shape:
- maintain a shadow copy of trainable parameters during training
- export/eval both raw and EMA checkpoints at the end
- start with `EMA_DECAY=0.997` and make it opt-in

Why `EMA` over `SWA`:
- simpler under a fixed 10-minute budget
- better match to the public branches cited in the scan

### 2. Add a portable compressor abstraction with `zstd` support and `zlib` fallback

Why second:
- high EV even before any new training run, because it can be A/B tested on existing checkpoints
- the workstation already has a `zstd` CLI, so repo prototyping is straightforward
- it also improves exporter portability discipline by separating `quantization policy` from `final compressor`

Suggested shape:
- introduce `COMPRESSOR=zlib|zstd`
- keep the current `zlib` path as the default fallback
- add a `zstd` backend that shells out to the CLI if available, or cleanly falls back when unavailable
- log compressor choice and compressed bytes explicitly in both Torch and MLX

Why not treat this as “done” once local CLI works:
- H100 portability still requires a deterministic fallback story
- the backend must not silently depend on host-specific tooling

### 3. Add an `int6` exporter path before attempting `int5`

Why third:
- lower implementation risk than `int5`
- directly targets the biggest remaining artifact-economics gap in the public frontier
- fits naturally into the existing `search_export_gap.py` policy machinery

Suggested shape:
- add `INT6_NAME_PATTERNS`
- add a new format such as `mixed_int6_int8_packed_v1`
- restrict first use to a narrow upper-layer corridor so measurement stays clean
- only after `int6` is stable, consider `int5` for a small subset such as upper MLP tensors

Why `int6` before `int5`:
- simpler packing / dequant logic
- lower quality-risk
- enough to tell us whether medium-bit export is the real remaining byte lever

## What should wait until after those 3

- token-pair features (`BigramHash`, `SmearGate`): promising, but a bigger train-path change than `EMA`
- `XSA`: likely higher risk than the current easiest wins
- `Backout`: interesting, but less clearly specified in our current codebase than `EMA` or `Partial RoPE`
- `Partial RoPE`: probably the easiest zero-parameter structure tweak, but still second-wave relative to `EMA + zstd + int6`

If we want one structural hedge in parallel with the top 3, `Partial RoPE` is the cleanest candidate.

## Recommended order for the next pre-H100 block

1. `EMA`
2. compressor abstraction with `zstd`
3. `int6` exporter backend
4. re-run exporter sweeps on current best checkpoints
5. only then choose the next real H100 branch
