# Web Breakthrough Scan (2026-03-21)

## Bottom line

We are **not** at a hard wall.

The strongest evidence from the public Parameter Golf frontier and recent quantization literature says our current stack is still missing several large, plausible levers:

1. `int5/int6` mixed export plus `zstd-22`
2. `EMA` / `SWA`
3. token-pair features such as `SmearGate` + `BigramHash`
4. stronger zero-parameter attention/position tweaks such as `XSA`, `Partial RoPE`, `LN scale`, or `Backout`
5. saliency-/rotation-aware quantization ideas that are stronger than our current plain codebook+QER exporter

So the right conclusion is **not** “run H100 immediately because nothing better exists.”
The right conclusion is: there is still clear room, and the next work unit should be a **pre-H100 integration pass on the biggest missing public levers**.

## Public frontier signals

### Visible public bar is already below the README top line

- README leaderboard top still shows `1.1428`: [README](https://raw.githubusercontent.com/openai/parameter-golf/main/README.md)
- But stronger visible open PR claims on `2026-03-21` are already lower:
  - `#338`: `1.1254` sliding, `1.1256` 3-seed mean: [PR #338](https://github.com/openai/parameter-golf/pull/338)
  - `#332`: `1.1320`: [PR #332](https://github.com/openai/parameter-golf/pull/332)
  - `#339`: `1.1364`, but over cap: [PR #339](https://github.com/openai/parameter-golf/pull/339)

### What the best public branches are actually combining

From [PR #338](https://github.com/openai/parameter-golf/pull/338):
- `11L`, `512d`, `MLP 3x`
- `XSA` on last 4 layers
- `EMA` (`0.997`)
- `SmearGate + BigramHash(2048)`
- `Int6 QAT + Late QAT + zstd-22`
- `Muon WD=0.04`
- `TTT` gives about `~0.002` post-quant BPB

From [PR #332](https://github.com/openai/parameter-golf/pull/332):
- `12L` funded by gradient-guided adaptive quantization
- top tensors `int7`, middle `int6`, bottom `int5`
- `Partial RoPE`
- `LN Scale`
- `XSA` on last 4 layers
- `EMA` replacing `SWA`
- smaller batch gave more steps and better fixed-budget quality
- explicit negative: late QAT can lose when throughput cost is too high

From [PR #339](https://github.com/openai/parameter-golf/pull/339):
- `Backout` connection gave `-0.0071` sliding BPB over their own baseline
- `11L`, `MLP 3x`, `SmearGate`, `BigramHash(4096)`, `OrthoInit`
- `SWA`
- mixed `int6` quant + `zstd`
- artifact was only slightly over cap and fixable via `INT5_MLP=1`

## What our current code still lacks

Repo scan on `2026-03-21`:
- no real `zstd` exporter path
- no `int5` / `int6` quantization path
- no `EMA` or `SWA` training path
- no `BigramHash`
- no `SmearGate`
- no `XSA`
- no `Partial RoPE`
- no `LN Scale`
- no `Backout`

We **did** already close two meaningful gaps:
- `slide64` evaluation
- proper optimizer `weight decay`

That means we are closer, but still not on the same ingredients stack as the public best runs.

## ArXiv ideas that actually look relevant

### 1. Saliency-aware error reconstruction

- [SERQ](https://arxiv.org/abs/2603.08185): saliency-aware low-rank error reconstruction

Why it matters for us:
- we already have a working `QER` exporter
- the most direct next step is to make residual repair **saliency-aware**, not just plain low-rank
- this is the closest literature-backed extension to our strongest local exporter branch

Verdict:
- very promising **local** and **H100-transfer** exporter idea
- higher EV than another blind microtail

### 2. Rotation-aware quantization

- [SpinQuant](https://arxiv.org/abs/2405.16406)
- [QuaRot](https://arxiv.org/abs/2404.00456)
- [FlatQuant](https://arxiv.org/abs/2410.09426)

Why it matters for us:
- these methods attack outliers before quantization
- our current exporter is still sensitive to branch/family transfer
- an offline fused rotation/affine transform could make low-bit export more portable than the current `share6`-specific codebook/QER win

Verdict:
- strong orthogonal research direction
- probably too much to rush into the very next H100 run
- good candidate for the next exporter invention pass if `int5/int6+zstd` still leaves a gap

### 3. Better nonuniform/codebook quantization

- [QuIP#](https://arxiv.org/abs/2402.04396)
- [HIGGS](https://arxiv.org/abs/2411.17525)
- [AQLM](https://arxiv.org/pdf/2401.06118)

Why it matters for us:
- `QuIP#` and `HIGGS` both support the idea that lattice / nonuniform codebooks and better bit allocation matter
- `HIGGS` is especially relevant because it explicitly connects layer-wise error to perplexity and supports optimal medium-bitwidth allocation under a byte budget
- `AQLM` says more expressive codebooks can preserve quality at lower effective bits, but its implementation cost is probably too high for our immediate challenge loop

Verdict:
- `HIGGS`-style nonuniform bit allocation is the highest-EV derivative for us
- direct `AQLM` replication is likely too heavy for the next bounded loop

## What this means for our project

## We are not at a wall

The clearest reason is simple:
- the strongest public runs are winning with ingredients we still do not have
- therefore our current ceiling is partly self-imposed

## The biggest missing levers are not local microtails

The next major gains probably come from:
1. better **artifact economics**
2. better **training averaging / generalization**
3. stronger **token interaction structure**
4. stronger **bit allocation**

not from squeezing another `0.001` off the same local `12x576 share6` tail.

## Ranked next experiments

### Conservative

Implement before the next real H100 run:
- `zstd` artifact compression
- mixed `int5/int6` export
- at least one averaging path: `EMA` or `SWA`

Why:
- this is the cleanest path to matching the public recipe stack
- it is also directly useful even if we later add more exotic structure

### Orthogonal

Local exporter research:
- `SERQ`-style saliency-aware QER
- `HIGGS`-style nonuniform bit allocation under a fixed byte budget

Why:
- both directly extend the exporter branch we already know how to measure
- both could let us make the model smaller without losing quality, which then funds a larger/better H100 model

### High-upside / wildcard

Structural hybrid:
- `SmearGate + BigramHash`
- maybe `Backout`
- maybe `XSA`/`Partial RoPE`/`LN scale`

Why:
- public evidence says these ideas can move the frontier materially
- but they are larger code changes, so they come after the compression/averaging pass unless we explicitly decide to swing bigger

## Decision

Do **not** interpret the current local plateau as a global wall.

We should **not** rush to H100 yet if the goal is a real breakthrough attempt.

The strongest immediate case is:
1. implement `zstd` + mixed `int5/int6`
2. add `EMA` or `SWA`
3. then run the conservative H100 branch

If we skip those and launch immediately, we are still knowingly leaving some of the strongest public levers on the table.
