# Quant/Export Breakthrough Memo (2026-03-21)

## Scope

Question: which recent quantization/export ideas are most worth adapting into this repo for **Parameter Golf** under:

- `< 16,000,000` counted bytes total
- `8xH100 / 600s` training budget
- dense Transformer models
- strong preference for methods that improve **shipped** quality without a huge code or runtime tax

## Bottom line

The highest-EV next export stack is **not** a giant new quantization framework. It is:

1. `int5/int6` mixed export plus `zstd`
2. budget-aware per-layer bit allocation
3. saliency-aware low-rank reconstruction on top of our existing QER path

Rotation-aware and richer codebook methods still look real, but they are better as a second wave unless the simpler path saturates.

## Ranked top 5

| Rank | Idea | Why it may help here | Likely difficulty | Sources |
| --- | --- | --- | --- | --- |
| 1 | **Mixed `int5/int6` + `zstd` exporter** | This is the most direct challenge-specific lever. The best visible Parameter Golf PRs are already using medium-bit exports plus `zstd`, which suggests we are still leaving easy artifact economics on the table. In this repo, it fits our current exporter architecture: we already have layer-pattern routing, mixed-precision decisions, and artifact accounting. The likely payoff is not just smaller artifacts, but funding wider `MLP` or more aggressive keep-float choices while staying under cap. | **Low-Medium**. Add `int5`/`int6` packers, a `zstd` compression backend, and extend the policy search over layer families. | Challenge rules: [README](https://raw.githubusercontent.com/openai/parameter-golf/main/README.md). Challenge-specific practical evidence: [PR #338](https://github.com/openai/parameter-golf/pull/338), [PR #332](https://github.com/openai/parameter-golf/pull/332), [PR #339](https://github.com/openai/parameter-golf/pull/339) |
| 2 | **Budget-aware medium-bit allocation** | We do not just need “better quantization”; we need the **best loss per byte** under a hard artifact budget. Recent work on fractional/optimal-bit allocation is highly aligned with that objective. The repo already has explicit per-tensor family routing, so the practical adaptation is a simpler version: use a sensitivity score and assign tensors to `int5`, `int6`, `int8`, or keep-float tiers under a byte cap, rather than implementing the full paper kernels. | **Medium**. We can reuse the current exporter search and add a byte-budgeted allocator without adopting the full research stack. | [Q-Palette](https://arxiv.org/abs/2509.20214) |
| 3 | **Saliency-aware low-rank error reconstruction** | This is the best direct extension of what the repo already knows how to do. We already have QER/low-rank reconstruction codepaths; the missing piece is better targeting. SERQ’s core idea is to weight reconstruction by saliency instead of using a plain residual fit. For us, that is attractive because it aims to reduce the clean-to-shipped gap exactly where our exporter loses quality, while keeping the extra machinery compact. | **Medium**. We already have residual reconstruction and some gradient/Fisher signals in the codebase, so the jump is incremental rather than architectural. | [SERQ](https://arxiv.org/abs/2603.08185) |
| 4 | **Rotation-aware / flattening-aware preconditioning** | Rotation-based quantization is an orthogonal lever: instead of choosing better quantizers only, it makes the weights easier to quantize in the first place by suppressing outliers. This matters for us because our best exporter improvements have not transferred cleanly across families. An offline fused rotation or affine transform could make lower-bit export more portable than the current branch-specific codebook/QER wins. | **Medium-High**. The simplest version is an offline/fused transform on selected weight matrices; the full learned-rotation path is heavier. | [QuaRot](https://arxiv.org/abs/2404.00456), [SpinQuant](https://arxiv.org/abs/2405.16406), [FlatQuant](https://arxiv.org/abs/2410.09426) |
| 5 | **Structured codebook / additive quantization** | This family is still one of the strongest routes when the objective is “tiny shipped artifact with minimal quality loss.” We already have basic codebook ideas in the repo, so the next serious step would be to move from scalar/block quantization toward better vector/codebook structure on only the most byte-expensive tensors. The warning is complexity: these methods can win strongly in extreme compression, but they also risk growing decode logic and counted code too much for this challenge if copied wholesale. | **High**. Best treated as a selective export path on a few tensor families, not a repo-wide replacement. | [QuIP#](https://proceedings.mlr.press/v235/tseng24a.html), [AQLM](https://arxiv.org/abs/2401.06118) |

## Recommendation for this repo

If the goal is the **highest expected-value** next integration pass, the order should be:

1. add `zstd` plus real `int5`/`int6` export
2. add a byte-budgeted bit allocator over the existing exporter policy search
3. upgrade QER to saliency-aware reconstruction

Only after that would I spend serious time on rotation-aware fusion or a larger codebook rewrite.

## Why this ranking fits our codebase

- We already have mixed exporter policies, counted-byte accounting, and exporter sweeps.
- We already have low-rank reconstruction infrastructure, so SERQ is a natural derivative.
- We do **not** yet have a lean `zstd` path or medium-bit exporter, even though the public frontier is using them.
- Full AQLM/VPTQ-style rewrites are probably too large for the current counted-code budget unless done very selectively.

## Not top-5 despite being interesting

- **Full VPTQ / extreme vector PTQ**: promising for very low bits, but likely too much machinery for near-term Parameter Golf integration. Source: [VPTQ](https://arxiv.org/abs/2409.17066)
- **Learnable butterfly rotations**: high upside, but more engineering risk than standard rotation/flattening methods. Source: [ButterflyQuant](https://arxiv.org/abs/2509.09679)

## Suggested immediate experiments

1. Implement `zstd` + `int5/int6` and rerun the exporter frontier on the alive modern checkpoint.
2. Add a greedy byte-budgeted allocator that chooses among `int5`, `int6`, `int8`, and keep-float per tensor family.
3. Replace plain low-rank residual fitting with a saliency-weighted objective using the signals we already log or can cheaply estimate.
