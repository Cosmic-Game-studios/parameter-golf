# Public Frontier Memo (2026-03-21)

## Scope

Public Parameter Golf frontier only, as visible on `2026-03-21`.

Ground-truth caveat: the public leaderboard lags the strongest open PRs. The repo README explicitly says leaderboard updates can take time and that PRs are the submission path:
- [README leaderboard + rules](https://github.com/openai/parameter-golf/blob/main/README.md)

## Strongest merged evidence

These are the strongest ideas that are already merged into `main`, so evidence strength is `strong`.

1. `1.1428` — `10L Int5-MLP + BigramHash(10240)`
   - Ingredients that look causal: mixed `int5/int6`, `BigramHash(10240)`, `SWA`, `WD=0.04`.
   - URL: [leaderboard row](https://github.com/openai/parameter-golf/blob/main/README.md), [record README](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md)

2. `1.1458` — `Int6 MLP3x + SmearGate + BigramHash`
   - Ingredients: `MLP 3x`, `SmearGate`, `BigramHash`, `OrthoInit`, `Muon WD`, `SWA`.
   - URL: [leaderboard row](https://github.com/openai/parameter-golf/blob/main/README.md), [record README](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md)

3. `1.1502` — `11L MLP3x + Int6 QAT`
   - Ingredients: `11L`, `MLP 3x`, `int6 QAT`, `zstd-22`, `WD=0.04`, `sliding stride=64`.
   - URL: [leaderboard row](https://github.com/openai/parameter-golf/blob/main/README.md), [record README](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md)

4. `1.1556` — `SmearGate + OrthoInit + Muon WD`
   - Ingredients: `SmearGate`, `BigramHash(4096)`, `MLP 3x`, `int6 STE QAT`, `sliding stride=64`.
   - URL: [leaderboard row](https://github.com/openai/parameter-golf/blob/main/README.md), [record README](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md)

5. `1.1586` — `10L Int6 QAT + Zstd MLP2.6x`
   - Ingredients: full `int6 QAT`, `zstd-22`, wider `MLP 1344`, `FP16` tied embedding, `seq_len=2048`, `Muon 0.99`, `sliding stride=64`.
   - URL: [leaderboard row](https://github.com/openai/parameter-golf/blob/main/README.md), [record README](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/README.md)

## Strongest open PR claims

These are not merged, so evidence is weaker than the merged frontier even when the writeup is strong.

1. `#315` — `1.1248` best seed, `1.1250` 3-seed mean, `15.6 MB`
   - Evidence strength: `moderate-strong`.
   - Why: exact recipe, 3 seeds, under cap, clear delta over prior public stack.
   - Core ingredients: `11L`, `XSA` on last 4 layers, `EMA(0.997)`, `SmearGate + BigramHash(2048)`, mixed `int6 + zstd-22`, `Partial RoPE`, `LN Scale`, `Late QAT`, `WD=0.04`, `seq_len=2048`.
   - URL: [PR #315](https://github.com/openai/parameter-golf/pull/315)

2. `#338` — `1.1254` best seed, `1.1256` 3-seed mean, `15.55 MB`
   - Evidence strength: `moderate`.
   - Why: 3 seeds and clean timing, but it is built on open PR `#315`, not merged code.
   - Core ingredients: same `#315` stack plus bounded `TTT`; the PR claims TTT buys about `~0.002` bpb and costs about `47s`.
   - URL: [PR #338](https://github.com/openai/parameter-golf/pull/338)

3. `#332` — `1.1320` 3-seed mean, `15.65 MB`
   - Evidence strength: `moderate-strong`.
   - Why: 3 seeds, under cap, exact run command, and a concrete negative result on Late QAT at 12L.
   - Core ingredients: `12L`, gradient-guided mixed `int5/int6/int7`, `Partial RoPE`, `LN Scale`, `XSA4`, `EMA`, `seq_len=2048`, lower batch for more steps.
   - URL: [PR #332](https://github.com/openai/parameter-golf/pull/332)

4. `#339` — `1.1364`, but `16,170,051` bytes
   - Evidence strength: `suggestive only`.
   - Why: the PR itself says the artifact is over cap, so it is not a legal record yet.
   - Core ingredients: `Backout` residual subtraction, `11L`, `MLP 3x`, `SmearGate`, `BigramHash(4096)`, `SWA`, mixed `int6`.
   - URL: [PR #339](https://github.com/openai/parameter-golf/pull/339)

5. `#348` — `1.1444`, `15.90 MB`
   - Evidence strength: `suggestive`.
   - Why: single-seed on top of the merged SOTA, plus at least one public review note flagging a README/code mismatch about `SWA every 25` vs `50`.
   - Core ingredients: `QAT`, bigger `BigramHash(12288)`, `stride=32`, `5%` pruning.
   - URL: [PR #348](https://github.com/openai/parameter-golf/pull/348)

## What still looks highest-EV for us

Compared against the current repo, we already have `slide64` evaluation and real optimizer weight decay in the main training stack. I therefore excluded those from the missing-ideas ranking.

### Top 5 missing ideas

1. Mixed `int5/int6` or adaptive `int5/int6/int7` export plus `zstd-22`
   - Why it ranks first: this is the byte-funding layer behind nearly every best public run. It repeatedly buys enough budget for deeper or wider models without giving the gain back at ship time.
   - Strong evidence: merged `1.1428`, `1.1502`, `1.1586`; open `#315`, `#332`.
   - URLs: [top merged leaderboard](https://github.com/openai/parameter-golf/blob/main/README.md), [PR #315](https://github.com/openai/parameter-golf/pull/315), [PR #332](https://github.com/openai/parameter-golf/pull/332)

2. One averaging path, but treat `EMA` vs `SWA` as stack-dependent
   - Why it ranks second: it is all over the frontier, and it is one of the cheapest high-upside additions relative to architecture changes.
   - Skeptical note: `#315` and `#338` lean on `EMA`, but `#333` explicitly reports that `EMA(0.997)` caused a large quant gap on that stack while `SWA` worked better. So the missing idea is not “EMA specifically”; it is “proper averaging with A/B measurement.”
   - URLs: [PR #315](https://github.com/openai/parameter-golf/pull/315), [PR #338](https://github.com/openai/parameter-golf/pull/338), [PR #333](https://github.com/openai/parameter-golf/pull/333)

3. Token-pair feature stack: `SmearGate` + `BigramHash`
   - Why it ranks third: this is the most repeated architecture-side motif across both merged winners and the best open claims. The evidence is much stronger than for most exotic attention variants.
   - Strong evidence: merged `1.1458`, `1.1556`; open `#315`, `#333`, `#339`, `#348`.
   - URLs: [merged SmearGate run](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md), [PR #315](https://github.com/openai/parameter-golf/pull/315), [PR #333](https://github.com/openai/parameter-golf/pull/333)

4. Zero-parameter structure stack: `XSA`, `Partial RoPE`, `LN Scale`
   - Why it ranks fourth: these are the cleanest “free” model-quality levers in the best open claims, and `#315` suggests they compound on top of the public SmearGate/Bigram/QAT stack.
   - Evidence strength: `moderate`, because the best evidence is still unmerged PR frontier, but it is highly consistent across `#315`, `#332`, and `#338`.
   - URLs: [PR #315](https://github.com/openai/parameter-golf/pull/315), [PR #332](https://github.com/openai/parameter-golf/pull/332), [PR #338](https://github.com/openai/parameter-golf/pull/338)

5. Late or bounded quantization adaptation, with TTT as the last-mile variant
   - Why it ranks fifth: the frontier is now showing two flavors of “adapt at the end” rather than throughout training.
   - Split judgment:
     - `Late QAT` looks useful at `11L` in `#315`, but `#332` says it loses at `12L` because throughput cost dominates.
     - `TTT` in `#338` looks real but small, around `~0.002` bpb, so it is probably a final-mile move once the base stack is already near the frontier.
   - URLs: [PR #315](https://github.com/openai/parameter-golf/pull/315), [PR #332](https://github.com/openai/parameter-golf/pull/332), [PR #338](https://github.com/openai/parameter-golf/pull/338)

## Low-EV or too-speculative right now

- Pure long-context escalation to `4096`:
  - evidence exists, but it is not what the current best merged or open records are winning with.
  - URL: [4096-context merged run](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md)
- `stride=32` as a primary bet:
  - plausible as a small eval win, but current public evidence is weaker than the other ideas and at least one PR has reproducibility/readme mismatch concerns.
  - URL: [PR #348](https://github.com/openai/parameter-golf/pull/348)
- `Backout` as the next main bet:
  - interesting, but current public evidence is only a single over-cap PR.
  - URL: [PR #339](https://github.com/openai/parameter-golf/pull/339)

## Bottom line

If the goal is to beat roughly `1.125` shipped `bpb`, the public frontier says the missing core is not another local micro-tail. It is a full-stack jump toward:

1. mixed/adaptive low-bit export with `zstd`,
2. measured averaging (`EMA` or `SWA`),
3. token-pair features,
4. `XSA + Partial RoPE + LN Scale`,
5. then a bounded endgame adaptation path (`Late QAT` or `TTT`).
