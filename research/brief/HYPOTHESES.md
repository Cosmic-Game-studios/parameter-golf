# Hypotheses

## Active ranked hypotheses

1. **H1: the current `u4k share10 + bus20 + cross_skip` family is locally near-saturated**
   - Why: the reproduced win is real, but the next bounded variants (`busjoint`, token-pair stack, `cross_skip top2`) all lost.
   - Cheapest falsifier: one genuinely orthogonal branch should beat the current bar materially; otherwise this family is only a baseline, not the route to `0.9`.
   - Status: `supported`.

2. **H2: stronger scaling will require more effective depth per byte, not just better micro-controls on the same donor**
   - Why: tiny communication helps, but gains are microscopic. To approach the long-term target we likely need a model family that compounds better with more steps and more data.
   - Cheapest falsifier: a recurrent/shared-depth branch with stronger scaling logic still fails to beat the local bar or looks obviously worse per minute.
   - Status: `open`.

3. **H3: exporter and training-system upgrades remain first-class bottlenecks for official work**
   - Why: the best official run is still far behind the public frontier, and public winners keep using stronger compression and training-system stacks.
   - Cheapest falsifier: a structurally stronger local branch still does not look promotion-worthy even before those system gaps are closed.
   - Status: `open`.

4. **H4: tokenizer/token-flow remains a legitimate high-upside branch, but only after cheap screening**
   - Why: the target `0.8-0.9` is unlikely to come from architecture alone. Token efficiency may matter a lot, but naive token-pair features on the current donor already lost.
   - Cheapest falsifier: a cheap tokenizer/token-flow screen shows poor quality-per-byte or impossible artifact costs.
   - Status: `open`.

5. **H5: the next branch should be orthogonal or high-upside, not another minimal `cross_skip` variant**
   - Why: the new `top2` asymmetry test already lost.
   - Cheapest falsifier: one more strongly motivated communication variant wins clearly enough to justify reopening the family.
   - Status: `supported`.
