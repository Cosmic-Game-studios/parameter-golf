# Open Questions

- Which orthogonal local branch has the strongest scaling rationale now that the first post-reset sparse/asymmetric `cross_skip` branch already lost?
- Is the next real breakthrough more likely to come from effective depth / recurrence, tokenizer efficiency, or a stronger exporter/training-stack co-design branch?
- What is the cheapest local branch that could materially change our official upside story rather than only shaving micro-bits off a saturated donor?
- Is the right middle ground on local `u4k` a partial-sharing geometry such as `12` layers with `8-10` unique layers, rather than full depth or `share6`?
- Can a pure local `u4k` token-processing or token-flow change on the `share10 dualrole + bus20` donor beat `1.98453610`, now that the first boundary-aware bus v2 branch has been refuted?
- Is there any exporter on the current `u4k share6` checkpoint meaningfully better than `projhi_attnhi_fp16`, or is that frontier effectively settled?
- Why does width-up on the `u4k share6 first_cycle` branch regress despite ample byte headroom: optimization budget, bad copied geometry, or real overcapacity under the local training cap?
- Which pure token-flow or token-locality change is cheapest and most likely to help the live `share10 dualrole + bus20` donor now that the first token-pair stack has mostly failed: sequence packing, boundary-weighted token mixing outside the bus, or a materially smaller `BigramHash` path on a stronger donor?
