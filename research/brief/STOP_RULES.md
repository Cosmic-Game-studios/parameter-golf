# Stop Rules

- Stop a local branch immediately if it is clearly worse than the active baseline under the same exporter and eval frame.
- Treat a result as `revert` when:
  - it loses to baseline under comparable conditions
  - and no orthogonal evidence suggests it deserves immediate reproduction
- Treat a result as `park` when:
  - it is close enough to matter
  - but another branch has higher EV right now
- Treat a result as `crash` when:
  - the harness, runtime, or save path fails before a valid metric exists
- Promote to H100 only when:
  - the branch is reproduced or critic-cleared
  - and it offers either a material local gain or a strong structural/scaling reason to believe official transfer could be much larger
- Block promotion when:
  - the result is only a near-tie and a cheap reproduction is still available
  - the apparent win depends on a changed eval frame that has not been isolated
  - the branch has no explicit official-upside story beyond “local was a bit better”

## Current family-specific stops
- Do not reopen `cross_skip + global_bus` joint retraining on the active donor.
- Do not reopen `SmearGate` on the active donor.
- Do not reopen the first token-pair stack on the active donor unchanged.
- Do not spend another default cycle on the refuted `cross_skip top2` branch.
- If a branch is only a microscopic variant of the current donor and not clearly orthogonal, prefer an orthogonal branch next.
