# Improvement Protocol

Protocol: `autoresearch-principles`

## Fixed rules
- Exactly one narrow hypothesis or change unit per bounded cycle.
- Every claimed gain must beat a named baseline under the same eval frame.
- Every cycle must end in one of: `keep`, `revert`, `crash`, `park`.
- Prefer the cheaper decisive experiment over the bigger speculative one.
- Do not promote a branch on one noisy win when cheap reproduction is still missing.

## Comparison contract
- Baseline must be written down before the run.
- Use the same tokenizer family, validation subset, exporter family, and counted-code path unless the hypothesis explicitly changes one of them.
- Distinguish:
  - checkpoint win
  - exporter win
  - systems/runtime win
  - structural falsification

## Branch taxonomy
- `conservative`: exploit the best live donor with a tiny, low-risk change
- `orthogonal`: change mechanism, not just knob values
- `high-upside`: potentially large improvement, but only after cheaper falsification gates
- `reproduction`: repeat the best story or a near-winner to clear noise
- `diagnostic`: isolate why a branch won or lost

## Promotion gates
- Local `keep`: better shipped `val_bpb` under comparable conditions
- Local `provisional`: win exists but reproduction or exporter check missing
- Official promotion:
  - reproduced or critic-cleared local win, or
  - a clearly orthogonal/high-upside branch with enough supporting evidence to justify H100 spend

## Critic gates
- Run a critic gate before promotion when any of the following is true:
  - the measured gain is inside a tiny noise band
  - the branch changes comparison frame, exporter family, tokenizer family, or counted-code path
  - the branch is the first apparent winner in a new family
  - the branch is being considered for official H100 spend
- A critic gate must explicitly answer:
  - did the branch beat a named baseline under a truly comparable eval frame?
  - is the win a checkpoint win, exporter win, or systems/runtime win?
  - is reproduction still cheap enough that we should reproduce before promotion?
  - is byte accounting explicit and still legal?
  - is there a real scaling story, or only a better local snapshot?
- Critic outcomes:
  - `clear`: safe to treat as a real keep or promotion candidate
  - `reproduce-first`: positive but too fragile to promote yet
  - `hold`: interesting, but another branch has higher EV right now
  - `reject`: do not promote; treat as `revert` or `park`

## Plateau rules
- If a family improves only by microscopic deltas across multiple cycles, force an orthogonal or high-upside branch next.
- Do not spend another cycle on a line already refuted under equivalent conditions unless new evidence materially changes the prior.
