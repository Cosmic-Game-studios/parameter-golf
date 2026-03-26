# Autonomy Policy

## Runtime mode
- Operate in bounded sessions, but keep the mission active until redirected, blocked, or superseded by stronger evidence.
- Default to one bounded experiment cycle per session unless there is a compelling reason to do more.

## Portable lab runtime
- The repo does not currently ship `tools/research_lab.py`.
- Until a portable CLI exists, emulate the runtime manually through repo-stored state:
  - `research/STATE.json`
  - `research/brief/*`
  - `research/plan/*`
  - `research/logs/*`
  - `research/reports/*`
  - `research/memory/*`
  - `research/handoffs/NEXT_SESSION.md`
- Do not rely on host-only shell aliases, GUI actions, or non-repo state.

## Session policy
- Start by syncing state, baseline, scoreboard, decisions, negatives, and handoff.
- Choose one high-EV next unit.
- Run it to a real decision when feasible.
- Write down the exact next action before stopping.

## Escalation policy
- If a branch saturates locally, do not keep polishing it by default.
- Escalate to:
  - a more orthogonal local branch
  - a higher-upside local branch
  - or an H100 promotion candidate if local evidence justifies it
