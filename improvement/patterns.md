## Pattern: tokenizer transfer between related SentencePiece vocabularies should exact-copy shared pieces before averaging anything
- Context: the first `u5k` tokenizer branch looked dead under a decode/re-encode averaging transplant, even though the new tokenizer reduced token count by almost `3.9%`.
- Signal: once the transplant copied exact shared SentencePiece rows first, the eval-only shipped gate improved from `3.4642` to `2.1733` without any training.
- Caveat: do not judge a new tokenizer family from a lossy transplant. For closely related tokenizers on the same corpus, exact piece overlap is high-value initialization signal.

## Pattern: verify tiny local wins with a full fixed-settings export sweep
- Context: the `sp1024 10x560` line often improves clean quality or raw shipped score by a tiny amount while staying on the same legal exporter.
- Signal: the local winner only moved by `~0.00026` shipped bpb, and the fixed export sweep confirmed that `projhi_attnhi_fp16` still remained the best legal policy.
- Caveat: tiny local wins are still real, but they should close a loop rather than open a new continuation chain unless a structurally new hypothesis appears.

## Pattern: shared-depth branches solve shipping before they solve raw quality
- Context: `12x544 share6` was clearly weaker than the global anchor on clean/shipped quality, but its best legal exporter was almost lossless from the first scratch checkpoint.
- Signal: both the scratch and continuation checkpoints kept `proj_top6_attn_top6_fp16` on top with only `~0.0022-0.0023` bpb export gap.
- Caveat: do not waste early shared-depth cycles on new export geometry. Once the gap is that small, the next lever is raw-quality improvement via width, steps, or recurrence choices.

## Pattern: once the shared-depth width-up wins, keep exploiting the same exporter
- Context: `12x560 share6` improved through scratch, `continue120`, `continue80_lowlr`, and `continue80_ultralowlr`.
- Signal: every fixed export sweep on the branch still picked `proj_top6_attn_top6_fp16`.
- Caveat: when the exporter remains fixed and the branch keeps dropping shipped `val_bpb`, another bounded continuation is higher-EV than reopening export-only search.

## Pattern: a wider shared-depth branch may need one adaptation run before the real signal appears
- Context: `12x576 share6` booted from the stronger `12x560 femtotail` checkpoint started behind the anchor at `2.2402` shipped, but the next two low-LR continuations improved to `2.1770` and `2.1454`.
- Signal: the widened branch stayed legal, kept a tiny export gap, and held nearly the same step time as the smaller winner.
- Caveat: do not kill a width-up shared-depth branch after the first boot pass if it is close, stable, and still compression-friendly. Give it one bounded low-LR follow-up.

## Pattern: partial export-sweep logs are a tooling hazard, not evidence
- Context: a full `search_export_gap.py` sweep on the stronger `12x560 share6 picotail` checkpoint stalled on non-winning policies and left partial logs behind.
- Signal: the frontier was still recoverable with a fresh targeted conservative check, and the cache layer needed to validate-or-rerun incomplete logs.
- Caveat: do not rank or compare policies from a cached sweep unless the log parses to full clean + roundtrip metrics.

## Pattern: a local end-of-run crash near `saved_model` can be a disk-space failure, not a model failure
- Context: the first `u4k 12x608` reramp continuations appeared to die silently between the final train step and `saved_model`, which initially looked like an MLX finalization bug.
- Signal: once the repo volume was nearly full, runs failed exactly as raw and quantized artifacts were being written; after freeing space, the exact same recipe finalized cleanly and produced a real shipped metric.
- Caveat: before treating a late local crash as model or framework evidence, check free disk and enforce a preflight floor in the launcher.

## Pattern: once a tokenizer-transfer branch is alive, token-flow is higher-EV than opening another raw tokenizer family
- Context: the revived `u5k` exact-copy branch survived into the low `2.10x` range, while the next two raw tokenizer probes (`u6k` unigram and `b4k` BPE) both lost their eval-only gates.
- Signal: `u6k` saved more tokens than `u5k` but still gated worse at `2.1997`, `b4k` was clearly bad at `2.7321`, and a simple `u5k seq_len=896` retune still produced a small real keep at `2.1028`.
- Caveat: do not keep opening new tokenizers just because they look interesting on token counts. Once one branch is alive, spend the next cycle on token-flow there until it clearly flattens.
