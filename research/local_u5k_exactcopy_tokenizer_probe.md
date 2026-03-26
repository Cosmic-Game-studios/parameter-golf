# Local `u5k` Exact-Copy Tokenizer Probe

## Goal
- Test whether a larger local unigram tokenizer can become competitive with the locked `u4k` legal anchor once tokenizer transfer is initialized correctly.

## Token economics
- Baseline tokenizer: `u4k` unigram
  - train tokens: `73,332,949`
  - val tokens: `45,480,154`
- New tokenizer: `u5k` unigram
  - train tokens: `70,489,732`
  - val tokens: `43,715,916`
- Delta vs `u4k`:
  - train: `-2,843,217` tokens (`-3.8771%`)
  - val: `-1,764,238` tokens (`-3.8791%`)

## Transplant finding
- Old transplant path was too lossy because it averaged many rows that already existed exactly in the source tokenizer.
- New exact-copy-aware transplant summary:
  - special rows copied: `4`
  - exact piece rows copied: `3,971`
  - averaged rows: `1,145`
  - random rows: `0`
  - mapping coverage: `1.0`

## Measured runs

### 1. Naive transplant eval-only gate
- Log: [debug_u5k_12x608_transplant_eval0.txt](/Users/ronaldschmidt/openai/logs/debug_u5k_12x608_transplant_eval0.txt)
- clean: `8.2197 / 3.4791`
- shipped: `8.18444729 / 3.46422313`
- Verdict: clearly bad initialization

### 2. Exact-copy transplant eval-only gate
- Log: [debug_u5k_12x608_transplant_eval0_exactcopy.txt](/Users/ronaldschmidt/openai/logs/debug_u5k_12x608_transplant_eval0_exactcopy.txt)
- clean: `5.1376 / 2.1746`
- shipped: `5.13459635 / 2.17331566`
- Verdict: breakthrough in initialization quality; the branch becomes immediately competitive enough to continue

### 3. Exact-copy low-LR continuation
- Log: [lab_u5k_12x608_kv2_transplant_lowlr40_fcproj_exactcopy.txt](/Users/ronaldschmidt/openai/logs/lab_u5k_12x608_kv2_transplant_lowlr40_fcproj_exactcopy.txt)
- clean: `4.9819 / 2.1087`
- shipped: `4.97053623 / 2.10387409`
- compressed model: `15,304,828` bytes
- Verdict: real keep over the exact-copy gate (`2.1733 -> 2.1039`)

### 4. Exact-copy ultralow continuation
- Log: [lab_u5k_12x608_kv2_transplant_continue40_ultralow_exactcopy.txt](/Users/ronaldschmidt/openai/logs/lab_u5k_12x608_kv2_transplant_continue40_ultralow_exactcopy.txt)
- clean: `4.9833 / 2.1093`
- shipped: `4.97065735 / 2.10392536`
- compressed model: `15,303,411` bytes
- Verdict: plateau / no keep versus the first low-LR continuation

## Conclusion
- Larger local unigram is not dead.
- It only becomes viable with an exact-piece-aware tokenizer transplant.
- After that fix, the `u5k 12x608` branch is alive but still behind the locked `u4k` local anchor at `1.98544845`.
- The next local tokenizer move should be sideways, not another identical `u5k` tail:
  - either a smaller unigram family
  - or a local BPE branch with the same transfer discipline
