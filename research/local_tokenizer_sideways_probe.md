# Local Tokenizer Sideways Probe

## Goal
- Test whether the next tokenizer move after the revived `u5k` branch should be a larger unigram, a sideways BPE family, or a token-flow retune on the already-alive branch.

## Frozen local token economics

| Family | Train tokens | Val tokens | Delta vs `u4k` train | Delta vs `u4k` val |
| --- | ---: | ---: | ---: | ---: |
| `u4k` unigram | `73,332,949` | `45,480,154` | baseline | baseline |
| `u5k` unigram | `70,489,732` | `43,715,916` | `-3.8771%` | `-3.8791%` |
| `u6k` unigram | `68,350,121` | `42,394,997` | `-6.7941%` | `-6.7822%` |
| `b4k` bpe | `73,388,814` | `45,529,579` | `+0.0762%` | `+0.1087%` |

## Exact-copy transplant overlap

| Family | Exact copied rows | Averaged rows | Random rows | Shared-piece overlap |
| --- | ---: | ---: | ---: | ---: |
| `u5k` unigram | `3,971` | `1,145` | `0` | `77.64%` |
| `u6k` unigram | `4,090` | `2,050` | `0` | `66.63%` |
| `b4k` bpe | `2,478` | `1,614` | `0` | `60.60%` |

## Measured gates and follow-up

### 1. `u6k` unigram exact-copy gate
- Log: [debug_u6k_12x608_transplant_eval0_exactcopy.txt](/Users/ronaldschmidt/openai/logs/debug_u6k_12x608_transplant_eval0_exactcopy.txt)
- clean: `5.3293 / 2.1992`
- shipped: `5.33034468 / 2.19966345`
- compressed model: `15,866,955` bytes
- Verdict: more token savings alone do not win. `u6k` improves token count materially over `u5k`, but starts from a worse gate.

### 2. `b4k` BPE exact-copy gate
- Log: [debug_b4k_12x608_transplant_eval0_exactcopy.txt](/Users/ronaldschmidt/openai/logs/debug_b4k_12x608_transplant_eval0_exactcopy.txt)
- clean: `6.2499 / 2.7376`
- shipped: `6.23736286 / 2.73211769`
- compressed model: `14,732,174` bytes
- Verdict: kill. This BPE branch has slightly *more* tokens than `u4k` and a clearly bad starting gate.

### 3. `u5k` token-flow retune at `seq_len=896`
- Log: [lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy.txt](/Users/ronaldschmidt/openai/logs/lab_u5k_12x608_kv2_transplant_seq896_continue40_exactcopy.txt)
- init checkpoint: [lab_u5k_12x608_kv2_transplant_lowlr40_fcproj_exactcopy_mlx_model.npz](/Users/ronaldschmidt/openai/logs/lab_u5k_12x608_kv2_transplant_lowlr40_fcproj_exactcopy_mlx_model.npz)
- clean: `4.9851 / 2.1087`
- shipped: `4.97108459 / 2.10282311`
- compressed model: `15,302,484` bytes
- Runtime: `206.1s` train + `13.9s` roundtrip eval
- Verdict: real keep. This is only a small gain over the best `u5k` `1024`-context tail (`2.10387409`), but it is the first measured sign that token-flow on the alive tokenizer branch is a better lever than opening another raw tokenizer family immediately.

## Conclusion
- `u5k` exact-copy transfer is still the best tokenizer-transfer branch.
- `u6k` is not the next winner even though it saves more tokens.
- `b4k` BPE is not competitive on this frozen local corpus.
- The next highest-EV local token-processing move is on the alive `u5k` branch itself:
  - keep exact-copy transfer
  - keep the current legal export family
  - continue token-flow tuning rather than opening another raw tokenizer family first
