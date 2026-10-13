# PEHT — folded-ternary tools for running a 27B LLM on phones

**PEHT = Pobozhii Egor — Hadamard Ternary.** The tools in this repo made a 27B ternary model
(Ternary-Bonsai-2, 2.06 bits/weight, 6.5 GiB) run natively on Android phones — and across two
phones over a USB cable.

Built by Egor Pobozhii, 17-year-old high school student, YCIS Qingdao.

## The finding

Ternary-Bonsai-2 is ternary **not in the primal basis, but in the Hadamard-rotated basis**:

- folded weights (as stored): pure {-d, 0, +d} per 128 weights, codes 33/33/33%, no +2d codes;
- primal weights (after un-rotating): Gaussian, kurtosis ~3.

Un-rotating and then re-quantizing to 2 bits collapses quality (perplexity > 10^6 in our tests).
So the only correct artifact keeps the weights folded and keeps the `prism.hadamard.*` keys:
the runtime applies the rotation to activations on the fly.

## What we did

1. **Lossless requantization** of the folded ternary from PQ2_0 to TQ2_0, keeping the rotation
   keys: **perplexity 14.2573 vs the 14.0477 reference (+1.5%)**, round-trip correlation 0.9985+.
2. **A runtime patch** for llama.cpp / llama.rn that parses `prism.hadamard.*` from the GGUF and
   applies the Hadamard rotation inside the graph (the same patch powers our Android app).

## Live numbers

| Platform | pp64 tok/s | tg32 tok/s |
|---|---|---|
| ASUS ROG Phone 7 (6 threads) | 3.93 | 2.88 |
| Redmi K100 Pro Max (8 threads) | 7.77 | 5.62 |
| Two phones, 70/30 layer split over USB (llama.cpp RPC) | - | ~2.8 end-to-end |

Baseline: the vendor's PQ2_0 format on the same phone is scalar (0.56 pp / 0.44 tg tok/s);
the folded-ternary path above is 12.8-13.9x faster at the same quality.

## Tools in this repo

| File | What it does |
|---|---|
| `convert_pq2_folded_to_tq2.py` | the main converter: PQ2_0 -> TQ2_0 keeping the weights folded and the `prism.hadamard.*` keys |
| `verify_folded.py` | byte-exact verification of a converted file (re-encodes source rows and compares bytes), plus round-trip error |
| `check_distribution.py` | prints folded code distribution and the primal Gaussian signature of a model |
| `dump_prism_kv.py` | prints the GGUF metadata, including the `prism.*` keys |
| `convert_pq2_0_to_tq2_0.py` | shared GGUF codec plus the earlier (rejected) primal-basis converter, kept for the record |
| `verify_convert.py` | legacy verifier for the earlier path, kept for the record |

Requirements: Python 3.9+ and numpy.

## Quick start

```bash
# requantize (a few minutes for the 27B on a laptop; no fp16 sources needed)
python convert_pq2_folded_to_tq2.py Ternary-Bonsai-2-27B-PQ2_0.gguf out-tq2-folded.gguf

# verify a few rows byte-exactly
python verify_folded.py src-pq2_0.gguf out-tq2-folded.gguf blk.0.ffn_down.weight 4

# inspect the folded/primal distributions
python check_distribution.py Ternary-Bonsai-2-27B-PQ2_0.gguf
```

The result runs on any runtime with the Hadamard transform (our PEHT runtime patch for
llama.cpp / llama.rn) and on the vendor's Android build.

## How it works (short)

The vendor stores the weights rotated: the runtime applies `R(s * x)` to the **activation**
(`R` = normalized Sylvester-Hadamard_1024, `s` = the sign vector from
`prism.hadamard.sign_values`). A folded weight therefore satisfies `W_f = W S R`.
Un-rotating gives `W = W_f R S`, but the primal is Gaussian and 2 bits destroy it.
The converter re-encodes `W_f` itself (a pure ternary grid) from per-128 blocks (PQ2_0) to
per-256 blocks (TQ2_0) and keeps the `prism.hadamard.*` keys; the runtime patch undoes the
rotation on activations, so correctness is preserved end to end.

## Links

- 2m45s one-take demo: https://youtube.com/shorts/CH1im2JR4yM
- Live numbers: https://gist.github.com/figkcoci-debug/76f41c409fc5afe5a431094cc228ddc0
- Method one-pager: https://gist.github.com/figkcoci-debug/11871f60812d2d18de5214fda3d94160
- Upstream RPC fix (llama.cpp PR): https://github.com/ggml-org/llama.cpp/pull/30415
- Vendor feedback thread: https://github.com/PrismML-Eng/Bonsai-demo/issues/260

## License

MIT, see LICENSE. Ternary-Bonsai-2 is a model by Prism ML and is governed by its own license;
these tools are unofficial and not affiliated with Prism ML.
