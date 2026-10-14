# How the TQ2_0 → TQ1_0 repack works (bit by bit)

This note explains exactly what `peht_packer.py` does to a GGUF file, why the repack is
**lossless by construction** (the weights come out bit-for-bit identical), and how that is
verified. It assumes no prior knowledge of GGML's ternary formats beyond the one fact that
makes everything possible: *in both formats the weights are already ternary*.

## TL;DR

- TQ2_0 and TQ1_0 store **the same weights**: per 256-weight block, a fp16 scale `d` and 256
  ternary digits (each weight is `-d`, `0` or `+d`).
- The only difference is **how the digits sit in bytes**: TQ2_0 uses 2-bit slots (2.0625
  bits/weight), TQ1_0 uses base-3 packing, five digits per byte (1.6875 bits/weight).
- So the conversion is a pure **re-base of the digits**, not a quantization: no floats are
  involved in the value path, the scale bytes are copied verbatim, and the original F16 model
  is never needed.
- 66 → 54 bytes per 256 weights: **-18% size** for the ternary tensors, same weights.
- Verified in three layers; on the real 27B file: **402 tensors / 2010 sampled rows — bit-exact**.

## Why this can be lossless at all

Quantization is lossy because it *derives* new values from floats. This tool never does that.
The input file already contains the final weights — every one of them is exactly one of three
states per block. Converting between containers cannot and does not change them.

What disappears is only the **wasted air**: a ternary value carries `log2(3) = 1.5849…` bits of
information, but a 2-bit slot always burns 2 bits. Multiplying that waste by 27B weights is
about 1.2 GiB — that is the entire win.

The zeros are **not** dropped and are not padding: about ⅓ of the weights are real zeros, and
their positions are information. Nothing here is a sparsity trick.

## Anatomy of the two blocks

Both formats process weights in blocks of 256 (`QK_K = 256`).

### TQ2_0 — 66 bytes per 256 weights (`2.0625` bpw)

```
[ 64 bytes: 2-bit codes ][ 2 bytes: fp16 scale d ]
```

- The 64 code bytes are read as two halves of 128 weights; byte `m` (`0..31`) of half `g`
  (`0` or `1`) holds four codes `l = 0..3` (bits `2l..2l+1`);
- natural index of the weight: `g*128 + l*32 + m`;
- value: `(code - 1) * d` → code 0 = `-d`, 1 = `0`, 2 = `+d`;
- code 3 would mean `+2d` — a fourth state. **TQ1_0 cannot hold it**, and the packer refuses
  such a tensor loudly instead of silently degrading it. (On the folded 27B model the +2d
  fraction is 0.0000%, so this never triggers.)

### TQ1_0 — 54 bytes per 256 weights (`1.6875` bpw)

```
[ 32 bytes: 160 digits, 5 per byte ][ 16 bytes: 80 digits, 5 per byte ]
[ 4 bytes: 16 digits, 4 per byte  ][ 2 bytes: fp16 scale d — same slot as TQ2_0 ]
```

- digits are the same ternary states: digit `0/1/2` → `-d/0/+d` (the constant `-1` mapping);
- the last 16 weights live four-per-byte in the four “high” bytes.

## The base-3 byte trick

Five ternary digits have `3^5 = 243` combinations — and a byte has 256. So five digits fit in
one byte: `243 ≤ 256`. GGML encodes the digit group

```
q = d0*81 + d1*27 + d2*9 + d3*3 + d4        (a number in 0..242)
```

as a byte `b = ceil(q * 256 / 243)`, computed as `(q*256 + 242) // 243`. The decoder walks the
digits back with pure modular arithmetic, no tables:

```
digit(n) = ((b * 3^n mod 256) * 3) >> 8
```

Worked example — digits `(2, 0, 1, 0, 2)`:

```
q = 2*81 + 1*9 + 2           = 173
b = (173*256 + 242) // 243   = 183
decode n=0: (183 * 1   mod 256)*3 >> 8 = 183*3 >> 8   = 2
decode n=1: (183 * 3   mod 256)*3 >> 8 =  37*3 >> 8   = 0
decode n=2: (183 * 9   mod 256)*3 >> 8 = 111*3 >> 8   = 1
decode n=3: (183 * 27  mod 256)*3 >> 8 =  77*3 >> 8   = 0
decode n=4: (183 * 81  mod 256)*3 >> 8 = 231*3 >> 8   = 2
```

`peht_packer.py` uses exactly this encoder and this decoder for the three digit regions
(160 digits in 32 bytes, 80 in 16 bytes, 16 in 4 bytes at a coarser 4-per-byte packing with a
×3 range shift — the same math GGML's `quantize_row_tq1_0_ref` / `dequantize_row_tq1_0` use).

## The algorithm

```
parse GGUF:              metadata (kept as-is), tensor table, alignment
build the new table:     TQ2_0 -> TQ1_0   (66 -> 54 bytes per block; offsets recomputed)
                         F32/F16/BF16     (norms, biases) copied as-is
                         anything else    -> abort (no silent conversions)
general.file_type:       -> 36 (LLAMA_FTYPE_MOSTLY_TQ1_0)
for every ternary tensor, streamed in chunks:
    read 66-byte blocks
    split the 2-bit codes, reassemble digits in the natural index order
    pack digit groups into the 54-byte layout (three regions above)
    copy the 2 scale bytes UNTOUCHED
    write
output:                  written to .tmp-peht, then atomically renamed
```

No tensor data ever passes through floats, and no metadata key is dropped — including the
`prism.hadamard.*` keys of folded models.

## How it is verified

Three layers, all in the tool:

1. **Self-test** (`--selftest`): 100 000 random blocks are packed, unpacked and dequantized
   with a Python mirror of GGML's decoder; the float32 bit patterns are compared
   (`np.array_equal` on `uint32` views). Any packing bug fails here.
2. **Row verification** (`--verify`, on by default in our runs): for every converted tensor
   rows are sampled (first, middle, last, plus two random), decoded from *both* files and
   compared as float32 bit patterns — i.e. "does the dequantized value of weight `i` match
   between input and output, exactly".
3. **The real run**: on the 27B folded model — 402 tensors, 2010 sampled rows, **bit-exact**;
   file 6.996 GB → 5.737 GB.

## What this tool does NOT do

- **It does not requantize.** No float path, no original F16 sources, no quality math.
- **It does not un-fold or un-rotate anything.** Folded models stay folded and keep their
  `prism.hadamard.*` keys; correct output still needs a Hadamard-aware runtime (our PEHT
  runtime patch, or the vendor's build). The container itself is a stock GGML type (34) —
  stock builds **load** it, unlike the vendor's 142/143 — but a correct answer on a folded
  model requires the matching activation transform.
- **It refuses what it cannot preserve**: a TQ2_0 tensor containing the fourth state
  (code 3 = `+2d`) aborts the run — better loud than lossy.
- **Kernel speed is a separate question.** TQ1_0 is not automatically faster than TQ2_0 on
  every CPU (2-bit shifts are cheap); the win here is size and memory bandwidth. On our
  phones the TQ1_0 file runs at the same speeds as before — the decoder is native in the
  build.

## Reproduce

```bash
pip install numpy tqdm
python peht_packer.py --selftest
python peht_packer.py model-tq2.gguf model-tq1.gguf --verify
```

## Credit and context

TQ1_0 and TQ2_0 are GGML's (llama.cpp) native ternary types, introduced for ternary LLMs
(see llama.cpp PR #8151). The repacker, its verification tooling and this note are part of
the **PEHT** tools (`Pobozhii Egor — Hadamard Ternary`), built for running a folded-ternary
27B model on two phones. To our knowledge, this is the first *public* tool that converts
between the two packed ternary formats directly on a packed GGUF — no re-quantization, no
float detour, with byte-exact verification included. If you know of an earlier one, tell us —
we will link it here.
