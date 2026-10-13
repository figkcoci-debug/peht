#!/usr/bin/env python3
"""verify_folded.py — byte-exact verification of a folded requantization.

Re-encodes rows of the source PQ2_0 file with the same math as
convert_pq2_folded_to_tq2.py (values stay folded; per-256 scale = amax; trits re-rounded)
and compares the bytes with what the converted file actually contains. Also reports the
round-trip error between the two files.

Usage:
  python verify_folded.py src-pq2.gguf dst-tq2.gguf TENSOR [NROWS] [ROW_START]

Example:
  python verify_folded.py Ternary-Bonsai-2-27B-PQ2_0.gguf folded.gguf blk.0.ffn_down.weight 4 4000
  (ROW_START lets you probe chunk boundaries of the converter)
"""

import importlib.util as u
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = u.spec_from_file_location("conv", os.path.join(HERE, "convert_pq2_0_to_tq2_0.py"))
mod = u.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    src = mod.parse(sys.argv[1])
    dst = mod.parse(sys.argv[2])
    name = sys.argv[3]
    nrows = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    r0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    t_src = next(t for t in src["tensors"] if t["name"] == name)
    t_dst = next(t for t in dst["tensors"] if t["name"] == name)
    if t_dst["type"] != mod.GGML_TYPE_TQ2_0:
        raise SystemExit(f"dst tensor type is {t_dst['type']}, expected TQ2_0")

    k = t_src["dims"][0]
    total = 1
    for d in t_src["dims"]:
        total *= d
    rows_total = total // k
    if r0 + nrows > rows_total:
        raise SystemExit("requested rows out of range")

    src_row = k // mod.QK_PQ2_0 * mod.PQ2_0_BLOCK
    dst_row = k // mod.QK_K * mod.TQ2_0_BLOCK

    # 1) decode the folded source rows
    f = src["f"]
    f.seek(src["data_start"] + t_src["offs"] + r0 * src_row)
    blob = f.read(nrows * src_row)
    a = np.frombuffer(blob, dtype=np.uint8).reshape(-1, mod.PQ2_0_BLOCK)
    d = a[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
    codes = ((a[:, 2:].reshape(-1, 32, 1) >> (2 * np.arange(4))) & 3).reshape(-1, mod.QK_PQ2_0)
    vals = ((codes.astype(np.float32) - 1.0) * d[:, None]).reshape(-1, mod.QK_K)

    # 2) re-encode exactly like the folded converter
    amax = np.abs(vals).max(axis=1)
    s = np.where(amax > 0, amax, 1.0).astype(np.float32)
    trits = np.clip(np.floor(np.abs(vals) / s[:, None] + 0.5) * np.sign(vals), -1, 1).astype(np.int8)
    expected = mod.pack_tq2_0(trits, s)

    # 3) read the real bytes from the converted file
    g = dst["f"]
    g.seek(dst["data_start"] + t_dst["offs"] + r0 * dst_row)
    actual = g.read(nrows * dst_row)

    same = actual == expected
    print(f"tensor {name}: k={k}, rows {r0}..{r0 + nrows - 1} of {rows_total}")
    print(f"  bytes: expected {len(expected)}, read {len(actual)}")
    print(f"  BYTE-MATCH: {'YES' if same else 'NO'}")
    if not same:
        diffs = [i for i in range(min(len(actual), len(expected))) if actual[i] != expected[i]]
        print(f"  diffs: {len(diffs)} of {len(actual)}; first: {diffs[:10]}")
        print(f"    in scales (idx%66>=64): {sum(1 for i in diffs if i % 66 >= 64)}")
        print(f"    in codes  (idx%66< 64): {sum(1 for i in diffs if i % 66 < 64)}")

    # 4) round-trip: decode the converted file and compare with the folded source values
    a6 = np.frombuffer(actual, dtype=np.uint8).reshape(-1, mod.TQ2_0_BLOCK)
    dq = a6[:, 64:66].copy().view(np.float16).astype(np.float32).reshape(-1)
    u4 = ((a6[:, :64].reshape(a6.shape[0], 64, 1) >> (2 * np.arange(4))) & 3).reshape(a6.shape[0], mod.QK_K)
    dec = ((u4.astype(np.float32) - 1.0) * dq[:, None]).reshape(-1)
    err = np.abs(dec - vals.reshape(-1)).mean()
    rel = err / (np.abs(vals).mean() + 1e-9)
    print(f"  round-trip: mean |error| {err:.5f}, relative {rel:.3%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
