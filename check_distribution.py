#!/usr/bin/env python3
# check_distribution.py — is a model ternary in the folded (Hadamard-rotated) basis?
#
# Reads a few rows of the requested tensors and prints:
#   * the folded 2-bit code distribution: values are {-d, 0, +d, +2d} for codes {0,1,2,3}.
#     A folded-ternary model shows ~33/33/33 and ~0% of code 3.
#   * stats for the folded weights and for the primal weights (after FWHT + signs):
#     folded = a ternary grid (|v| p50=p75=d), primal = Gaussian (kurtosis ~3).
#
# Usage:
#   python check_distribution.py model-pq2_0.gguf [--tensors NAME ...] [--rows N]
#
# Defaults: blk.0.ffn_down.weight and blk.0.ssm_out.weight, --rows 256.

import argparse
import importlib.util as u
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = u.spec_from_file_location("conv", os.path.join(HERE, "convert_pq2_0_to_tq2_0.py"))
conv = u.module_from_spec(spec)
spec.loader.exec_module(conv)


def read_rows(st, name, nrows):
    t = next(t for t in st["tensors"] if t["name"] == name)
    k = t["dims"][0]
    total = 1
    for d in t["dims"]:
        total *= d
    rows = total // k
    nrows = min(nrows, rows)
    st["f"].seek(st["data_start"] + t["offs"])
    raw = st["f"].read(nrows * (k // conv.QK_PQ2_0) * conv.PQ2_0_BLOCK)
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, conv.PQ2_0_BLOCK)
    d = a[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
    codes = ((a[:, 2:].reshape(-1, 32, 1) >> (2 * np.arange(4))) & 3).reshape(-1, conv.QK_PQ2_0)
    w = ((codes.astype(np.float32) - 1.0) * d[:, None]).reshape(nrows, k)
    return w, k, codes


def stats(label, w):
    a = w.ravel()
    sd = a.std()
    q = np.percentile(np.abs(a), [50, 75, 90, 99])
    kurt = ((a - a.mean()) ** 4).mean() / (sd ** 4 + 1e-12)
    print(f"    {label}: std={sd:.5f} | |v| p50/p75/p90/p99 = "
          f"{q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}/{q[3]:.4f} | kurt={kurt:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tensors", nargs="*", default=["blk.0.ffn_down.weight", "blk.0.ssm_out.weight"])
    ap.add_argument("--rows", type=int, default=256)
    args = ap.parse_args()

    st = conv.parse(args.model)
    print("file:", args.model)
    has_fold = st["signs"] is not None and st["sign_widths"] is not None
    if not has_fold:
        print("note: no prism.hadamard.* keys, showing the stored-code distribution only")

    for name in args.tensors:
        w, k, codes = read_rows(st, name, args.rows)
        cnt = np.bincount(codes.reshape(-1), minlength=4).astype(np.float64)
        pct = cnt / cnt.sum() * 100.0
        print(f"{name}:")
        print(f"    folded codes 0/1/2/3 = {pct[0]:.2f}% / {pct[1]:.2f}% / {pct[2]:.2f}% / {pct[3]:.4f}%")
        stats("folded", w[:8])
        if has_fold:
            sv = conv.sign_slice(st["signs"], st["sign_widths"], k).astype(np.float32)
            primal = conv.inverse_rotate(w[:8], k, sv, ".ssm_out." in name, "sh")
            stats("primal", primal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
