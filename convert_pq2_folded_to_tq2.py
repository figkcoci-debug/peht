#!/usr/bin/env python3
# convert_pq2_folded_to_tq2.py — requantize WITHOUT un-rotating:
# folded ternary PQ2_0 weights ({-d, 0, +d, +2d}) -> TQ2_0 ({-s, 0, +s}) in the SAME (folded)
# basis, KEEPING the prism.hadamard.* keys. The runtime un-rotates activations on the fly.
#
# Why: a folded-ternary model lives in the rotated basis (the primal is Gaussian and 2 bits do
# not survive it: measured perplexity > 10^6). The folded weights are a pure ternary grid:
# ~33% zeros, |v| p99 = 1.2d. The re-encode is essentially exact (per-256 scale = amax).
#
# Usage: python convert_pq2_folded_to_tq2.py src-pq2.gguf dst-tq2.gguf

import argparse
import os
import struct
import sys

import numpy as np

import importlib.util as u
HERE = os.path.dirname(os.path.abspath(__file__))
CONV = os.path.join(HERE, "convert_pq2_0_to_tq2_0.py")
spec = u.spec_from_file_location("conv", CONV)
conv = u.module_from_spec(spec)
spec.loader.exec_module(conv)

GGUF_MAGIC = 0x46554747
PQ2_0 = 142
TQ2_0 = 35
QK_PQ2_0 = 128
QK_K = 256
PQ2_0_BLOCK = 34
TQ2_0_BLOCK = 66
FTYPE_TQ2_0 = 37


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--chunk-weights", type=int, default=64 * 1024 * 1024,
                    help="weights to read/write per chunk")
    args = ap.parse_args()

    st = conv.parse(args.src)
    f = st["f"]
    align = st["align"]
    if st["signs"] is None:
        raise ValueError("no prism.hadamard.sign_values in the source file")

    tensors = st["tensors"]
    kv_out = list(st["kv"])  # keys are kept as-is: prism.hadamard.* must survive!
    for i, (key, ttype, v) in enumerate(kv_out):
        if key == "general.file_type":
            kv_out[i] = (key, ttype, FTYPE_TQ2_0)

    PASSTHROUGH = {0: 4, 30: 2}  # F32, BF16
    pos = 0
    for t in tensors:
        n = 1
        for d in t["dims"]:
            n *= d
        t["n"] = n
        t["convert"] = t["type"] == PQ2_0
        if t["convert"]:
            if n % QK_K:
                raise ValueError(f"{t['name']}: {n} not divisible by {QK_K}")
            t["new_size"] = n // QK_K * TQ2_0_BLOCK
        elif t["type"] in PASSTHROUGH:
            t["new_size"] = n * PASSTHROUGH[t["type"]]
        else:
            raise ValueError(f"unsupported tensor type {t['type']} for {t['name']}")
        pos = (pos + align - 1) // align * align
        t["new_offs"] = pos
        pos += t["new_size"]

    import io
    kv_buf = io.BytesIO()
    for key, ttype, v in kv_out:
        kb = key.encode("utf-8")
        kv_buf.write(struct.pack("<Q", len(kb)))
        kv_buf.write(kb)
        kv_buf.write(struct.pack("<I", ttype))
        conv.write_value(kv_buf, ttype, v)
    kv_bytes = kv_buf.getvalue()

    infos_size = sum(8 + len(t["name"].encode("utf-8")) + 4 + 8 * len(t["dims"]) + 12 for t in tensors)
    out_data_start = (8 + 16 + len(kv_bytes) + infos_size + align - 1) // align * align

    chunk_weights = args.chunk_weights
    with open(args.dst, "wb") as out:
        out.write(struct.pack("<IIQQ", GGUF_MAGIC, st["version"], len(tensors), len(kv_out)))
        out.write(kv_bytes)
        for t in tensors:
            name = t["name"].encode("utf-8")
            out.write(struct.pack("<Q", len(name)))
            out.write(name)
            out.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<IQ", TQ2_0 if t["convert"] else t["type"], t["new_offs"]))
        here = out.tell()
        out.write(b"\x00" * (out_data_start - here))

        for ti, t in enumerate(tensors):
            base = out_data_start + t["new_offs"]
            f.seek(st["data_start"] + t["offs"])
            if not t["convert"]:
                left = t["new_size"]
                while left > 0:
                    take = min(left, 64 * 1024 * 1024)
                    out.write(f.read(take))
                    left -= take
                print(f"  [{ti+1}/{len(tensors)}] {t['name']}: copied", flush=True)
                continue

            k = t["dims"][0]
            rows = t["n"] // k
            rows_per_chunk = max(1, chunk_weights // k)
            src_row = k // QK_PQ2_0 * PQ2_0_BLOCK
            dst_row = k // QK_K * TQ2_0_BLOCK
            r0 = 0
            while r0 < rows:
                r1 = min(rows, r0 + rows_per_chunk)
                nr = r1 - r0
                blob = f.read(nr * src_row)
                if len(blob) != nr * src_row:
                    raise EOFError(f"{t['name']}: short read")
                a = np.frombuffer(blob, dtype=np.uint8).reshape(-1, PQ2_0_BLOCK)
                d = a[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
                codes = ((a[:, 2:].reshape(-1, 32, 1) >> (2 * np.arange(4))) & 3).reshape(-1, QK_PQ2_0)
                vals = ((codes.astype(np.float32) - 1.0) * d[:, None]).reshape(-1, QK_K)
                # re-encode: per-256 scale = amax, trits = clip(round(v/s), -1..1)
                amax = np.abs(vals).max(axis=1)
                s = np.where(amax > 0, amax, 1.0).astype(np.float32)
                trits = np.clip(np.floor(np.abs(vals) / s[:, None] + 0.5) * np.sign(vals), -1, 1).astype(np.int8)
                out.seek(base + r0 * dst_row)
                out.write(conv.pack_tq2_0(trits, s))
                r0 = r1
            print(f"  [{ti+1}/{len(tensors)}] {t['name']}: {t['n']/1e6:.2f}M requantized (folded)", flush=True)

    print(f"done: {args.dst}")


if __name__ == "__main__":
    sys.exit(main())
