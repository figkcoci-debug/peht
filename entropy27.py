#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# entropy27.py — сколько РЕАЛЬНО информации в тернарном потоке (TQ2_0, свёрнутый базис)?
#
# Зачем. Формат TQ2_0 хранит КАЖДЫЙ трит ровно 2 битами. Если распределение кодов не
# равномерно, часть этих бит — «воздух», и поток можно упаковать энтропийным кодером
# БЕЗ ПОТЕРИ КАЧЕСТВА (значения те же, размер меньше). Этот скрипт меряет на живом файле:
#   * доли кодов 0/1/2/3 — суммарно и по классам тензоров;
#   * маргинальную энтропию H (бит/трит) и оценку размера после идеального кодера;
#   * условную энтропию H(следующий|текущий) на сэмпле — есть ли скрытая структура
#     (у белого шума условная равна маргинальной, и надеяться не на что);
#   * состав файла: триты / скейлы / прочие (F32/BF16) тензоры.
#
# Usage: python entropy27.py model.gguf [--sample 32M] [--chunk-mb 64]
import argparse
import importlib.util as u
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = u.spec_from_file_location("conv", os.path.join(HERE, "convert_pq2_0_to_tq2_0.py"))
conv = u.module_from_spec(spec)
spec.loader.exec_module(conv)

QK = 256
TRIT_BYTES = 64     # 256 тритов по 2 бита
BLOCK = 66          # 64 байта тритов + 2 байта fp16-скейла
F32, BF16, TQ2_0 = 0, 30, 35

def H_of(counts):
    p = counts.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def class_of(name):
    for key in ("ffn_down", "ffn_gate", "ffn_up", "attn_qkv", "attn_q", "attn_k",
                "attn_v", "ssm_out", "ssm_in", "ssm", "token_embd", "output"):
        if key in name:
            return key
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sample", type=int, default=32_000_000,
                    help="сколько кодов взять для условной энтропии")
    ap.add_argument("--chunk-mb", type=int, default=64)
    args = ap.parse_args()

    st = conv.parse(args.model)
    f = st["f"]
    data_start = st["data_start"]
    chunk_blocks = args.chunk_mb * 1024 * 1024 // BLOCK

    total = np.zeros(4, dtype=np.int64)
    per_class = {}
    rows = []
    total_blocks = 0
    other_bytes = 0

    sample_parts = []
    sample_have = 0

    for t in st["tensors"]:
        n = 1
        for d in t["dims"]:
            n *= d
        if t["type"] != TQ2_0:
            other_bytes += n * (4 if t["type"] == F32 else 2)
            continue
        nb = n // QK
        p64 = None  # маргинал по носителю держим здесь, чтобы не хранить весь код
        f.seek(data_start + t["offs"])
        left = nb
        while left > 0:
            take = min(left, chunk_blocks)
            blob = f.read(take * BLOCK)
            if len(blob) != take * BLOCK:
                raise EOFError(t["name"])
            a = np.frombuffer(blob, dtype=np.uint8).reshape(take, BLOCK)
            payload = a[:, :64]
            tc = np.zeros(4, dtype=np.int64)
            for sh in (0, 2, 4, 6):
                tc += np.bincount(((payload >> sh) & 3).reshape(-1), minlength=4)
            if p64 is None:
                p64 = tc.copy()
            else:
                p64 += tc
            if sample_have < args.sample:
                take_codes = min(take * QK, args.sample - sample_have)
                codes = np.stack([(payload & 3), ((payload >> 2) & 3),
                                  ((payload >> 4) & 3), ((payload >> 6) & 3)], axis=1)
                sample_parts.append(codes.reshape(-1)[:take_codes].copy())
                sample_have += take_codes
            left -= take
        total += p64
        total_blocks += nb
        cls = class_of(t["name"])
        c = per_class.setdefault(cls, {"counts": np.zeros(4, dtype=np.int64), "weights": 0})
        c["counts"] += p64
        c["weights"] += n
        rows.append((t["name"], n, H_of(p64), p64))
        print(f"  [{t['name']}] {n/1e6:.2f}M  H={H_of(p64):.4f}  "
              f"0/1/2/3 = " + "/".join(f"{100*x/p64.sum():.2f}%" for x in p64), flush=True)

    # ---------- итоги ----------
    trit_bytes = total_blocks * TRIT_BYTES
    scale_bytes = total_blocks * 2
    file_size = os.path.getsize(args.model)
    Hg = H_of(total)
    codes_total = total.sum()

    ideal_trit_bytes = codes_total * Hg / 8.0
    save_bytes = trit_bytes - ideal_trit_bytes

    print()
    print("=" * 72)
    print(f"тритов:            {codes_total/1e9:.3f} млрд ({total_blocks} блоков)")
    print(f"доли кодов 0/1/2/3: " + " / ".join(f"{100*x/codes_total:.4f}%" for x in total))
    print(f"энтропия H:        {Hg:.4f} бит/трит  (сейчас тратится ровно 2.0000)")
    print()
    print(f"состав файла:      триты  {trit_bytes/1e9:.3f} ГБ")
    print(f"                   скейлы {scale_bytes/1e9:.3f} ГБ")
    print(f"                   прочее {other_bytes/1e9:.3f} ГБ")
    print(f"                   всего  {file_size/1e9:.3f} ГБ")
    print()
    print(f"идеальный кодер:   триты {ideal_trit_bytes/1e9:.3f} ГБ  →  экономия {save_bytes/1e9:.3f} ГБ")
    print(f"проекция файла:    {file_size/1e9:.3f} ГБ  →  {(file_size - save_bytes)/1e9:.3f} ГБ "
          f"({100*save_bytes/file_size:.1f}% меньше)")
    print()
    print("по классам тензоров:")
    for cls, c in sorted(per_class.items(), key=lambda kv: -kv[1]["weights"]):
        h = H_of(c["counts"])
        w = c["weights"]
        print(f"  {cls:<10} весов {w/1e9:.3f} млрд  H={h:.4f}  "
              f"0/1/2/3 = " + "/".join(f"{100*x/c['counts'].sum():.2f}%" for x in c["counts"]))

    if sample_parts:
        codes = np.concatenate(sample_parts)
        cur = codes[:-1].astype(np.int64)
        nxt = codes[1:].astype(np.int64)
        joint = np.bincount(cur * 4 + nxt, minlength=16).reshape(4, 4)
        H_joint = H_of(joint.reshape(-1))
        H_next = H_joint - Hg
        print()
        print(f"условная энтропия на сэмпле {len(codes)/1e6:.1f}M кодов:")
        print(f"  H(next|cur) = {H_next:.4f} бит   (у белого шума = H = {Hg:.4f}: "
              f"{'структуры нет' if abs(H_next - Hg) < 0.01 else 'есть структура!'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
