#!/usr/bin/env python3
# convert_pq2_0_to_tq2_0.py - rewrite Ternary-Bonsai-2 27B from PQ2_0 (type 142, rotated basis)
# to TQ2_0 (type 35, true trits in the PRIMAL basis) so it runs on the stock llama.cpp
# ternary "additions" kernels (the same stack that gives the 8B model ~46 GB/s on ARM).
#
# WHY THIS WORKS (from the Prism fork source, pinned adfffbe):
#   * PQ2_0 block: 128 weights, fp16 scale d (= amax), 2-bit codes; value = (code-1)*d,
#     code order: element j -> byte j/4, bits 2*(j%4), little-endian codes.
#   * The stored weights are rotated: runtime applies to the ACTIVATION (per row, blocks of 1024):
#         x' = R ( s * x )        R = normalized Sylvester-Hadamard_1024 (/32),  s = sign vector
#     (signs first, then FWHT; see build_lora_mm + GGML_HINT_SRC0_IS_HADAMARD in llama-graph.cpp).
#     So the folded weight satisfies  W_f = W S R  (S=diag(s), R symmetric orthogonal),
#     and the primal weights are   W = W_f R S   ->  "FWHT, then signs".
#   * Sign vectors live in prism.hadamard.sign_values, sliced by input width (sign_widths):
#         K=5120 -> [0:5120], K=6144 -> [5120:11264], K=17408 -> [11264:28672].
#   * ssm_out (gdn_v_grouped) additionally permutes the input from tiled [hd,nk,rep] to grouped
#     [hd,rep,nk] before signs+FWHT; inverse therefore undoes the perm AFTER signs:
#         hd = 6144/48 = 128, nk = 16, rep = 48/16 = 3.
#   * token_embd.weight (inverse_weight_names) uses the same inverse formula (build_embd_rows
#     applies FWHT then signs to the looked-up row).
#   * F32/BF16 tensors are not rotated: copied verbatim.
#
# The prism.hadamard.* metadata is STRIPPED from the output: the primal-basis file must not be
# fed to a runtime that folds the rotation again.
#
# Usage: python convert_pq2_0_to_tq2_0.py model-pq2_0.gguf model-tq2_0.gguf [--max-tensors N]

import argparse
import struct
import sys

import numpy as np

GGUF_MAGIC = 0x46554747

GGML_TYPE_F32 = 0
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_PQ2_0 = 142

QK_PQ2_0 = 128
PQ2_0_BLOCK = 34      # fp16 scale + 32 bytes of 2-bit codes
QK_K = 256
TQ2_0_BLOCK = 66      # 64 bytes of trits + fp16 scale
HAD_BLOCK = 1024

FTYPE_TQ2_0 = 37      # LLAMA_FTYPE_MOSTLY_TQ2_0

# ternary scale candidates, as fractions of the block max |w| (from the Q2_0->TQ2_0 converter)
SCALE_FRACS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)

STRIP_PREFIX = "prism.hadamard."


# ------------------------------------------------------------------ gguf reading

FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
         6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Rdr:
    def __init__(self, fh):
        self.fh = fh

    def read(self, n):
        d = self.fh.read(n)
        if len(d) != n:
            raise EOFError("short read")
        return d

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", "replace")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32()
            n = self.u64()
            if et == 8:
                return [self.string() for _ in range(n)]
            if et == 9:
                raise ValueError("nested array")
            if et not in FIXED:
                raise ValueError(f"bad array elem type {et}")
            fmt = FIXED[et]
            sz = struct.calcsize(fmt)
            return list(struct.unpack("<" + fmt[1] * n, self.read(sz * n)))
        if t not in FIXED:
            raise ValueError(f"bad kv type {t}")
        return struct.unpack(FIXED[t], self.read(struct.calcsize(FIXED[t])))[0]


def write_value(out, t, v):
    if t == 8:
        b = v.encode("utf-8")
        out.write(struct.pack("<Q", len(b)))
        out.write(b)
    elif t == 9:
        if not v:
            raise ValueError("empty array")
        et, items = v
        out.write(struct.pack("<I", et))
        out.write(struct.pack("<Q", len(items)))
        for it in items:
            write_value(out, et, it)
    else:
        out.write(struct.pack(FIXED[t], v))


def parse(path):
    f = open(path, "rb")
    magic, version = struct.unpack("<II", f.read(8))
    if magic != GGUF_MAGIC:
        raise ValueError("not a gguf file")
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    r = Rdr(f)
    kv = []                      # (key, type, value)  - value may be (elem_type, list) for arrays
    align = 32
    signs = None                 # np.int32[28672]
    sign_widths = None           # list of 3
    rotated = set()
    inverse_names = set()

    for _ in range(n_kv):
        key = r.string()
        t = r.u32()
        if t == 9:
            et = r.u32()
            n = r.u64()
            if et == 8:
                items = [r.string() for _ in range(n)]
            elif et in FIXED:
                fmt = FIXED[et]
                sz = struct.calcsize(fmt)
                items = list(struct.unpack("<" + fmt[1] * n, r.read(sz * n)))
            else:
                raise ValueError(f"unsupported array elem type {et} for {key}")
            v = (et, items)
        else:
            v = r.value(t)
        kv.append((key, t, v))

        if key == "general.alignment":
            align = int(v)
        elif key == "prism.hadamard.sign_values":
            signs = np.array(v[1], dtype=np.int64)
        elif key == "prism.hadamard.sign_widths":
            sign_widths = [int(x) for x in v[1]]
        elif key == "prism.hadamard.weight_names":
            rotated = set(v[1])
        elif key == "prism.hadamard.inverse_weight_names":
            inverse_names = set(v[1])

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        offs = r.u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "offs": offs})

    infos_end = f.tell()
    return {
        "f": f, "version": version, "n_tensors": n_tensors, "kv": kv,
        "align": align, "signs": signs, "sign_widths": sign_widths,
        "rotated": rotated, "inverse_names": inverse_names,
        "tensors": tensors, "data_start": (infos_end + align - 1) // align * align,
    }


# ------------------------------------------------------------------ transforms

def fwht_blocks(x):
    """x: (rows*nblk, HAD_BLOCK) float32 -> normalized FWHT along last axis (H/sqrt(n))."""
    n = HAD_BLOCK
    a = x.copy()
    length = 1
    while length < n:
        a = a.reshape(a.shape[0], n // (2 * length), 2, length)
        u = a[:, :, 0, :].copy()
        v = a[:, :, 1, :].copy()
        a[:, :, 0, :] = u + v
        a[:, :, 1, :] = u - v
        a = a.reshape(a.shape[0], n)
        length <<= 1
    a *= np.float32(1.0 / np.sqrt(n))
    return a


def sign_slice(signs, widths, k):
    off = 0
    for w in widths:
        w = int(w)
        if w == k:
            return signs[off:off + k]
        off += w
    raise ValueError(f"no sign vector for width {k}")


def inverse_rotate(w, k, signs_vec, is_ssm_out, order="sh"):
    """w: (rows, k) float32 in the rotated basis -> primal basis rows. Returns (rows, k).

    order: "sh" — H then S (current, matches the fork graph), "hs" — S then H
    (experiment alternative), "none" — no un-rotation (control).
    """
    if order == "none":
        x = w
    elif order == "hs":
        # signs BEFORE the transform (latent basis, original element order)
        x = w * signs_vec[None, :].astype(np.float32)
        rows0 = w.shape[0]
        nblk0 = k // HAD_BLOCK
        x = fwht_blocks(x.reshape(rows0 * nblk0, HAD_BLOCK)).reshape(rows0, k)
    else:
        rows0 = w.shape[0]
        nblk0 = k // HAD_BLOCK
        x = fwht_blocks(w.reshape(rows0 * nblk0, HAD_BLOCK)).reshape(rows0, k)
        x = x * signs_vec[None, :].astype(np.float32)   # signs AFTER the transform
    if is_ssm_out:
        # grouped [hd,rep,nk] -> tiled [hd,nk,rep]; hd=128, rep=3, nk=16
        x = x.reshape(x.shape[0], 128, 3, 16).transpose(0, 1, 3, 2).reshape(x.shape[0], k)
    return x


# ------------------------------------------------------------------ ternary pack

def quantize_ternary_256(w):
    """w: (n256, 256) float32 -> (trits int8 in {-1,0,1}, scale (n256,)).

    Same least-squares scale search as the Q2_0->TQ2_0 converter: max-fraction grid, then
    least-squares refinement with trits reassigned at each new scale (converges in 2-3 rounds).
    Rounding matches C lroundf (half away from zero).
    """
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax, 1.0)
    n = w.shape[0]
    best_err = np.full(n, np.inf, dtype=np.float32)
    best_s = scale.copy()
    best_t = np.zeros((n, QK_K), dtype=np.int8)

    def try_scale(s):
        nonlocal best_err, best_s, best_t
        q = np.floor(np.abs(w) / s[:, None] + 0.5)
        t = (np.sign(w) * np.minimum(q, 1.0)).astype(np.int8)
        err = ((w - t * s[:, None]) ** 2).mean(axis=1)
        upd = err < best_err
        best_err = np.where(upd, err, best_err)
        best_s = np.where(upd, s, best_s)
        best_t = np.where(upd[:, None], t, best_t)

    for fr in SCALE_FRACS:
        try_scale(scale * fr)

    for _ in range(3):
        num = (w * best_t.astype(np.float32)).sum(axis=1)
        cnt = (best_t != 0).sum(axis=1).astype(np.float32)
        s = np.where(cnt > 0, np.abs(num) / np.maximum(cnt, 1.0), best_s)
        s = np.where(s > 0, s, best_s)
        try_scale(s)

    return best_t, best_s


def quantize_ternary_256_fast(w):
    """Fast quantizer: d=amax with plain rounding, exactly like the fork reference
    (quantize_row_tq2_0_ref). The only difference from the LS path is the scale choice."""
    amax = np.abs(w).max(axis=1)
    s = np.where(amax > 0, amax, 1.0)
    q = np.floor(np.abs(w) / s[:, None] + 0.5)
    t = (np.sign(w) * np.minimum(q, 1.0)).astype(np.int8)
    return t, s


def pack_tq2_0(trits, scales):
    """trits: (n256, 256) int8 {-1,0,1}; scales (n256,) -> bytes (n256*66)."""
    n = trits.shape[0]
    u = (trits + 1).astype(np.uint8)    # 0,1,2
    qs = np.zeros((n, 64), dtype=np.uint8)
    for q in range(2):
        base = q * 32
        for m in range(4):
            idx = q * 128 + np.arange(32) + m * 32
            qs[:, base:base + 32] |= (u[:, idx] & 3) << (2 * m)
    out = np.zeros((n, TQ2_0_BLOCK), dtype=np.uint8)
    out[:, :64] = qs
    out[:, 64:66] = scales.astype(np.float16).view(np.uint8).reshape(n, 2)
    return out.tobytes()


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-tensors", type=int, default=0, help="debug: convert only first N tensors, rest verbatim")
    ap.add_argument("--undo", choices=("sh", "hs", "none"), default="sh",
                    help="un-rotation order: sh = H then signs (current), hs = signs then H, none = no un-rotation")
    ap.add_argument("--fast", action="store_true",
                    help="fast quantizer (d=amax, like the fork reference) instead of the LS search")
    ap.add_argument("--chunk-weights", type=int, default=64 * 1024 * 1024,
                    help="weights read/written per chunk (reduce for small experiments)")
    args = ap.parse_args()
    print(f"mode: undo={args.undo}, fast={args.fast}", flush=True)

    st = parse(args.src)
    f = st["f"]
    align = st["align"]
    signs = st["signs"]
    widths = st["sign_widths"]
    quantize_fn = quantize_ternary_256_fast if args.fast else quantize_ternary_256
    if signs is None or widths is None:
        raise ValueError("no prism.hadamard.sign_values/sign_widths in the source file")

    tensors = st["tensors"]
    for t in tensors:
        if t["type"] == GGML_TYPE_PQ2_0 and t["name"] not in st["rotated"] and t["name"] not in st["inverse_names"]:
            raise ValueError(f"PQ2_0 tensor not in rotation lists: {t['name']}")

    # plan the output layout
    PASSTHROUGH_SIZES = {GGML_TYPE_F32: 4, GGML_TYPE_BF16: 2}
    pos = 0
    for i, t in enumerate(tensors):
        n = 1
        for d in t["dims"]:
            n *= d
        t["n"] = n
        pos = (pos + align - 1) // align * align
        t["new_offs"] = pos
        convert = (t["type"] == GGML_TYPE_PQ2_0) and (args.max_tensors == 0 or i < args.max_tensors)
        t["convert"] = convert
        if convert:
            if n % QK_K:
                raise ValueError(f"{t['name']}: {n} weights not a multiple of {QK_K}")
            t["new_size"] = n // QK_K * TQ2_0_BLOCK
        elif t["type"] == GGML_TYPE_PQ2_0:
            # debug mode only (--max-tensors): keep the original blocks untouched
            if n % QK_PQ2_0:
                raise ValueError(f"{t['name']}: {n} weights not a multiple of {QK_PQ2_0}")
            t["new_size"] = n // QK_PQ2_0 * PQ2_0_BLOCK
        elif t["type"] in PASSTHROUGH_SIZES:
            t["new_size"] = n * PASSTHROUGH_SIZES[t["type"]]
        else:
            raise ValueError(f"unsupported tensor type {t['type']} for {t['name']}")
        pos += t["new_size"]

    # KV block without prism.hadamard.* (rebuilt); file_type -> TQ2_0
    kv_out = []
    for key, ttype, v in st["kv"]:
        if key.startswith(STRIP_PREFIX):
            continue
        if key == "general.file_type":
            v = FTYPE_TQ2_0
        kv_out.append((key, ttype, v))

    # serialize to measure sizes
    import io
    kv_buf = io.BytesIO()
    for key, ttype, v in kv_out:
        kb = key.encode("utf-8")
        kv_buf.write(struct.pack("<Q", len(kb)))
        kv_buf.write(kb)
        kv_buf.write(struct.pack("<I", ttype))
        write_value(kv_buf, ttype, v)
    kv_bytes = kv_buf.getvalue()

    infos_size = sum(8 + len(t["name"].encode("utf-8")) + 4 + 8 * len(t["dims"]) + 12 for t in tensors)
    prefix = 8 + 16 + len(kv_bytes)               # magic, version, counts, KVs
    out_data_start = (prefix + infos_size + align - 1) // align * align

    # row-chunk sizing: about 64M weights per chunk (256 MB float32)
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
            ttype = GGML_TYPE_TQ2_0 if t["convert"] else t["type"]
            out.write(struct.pack("<IQ", ttype, t["new_offs"]))
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
                print(f"  [{ti + 1}/{len(tensors)}] {t['name']}: copy {t['n']/1e6:.2f}M", flush=True)
                continue

            k = t["dims"][0]
            rows = t["n"] // k
            if k % HAD_BLOCK:
                raise ValueError(f"{t['name']}: ne0={k} not divisible by {HAD_BLOCK}")
            if k % QK_K:
                raise ValueError(f"{t['name']}: ne0={k} not divisible by {QK_K}")
            signs_vec = sign_slice(signs, widths, k)
            is_ssm = ".ssm_out." in t["name"]

            rows_per_chunk = max(1, chunk_weights // k)
            src_row_bytes = k // QK_PQ2_0 * PQ2_0_BLOCK
            dst_row_bytes = k // QK_K * TQ2_0_BLOCK

            r0 = 0
            while r0 < rows:
                r1 = min(rows, r0 + rows_per_chunk)
                nr = r1 - r0
                blob = f.read(nr * src_row_bytes)
                if len(blob) != nr * src_row_bytes:
                    raise EOFError(f"{t['name']}: short read")
                a = np.frombuffer(blob, dtype=np.uint8).reshape(nr * (k // QK_PQ2_0), PQ2_0_BLOCK)
                d = a[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
                nb128 = a.shape[0]
                codes = ((a[:, 2:].reshape(nb128, 32, 1) >> (2 * np.arange(4))) & 3).reshape(nb128, QK_PQ2_0)
                w = ((codes.astype(np.float32) - 1.0) * d[:, None]).reshape(nr, k)

                w = inverse_rotate(w, k, signs_vec, is_ssm, args.undo)

                w256 = w.reshape(-1, QK_K)
                trits, scales = quantize_fn(w256)
                out.seek(base + r0 * dst_row_bytes)
                out.write(pack_tq2_0(trits, scales))
                r0 = r1

            print(f"  [{ti + 1}/{len(tensors)}] {t['name']}: {t['n']/1e6:.2f}M weights converted", flush=True)

    print(f"done: {args.dst}")


if __name__ == "__main__":
    sys.exit(main())
