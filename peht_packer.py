#!/usr/bin/env python3
"""
peht_packer.py — lossless TQ2_0 -> TQ1_0 repacker for ternary GGUF models.

GGML's TQ2_0 stores ternary weights (every weight is -d, 0 or +d) at exactly 2 bits.
The true information content of a ternary value is log2(3) = 1.5849... bits, so 2-bit
packing burns ~0.42 bits per weight on nothing. TQ1_0 is GGML's native base-3 container:
five trits per byte (3^5 = 243 <= 256). This tool rewrites every TQ2_0 tensor of a GGUF
into TQ1_0, keeping the fp16 block scales and every weight value bit-for-bit identical.

  * zeros are REAL values (~1/3 of all weights) — their positions are information and
    nothing is dropped; what disappears is only the wasted air between 2 bits and log2(3)
  * 2.0625 -> 1.6875 bits per weight; a 27B model shrinks ~18% (about 1.2 GiB)
  * all GGUF metadata (including prism.hadamard.* keys of folded models) is preserved
  * the output runs on any stock llama.cpp / llama.rn build (native tq1_0 kernels)

F32 / F16 / BF16 tensors (norms, biases) are copied as-is. Any other quant type aborts.

Usage:
    python peht_packer.py input_tq2.gguf output_tq1.gguf [--verify] [--force]

Requires: Python 3.9+, numpy, tqdm       (pip install numpy tqdm)

PEHT — Pobozhii Egor — Hadamard Ternary tools.
"""

import argparse
import os
import struct
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("peht_packer needs numpy:  pip install numpy")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# --------------------------------------------------------------------- gguf codec

GGUF_MAGIC = 0x46554747

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35

QK = 256              # TQ1_0 / TQ2_0 block size (weights)
TQ2_BLOCK = 66        # 64 bytes of 2-bit codes + fp16 scale
TQ1_BLOCK = 54        # 48+4 bytes of base-3 digits + fp16 scale = 1.6875 bpw

FTYPE_TQ1_0 = 36      # LLAMA_FTYPE_MOSTLY_TQ1_0

# scalars copied as-is: type -> bytes per weight
PASSTHROUGH = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2, GGML_TYPE_BF16: 2}

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
            if et not in FIXED:
                raise ValueError(f"unsupported array element type {et}")
            fmt = FIXED[et]
            sz = struct.calcsize(fmt)
            return list(struct.unpack("<" + fmt[1] * n, self.read(sz * n)))
        if t not in FIXED:
            raise ValueError(f"unsupported kv type {t}")
        return struct.unpack(FIXED[t], self.read(struct.calcsize(FIXED[t])))[0]


def write_value(out, t, v):
    if t == 8:
        b = v.encode("utf-8")
        out.write(struct.pack("<Q", len(b)))
        out.write(b)
    elif t == 9:
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
        raise ValueError(f"{path}: not a GGUF file")
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    r = Rdr(f)
    kv = []
    align = 32
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
                raise ValueError(f"unsupported array element type {et} in key {key}")
            v = (et, items)
        else:
            v = r.value(t)
        if key == "general.alignment":
            align = int(v)
        kv.append((key, t, v))

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
        "f": f, "version": version, "kv": kv, "tensors": tensors, "align": align,
        "data_start": (infos_end + align - 1) // align * align,
    }


# ------------------------------------------------------- base-3 <-> 2-bit packing
#
# TQ1_0 layout (ggml-common.h, block_tq1_0, 54 bytes per 256 weights):
#   qs[48]  - 240 weights, five base-3 digits per byte, 3^5 = 243
#   qh[4]   - 16 more weights, four digits per byte, shifted to the top of 3^6
#   d       - fp16 scale, value = (digit - 1) * d
# The encoder rounds the 243-code up into a byte (ceil(q * 256 / 243)), exactly like
# ggml's quantize_row_tq1_0_ref; the decoder walks digits back with modular arithmetic.

_POW3 = (1, 3, 9, 27, 81)


def pack_tq1_0(code, d_raw):
    """code: (B, 256) uint8 digits 0..2, d_raw: (B, 2) raw fp16 -> (B, 54) bytes."""
    B = code.shape[0]
    out = np.empty((B, TQ1_BLOCK), dtype=np.uint8)
    x = code.astype(np.uint32)

    xa = x[:, 0:160].reshape(B, 5, 32)
    ca = xa[:, 0] * 81 + xa[:, 1] * 27 + xa[:, 2] * 9 + xa[:, 3] * 3 + xa[:, 4]
    out[:, 0:32] = ((ca * 256 + 242) // 243).astype(np.uint8)

    xb = x[:, 160:240].reshape(B, 5, 16)
    cb = xb[:, 0] * 81 + xb[:, 1] * 27 + xb[:, 2] * 9 + xb[:, 3] * 3 + xb[:, 4]
    out[:, 32:48] = ((cb * 256 + 242) // 243).astype(np.uint8)

    xh = x[:, 240:256].reshape(B, 4, 4)
    c4 = xh[:, 0] * 27 + xh[:, 1] * 9 + xh[:, 2] * 3 + xh[:, 3]
    out[:, 48:52] = ((c4 * 3 * 256 + 242) // 243).astype(np.uint8)

    out[:, 52:54] = d_raw
    return out


def unpack_tq2_0(a):
    """a: (B, 66) TQ2_0 blocks -> digit array (B, 256) uint8 0..2, d_raw (B, 2).

    TQ2_0 layout: byte j+m holds four 2-bit codes; natural index = group*128 + l*32 + m,
    value = (code - 1) * d.
    """
    d_raw = a[:, 64:66].copy()
    qb = a[:, 0:64].reshape(-1, 2, 32)
    c = np.empty((qb.shape[0], 2, 4, 32), dtype=np.uint8)
    for l in range(4):
        c[:, :, l, :] = (qb >> (2 * l)) & 3
    return c.reshape(-1, QK), d_raw


def _fp16_to_f32(raw):
    return np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=np.float16).astype(np.float32)


def dequant_tq1_mirror(blocks):
    """Python mirror of ggml's dequantize_row_tq1_0 — used by --verify and --selftest."""
    B = blocks.shape[0]
    d = _fp16_to_f32(blocks[:, 52:54])
    y = np.empty((B, QK), dtype=np.float32)

    def dig(byte_arr, n):
        q = (byte_arr.astype(np.uint16) * _POW3[n]) & 0xFF
        return (((q * 3) & 0xFFFF) >> 8).astype(np.float32)

    a = blocks[:, 0:32]
    for n in range(5):
        y[:, n * 32:(n + 1) * 32] = (dig(a, n) - 1.0) * d[:, None]
    b = blocks[:, 32:48]
    for n in range(5):
        y[:, 160 + n * 16:160 + (n + 1) * 16] = (dig(b, n) - 1.0) * d[:, None]
    h = blocks[:, 48:52]
    for n in range(4):
        y[:, 240 + n * 4:240 + (n + 1) * 4] = (dig(h, n) - 1.0) * d[:, None]
    return y


class Progress:
    """tqdm if available, tiny percent fallback otherwise."""

    def __init__(self, total, desc):
        if tqdm is not None:
            self._bar = tqdm(total=total, desc=desc, unit="B", unit_scale=True,
                             unit_divisor=1024, dynamic_ncols=True, mininterval=0.5)
        else:
            self._bar = None
            self._total = max(total, 1)
            self._n = 0
            self._step = -1

    def update(self, n):
        if self._bar is not None:
            self._bar.update(n)
        else:
            self._n += n
            step = int(100 * self._n / self._total)
            if step != self._step and step % 5 == 0:
                self._step = step
                print(f"  ... {step}%", flush=True)

    def touch(self, text):
        if self._bar is not None:
            self._bar.set_postfix_str(text, refresh=False)

    def close(self):
        if self._bar is not None:
            self._bar.close()


# ------------------------------------------------------------------------ selftest

def selftest():
    rng = np.random.default_rng(27)
    B = 100_000
    code = rng.integers(0, 3, size=(B, QK), dtype=np.uint8)
    d_raw = rng.integers(0, 256, size=(B, 2), dtype=np.uint8)
    d_raw[:, 1] &= 0x7B
    blocks = pack_tq1_0(code, d_raw)
    d = _fp16_to_f32(d_raw)
    want = (code.astype(np.float32) - 1.0) * d[:, None]
    got = dequant_tq1_mirror(blocks)
    ok1 = np.array_equal(got.view(np.uint32), want.view(np.uint32))

    c32 = code.reshape(B, 2, 4, 32)
    qb = np.zeros((B, 2, 32), dtype=np.uint8)
    for l in range(4):
        qb |= (c32[:, :, l, :] << (2 * l)).astype(np.uint8)
    a2 = np.concatenate([qb.reshape(B, 64), d_raw], axis=1)
    code2, d2 = unpack_tq2_0(a2)
    ok2 = np.array_equal(code2, code)

    print(f"selftest: {B} blocks — pack/unpack byte-exact: {ok1 and ok2}")
    if not (ok1 and ok2):
        raise SystemExit("SELFTEST FAILED")


# ---------------------------------------------------------------------- repacking

def main():
    ap = argparse.ArgumentParser(
        prog="peht_packer",
        description="Lossless TQ2_0 -> TQ1_0 repack (2.0625 -> 1.6875 bits per ternary weight).",
        epilog="example: python peht_packer.py model-tq2.gguf model-tq1.gguf --verify",
    )
    ap.add_argument("src", nargs="?", help="input GGUF with TQ2_0 tensors")
    ap.add_argument("dst", nargs="?", help="output GGUF (TQ1_0)")
    ap.add_argument("--verify", action="store_true",
                    help="after packing, sample rows and compare values bit-exactly")
    ap.add_argument("--force", "-f", action="store_true", help="overwrite the output file")
    ap.add_argument("--selftest", action="store_true", help="run the packing selftest and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0
    if not args.src or not args.dst:
        ap.error("input and output paths are required")

    if not os.path.isfile(args.src):
        ap.error(f"no such file: {args.src}")
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        ap.error("input and output are the same file")
    if os.path.exists(args.dst) and not args.force:
        ap.error(f"{args.dst} already exists (use --force to overwrite)")

    st = parse(args.src)
    f = st["f"]
    align = st["align"]
    tensors = st["tensors"]
    src_size = os.path.getsize(args.src)

    kv_out = list(st["kv"])
    has_ftype = False
    for i, (key, ttype, v) in enumerate(kv_out):
        if key == "general.file_type":
            kv_out[i] = (key, ttype, FTYPE_TQ1_0)
            has_ftype = True
    if not has_ftype:
        kv_out.append(("general.file_type", 4, FTYPE_TQ1_0))

    pos = 0
    n_conv = n_pass = 0
    total_src = 0
    for t in tensors:
        n = 1
        for dim in t["dims"]:
            n *= dim
        t["n"] = n
        t["convert"] = t["type"] == GGML_TYPE_TQ2_0
        if t["convert"]:
            if n % QK:
                raise SystemExit(f"{t['name']}: {n} weights is not a multiple of {QK}")
            t["src_size"] = n // QK * TQ2_BLOCK
            t["new_size"] = n // QK * TQ1_BLOCK
            n_conv += 1
        elif t["type"] in PASSTHROUGH:
            t["src_size"] = t["new_size"] = n * PASSTHROUGH[t["type"]]
            n_pass += 1
        else:
            raise SystemExit(
                f"{t['name']}: type {t['type']} is not supported "
                f"(expected TQ2_0/F32/F16/BF16)")
        total_src += t["src_size"]
        pos = (pos + align - 1) // align * align
        t["new_offs"] = pos
        pos += t["new_size"]

    if n_conv == 0:
        raise SystemExit("nothing to pack: no TQ2_0 tensors in this GGUF")

    print("peht_packer — lossless TQ2_0 -> TQ1_0 repack (base-3, 5 trits per byte)")
    print(f"  input : {args.src}  ({src_size / 1e9:.3f} GB)")
    print(f"  tensors: {len(tensors)}  (ternary {n_conv}, passthrough {n_pass})")
    print(f"  expected data: {total_src / 1e9:.3f} GB -> "
          f"{sum(t['new_size'] for t in tensors) / 1e9:.3f} GB")

    import io
    kv_buf = io.BytesIO()
    for key, ttype, v in kv_out:
        kb = key.encode("utf-8")
        kv_buf.write(struct.pack("<Q", len(kb)))
        kv_buf.write(kb)
        kv_buf.write(struct.pack("<I", ttype))
        write_value(kv_buf, ttype, v)
    kv_bytes = kv_buf.getvalue()

    infos_size = sum(8 + len(t["name"].encode("utf-8")) + 4 + 8 * len(t["dims"]) + 12
                     for t in tensors)
    data_start = (8 + 16 + len(kv_bytes) + infos_size + align - 1) // align * align

    tmp = args.dst + ".tmp-peht"
    bar = Progress(total_src, "packing")
    try:
        with open(tmp, "wb") as out:
            out.write(struct.pack("<IIQQ", GGUF_MAGIC, st["version"], len(tensors), len(kv_out)))
            out.write(kv_bytes)
            for t in tensors:
                name = t["name"].encode("utf-8")
                out.write(struct.pack("<Q", len(name)))
                out.write(name)
                out.write(struct.pack("<I", len(t["dims"])))
                for dim in t["dims"]:
                    out.write(struct.pack("<Q", dim))
                out.write(struct.pack("<IQ",
                                      GGML_TYPE_TQ1_0 if t["convert"] else t["type"],
                                      t["new_offs"]))
            out.write(b"\x00" * (data_start - out.tell()))

            for t in tensors:
                base = data_start + t["new_offs"]
                bar.touch(t["name"][:32])
                f.seek(st["data_start"] + t["offs"])
                if not t["convert"]:
                    out.seek(base)
                    left = t["new_size"]
                    while left > 0:
                        take = min(left, 64 * 1024 * 1024)
                        out.write(f.read(take))
                        left -= take
                        bar.update(take)
                    continue

                k = t["dims"][0]
                rows = t["n"] // k
                rows_per_chunk = max(1, (64 * 1024 * 1024) // k)
                src_row = k // QK * TQ2_BLOCK
                dst_row = k // QK * TQ1_BLOCK
                r0 = 0
                while r0 < rows:
                    r1 = min(rows, r0 + rows_per_chunk)
                    nr = r1 - r0
                    blob = f.read(nr * src_row)
                    if len(blob) != nr * src_row:
                        raise EOFError(f"{t['name']}: short read")
                    a = np.frombuffer(blob, dtype=np.uint8).reshape(-1, TQ2_BLOCK)
                    code, d_raw = unpack_tq2_0(a)
                    if code.max() > 2:
                        raise SystemExit(
                            f"{t['name']}: contains the fourth state (code 3 = +2d); "
                            f"TQ1_0 is ternary and cannot hold it losslessly")
                    out.seek(base + r0 * dst_row)
                    out.write(pack_tq1_0(code, d_raw).tobytes())
                    r0 = r1
                    bar.update(nr * src_row)
        os.replace(tmp, args.dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    finally:
        bar.close()

    dst_size = os.path.getsize(args.dst)
    saved = (src_size - dst_size) / 1e9
    print()
    print("-" * 62)
    print(f"  output: {args.dst}  ({dst_size / 1e9:.3f} GB)")
    print(f"  size  : {100.0 * dst_size / src_size:.2f}% of the input")
    print()
    print(f"\u2705 Успешно! Сэкономлено {saved:.2f} Гигабайт памяти.")
    print("-" * 62)

    if args.verify:
        verify(args.src, st, args.dst)
    return 0


def verify(src_path, src_st, dst_path):
    """Sample rows of every repacked tensor and compare before/after bit-exactly."""
    st_dst = parse(dst_path)
    ts, td = src_st["tensors"], st_dst["tensors"]
    if len(ts) != len(td):
        raise SystemExit("VERIFY FAILED: tensor count mismatch")
    for a, b in zip(ts, td):
        if a["name"] != b["name"] or a["dims"] != b["dims"]:
            raise SystemExit(f"VERIFY FAILED: {a['name']}")
        want_type = GGML_TYPE_TQ1_0 if a["convert"] else a["type"]
        if b["type"] != want_type:
            raise SystemExit(f"VERIFY FAILED: {a['name']} type {b['type']} != {want_type}")

    rng = np.random.default_rng(27)
    fsrc, fdst = src_st["f"], st_dst["f"]
    rows_checked = 0
    for a, b in zip(ts, td):
        if not a["convert"]:
            continue
        k = a["dims"][0]
        rows = a["n"] // k
        src_row = k // QK * TQ2_BLOCK
        dst_row = k // QK * TQ1_BLOCK
        sample = sorted(set(
            rng.integers(0, rows, size=2).tolist() + [0, rows // 2, rows - 1]))
        for r in sample:
            fsrc.seek(src_st["data_start"] + a["offs"] + r * src_row)
            fdst.seek(st_dst["data_start"] + b["offs"] + r * dst_row)
            a_blob = fsrc.read(src_row)
            b_blob = fdst.read(dst_row)
            code, d_raw = unpack_tq2_0(
                np.frombuffer(a_blob, dtype=np.uint8).reshape(-1, TQ2_BLOCK))
            d = _fp16_to_f32(d_raw)
            want = (code.astype(np.float32) - 1.0) * d[:, None]
            got = dequant_tq1_mirror(
                np.frombuffer(b_blob, dtype=np.uint8).reshape(-1, TQ1_BLOCK))
            if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                raise SystemExit(f"VERIFY FAILED: values differ in {a['name']} row {r}")
            rows_checked += 1
    print(f"\u2705 verified: {len([t for t in ts if t['convert']])} tensors, "
          f"{rows_checked} sampled rows — bit-exact")


if __name__ == "__main__":
    sys.exit(main())
