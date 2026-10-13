# verify_convert.py — self-check of the LEGACY converter on one tensor:
# 1) read source PQ2_0 rows, un-rotate (as the legacy converter does),
# 2) compute the bytes that SHOULD be in the TQ2_0 file (ternary + pack),
# 3) read the REAL bytes from the converted file at the same offset,
# 4) compare: a mismatch means a byte-level bug; a match means the converter does what it
#    intends and the question moves to runtime semantics.
import importlib.util as u
import os
import sys
import numpy as np

# Legacy verifier: checks the earlier "un-rotate to the primal basis" converter.
# Kept for the record; for the current (folded) path use verify_folded.py.
HERE = os.path.dirname(os.path.abspath(__file__))
spec = u.spec_from_file_location("conv", os.path.join(HERE, "convert_pq2_0_to_tq2_0.py"))
mod = u.module_from_spec(spec)
spec.loader.exec_module(mod)

if len(sys.argv) < 4:
    print("usage: python verify_convert.py src-pq2.gguf dst-tq2.gguf TENSOR [NROWS] [ROW_START]")
    print("  (legacy path; for the folded path use verify_folded.py)")
    sys.exit(2)

src = mod.parse(sys.argv[1])          # PQ2_0
dst = mod.parse(sys.argv[2])          # TQ2_0 (our conversion)
name = sys.argv[3]
nrows = int(sys.argv[4]) if len(sys.argv) > 4 else 4

t_src = next(t for t in src["tensors"] if t["name"] == name)
t_dst = next(t for t in dst["tensors"] if t["name"] == name)
k = t_src["dims"][0]
r_off = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # row to start from (chunk-boundary probe)
n_src = 1
for dd in t_src["dims"]:
    n_src *= dd
rows_total = n_src // k
assert t_dst["type"] == mod.GGML_TYPE_TQ2_0, f"dst type is {t_dst['type']}, expected TQ2_0"

signs_vec = mod.sign_slice(src["signs"], src["sign_widths"], k)
is_ssm = ".ssm_out." in name

f = src["f"]
f.seek(src["data_start"] + t_src["offs"] + r_off * (k // mod.QK_PQ2_0 * mod.PQ2_0_BLOCK))
row_bytes_src = k // mod.QK_PQ2_0 * mod.PQ2_0_BLOCK
blob = f.read(nrows * row_bytes_src)
a = np.frombuffer(blob, dtype=np.uint8).reshape(nrows * (k // mod.QK_PQ2_0), mod.PQ2_0_BLOCK)
d = a[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
nb128 = a.shape[0]
codes = ((a[:, 2:].reshape(nb128, 32, 1) >> (2 * np.arange(4))) & 3).reshape(nb128, mod.QK_PQ2_0)
latent = ((codes.astype(np.float32) - 1.0) * d[:, None]).reshape(nrows, k)

primal = mod.inverse_rotate(latent, k, signs_vec, is_ssm)
trits, scales = mod.quantize_ternary_256(primal.reshape(-1, mod.QK_K))
expected = mod.pack_tq2_0(trits, scales)

g = dst["f"]
g.seek(dst["data_start"] + t_dst["offs"] + r_off * (k // mod.QK_K * mod.TQ2_0_BLOCK))
row_bytes_dst = k // mod.QK_K * mod.TQ2_0_BLOCK
actual = g.read(nrows * row_bytes_dst)

same = actual == expected
print(f"tensor {name}: k={k}, rows {nrows}, ssm={is_ssm}")
print(f"  bytes: expected {len(expected)}, read {len(actual)}")
print(f"  BYTE-MATCH: {'YES' if same else 'NO'}")
if not same:
    diffs = [i for i in range(min(len(actual), len(expected))) if actual[i] != expected[i]]
    print(f"  diffs: {len(diffs)} of {len(actual)}")
    print(f"  first 10: {diffs[:10]}")
    print(f"  in scales (idx%66>=64): {sum(1 for i in diffs if i % 66 >= 64)}")
    print(f"  in codes  (idx%66< 64): {sum(1 for i in diffs if i % 66 < 64)}")

# also compare the decoded VALUES of both files
def deq_tq(blob, k, nrows):
    a6 = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 66)
    dq = a6[:, 64:66].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = a6[:, :64]
    vals = np.zeros((a6.shape[0], 256), dtype=np.float32)
    u4 = ((qs.reshape(a6.shape[0], 64, 1) >> (2 * np.arange(4))) & 3).reshape(a6.shape[0], 256)
    vals = (u4.astype(np.float32) - 1.0) * dq[:, None]
    return vals.reshape(nrows, k)

w_file = deq_tq(actual, k, nrows)
err = np.abs(w_file - primal).mean()
rel = err / (np.abs(primal).mean() + 1e-9)
print(f"  decoded file vs primal: mean |error| {err:.5f} (relative {rel:.3%})")
