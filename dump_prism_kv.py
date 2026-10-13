#!/usr/bin/env python3
"""dump_prism_kv.py — print the GGUF metadata of a model (all keys, including prism.*).
Handy to check that a folded-ternary model still carries its rotation keys.

Usage: python dump_prism_kv.py model.gguf
"""

import importlib.util as u
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = u.spec_from_file_location("conv", os.path.join(HERE, "convert_pq2_0_to_tq2_0.py"))
mod = u.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    st = mod.parse(sys.argv[1])
    print("file:", sys.argv[1])
    print("tensors:", st["n_tensors"], "| alignment:", st["align"])
    for key, t, v in st["kv"]:
        if isinstance(v, tuple):
            et, items = v
            shown = items[:10]
            print(f"  {key} [array {len(items)}] {shown}{' ...' if len(items) > 10 else ''}")
        elif isinstance(v, str) and len(v) > 90:
            print(f"  {key} = <string of {len(v)} chars>")
        else:
            print(f"  {key} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
