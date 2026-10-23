"""Make expected_c_perimeter.npz: the C code's perimeter (step 3, attribute [7]) of every target of the bundled basin.

The C code of FlowMorph 1.1 has a small program for this, flowmorph_vs_cells_dump (``make flowmorph-vs-cells-dump``
in the C tree): it reads a grid, the rank of every pixel and the rank range of every catchment, and writes the
perimeter of each in metres as its step 3 takes it.  This script gives it the bundled basin as this package reads
it (the ranks, the targets, the grid as the C code takes it) and writes the C's metres of every target as doubles
(``perimeter_m``, ``target_cell``) and the layer the package writes (``perimeter_km``: float32 km at the targets,
-9999 elsewhere).

    python tests/reference/make_expected_c_perimeter.py <path of flowmorph_vs_cells_dump> [<work directory>]
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))
import flowmorph  # noqa: E402

DATA = os.path.join(HERE, os.pardir, os.pardir, "data")


def main():
    tool = sys.argv[1]
    work = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp()
    fm = flowmorph.FlowMorph.from_raster(os.path.join(DATA, "dir_example.tif"), os.path.join(DATA, "upa_example.tif"),
                                         os.path.join(DATA, "elv_example.tif"))
    nrow, ncol = fm.shape
    seq = np.ascontiguousarray(fm.seq_dfs, dtype=np.int32).reshape(-1)
    upg = np.asarray(fm.upg).reshape(-1)
    targets = np.flatnonzero(np.asarray(fm.is_target).reshape(-1) & (seq >= 0))
    last = seq[targets].astype(np.int64)
    first = last - upg[targets].astype(np.int64) + 1
    paths = [os.path.join(work, name) for name in ("grid.txt", "seq.int32", "targets.int64", "perimeter_m.float64")]
    with open(paths[0], "w") as handle:
        handle.write(f"{nrow} {ncol} " + " ".join(repr(float(value)) for value in fm.transform) + "\n")
    seq.tofile(paths[1])
    np.stack([first, last, targets.astype(np.int64)], axis=1).astype(np.int64).tofile(paths[2])
    subprocess.run([tool] + paths, check=True)
    perimeter_m = np.fromfile(paths[3], dtype=np.float64)
    layer = np.full(nrow * ncol, np.float32(-9999.0), dtype=np.float32)
    layer[targets] = np.where(perimeter_m > 0.0, (perimeter_m / 1000.0).astype(np.float32), np.float32(-9999.0))
    np.savez_compressed(os.path.join(HERE, "expected_c_perimeter.npz"), perimeter_km=layer.reshape(nrow, ncol),
                        perimeter_m=perimeter_m, target_cell=targets.astype(np.int64))
    print(f"expected_c_perimeter.npz: {targets.size} targets")


if __name__ == "__main__":
    main()
