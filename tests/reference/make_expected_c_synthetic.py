"""Make expected_c_vs_cells_synthetic.npz: the C code's perimeter of made-up catchments where the cells are two to
nine columns wide, among them grids west of 180 W (global columns below 0) and catchments that touch the grid's
edges, as the C code's flowmorph_vs_cells_dump gives it (``make flowmorph-vs-cells-dump`` in the C tree).

    python tests/reference/make_expected_c_synthetic.py <path of flowmorph_vs_cells_dump> [<work directory>]

Every catchment is one connected piece of a grid of 60 x 360 pixels of 1/1200 degree (its pixels rank 0, the rest -1),
its target the first of its pixels; the file holds the masks (packed bits), the transforms as the C code takes them
and the C's perimeter in metres (float64).
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir))
from flowmorph.grid import snap_transform  # noqa: E402

NROW, NCOL = 60, 360


def catchments(rng):
    """(mask, latitude of the grid's middle, west) of every case."""
    cases = []
    for latitude in (62.0, 68.0, 71.0, 75.0, 80.0, 83.5):
        for _ in range(3):
            cases.append((latitude, float(rng.uniform(-170.0, 170.0)), True))
    for latitude in (72.0, 78.0):                         # west of 180 W: global columns below 0
        cases.append((latitude, -180.4, True))
    for latitude in (66.0, 76.0, 81.0):                   # touching the grid's edges
        cases.append((latitude, float(rng.uniform(-170.0, 170.0)), False))
    out = []
    for latitude, west, off_the_edges in cases:
        field = ndimage.gaussian_filter(rng.standard_normal((NROW, NCOL)), (2.5, 9.0))
        mask = field > (0.0 if off_the_edges else -0.4)
        if off_the_edges:
            mask[0, :] = mask[-1, :] = False
            mask[:, 0] = mask[:, -1] = False
        labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        out.append((labels == 1 + int(np.argmax(sizes)), latitude, west))
    return out


def main():
    tool = sys.argv[1]
    work = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp()
    rng = np.random.default_rng(20261018)
    masks, transforms, perimeters = [], [], []
    for mask, latitude, west in catchments(rng):
        transform = snap_transform((west, 1.0 / 1200.0, 0.0, latitude + NROW / 2400.0, 0.0, -1.0 / 1200.0))
        seq = np.where(mask, 0, -1).astype(np.int32).reshape(-1)
        paths = [os.path.join(work, name) for name in ("grid.txt", "seq.int32", "targets.int64", "out.float64")]
        with open(paths[0], "w") as handle:
            handle.write(f"{NROW} {NCOL} " + " ".join(repr(float(value)) for value in transform) + "\n")
        seq.tofile(paths[1])
        np.array([[0, 0, int(np.flatnonzero(seq >= 0)[0])]], dtype=np.int64).tofile(paths[2])
        subprocess.run([tool] + paths, check=True, stdout=subprocess.DEVNULL)
        perimeters.append(float(np.fromfile(paths[3], dtype=np.float64)[0]))
        masks.append(np.packbits(mask, axis=1))
        transforms.append(transform)
    np.savez_compressed(os.path.join(HERE, "expected_c_vs_cells_synthetic.npz"), masks=np.array(masks),
                        transforms=np.array(transforms), perimeter_m=np.array(perimeters), ncol=NCOL)
    print(f"expected_c_vs_cells_synthetic.npz: {len(perimeters)} catchments")


if __name__ == "__main__":
    main()
