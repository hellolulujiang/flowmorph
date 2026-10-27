"""The plan form on made-up catchments: the perimeter (flowmorph.vscells) and the length, width and hull
(flowmorph.sphere) against values known beforehand, on small grids made up for the purpose."""

import math
from fractions import Fraction

import numpy as np
import pytest

from flowmorph import earth, sphere, vscells

WGS84 = earth.WGS84
PIXEL = 1.0 / 1200.0


def _grid(mask, north, west=10.0, pixel=PIXEL):
    """A catchment given as a mask: the rank array the kernels read (0 inside, -1 outside), the boundary pixels and
    the grid's tables."""
    nrow, ncol = mask.shape
    seq = np.where(mask, 0, -1).astype(np.int32).reshape(-1)
    padded = np.pad(mask, 1)
    rows, cols = np.nonzero(mask)
    on_boundary = np.zeros(rows.size, bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                on_boundary |= ~padded[rows + 1 + dr, cols + 1 + dc]
    cells = (rows[on_boundary] * ncol + cols[on_boundary]).astype(np.int64)
    tables = sphere.grid_tables(west, pixel, north, -pixel, nrow, ncol)
    return seq, cells, tables


def _perimeter_m(mask, north, west=10.0, pixel=PIXEL, edge=None):
    """P (m) of a catchment as step 3 takes it: from its boundary pixels (the topmost-leftmost, the last row, ten
    steps a boundary pixel and a hundred), 0 for fewer than three.  ``pixel`` places the rows (the transform's term),
    ``edge`` gives the lengths of a pixel's edges (1/1200 exactly on MERIT Hydro's grid; ``pixel`` if not given)."""
    nrow, ncol = mask.shape
    edge = pixel if edge is None else edge
    seq, cells, _ = _grid(mask, north, west, pixel)
    if cells.size < 3:
        return 0.0
    start, last = int(cells.min()), int(cells.max())
    status, value = vscells.perimeter_m(seq, nrow, ncol, 0, 0, north, -pixel, vscells.global_col_of_col0(west, edge),
                                        edge, edge, start // ncol, last // ncol, start // ncol, start % ncol,
                                        10 * cells.size + 100, WGS84)
    assert status == vscells.VS_OK
    return value


def _centre(row, col, north, west=10.0, pixel=PIXEL):
    return north - (row + 0.5) * pixel, west + (col + 0.5) * pixel


# ---- the perimeter: a reference written apart from the package ----

def _reference_on_cells_m(mask, north, west, k, pixel=PIXEL):
    """The weighted trace on cells of k columns, written as plainly as it reads: the columns merged k at a time from
    the global columns that are multiples of k (a cell is in when one of its pixels is), the Moore trace of the merged
    grid from its topmost-leftmost cell with Jacob's stopping rule, the steps weighted 0.980, 1.406 / 1.41421356 and
    -0.091 a corner, a cell's edges k dx by dy at the latitude halfway along the step.  0 for fewer than three steps."""
    nrow, ncol = mask.shape
    col0 = int(math.floor((west + 180.0) / pixel + 0.25))      # whole pixels from 180 W to the west edge, a quarter added
    rows, cols = np.nonzero(mask)
    cell_of_col = (cols + col0) // k
    first_cell = int(cell_of_col.min())
    width = int(cell_of_col.max()) - first_cell + 1
    merged = np.zeros((nrow, width), bool)
    merged[rows, cell_of_col - first_cell] = True
    inside = lambda r, c: 0 <= r < nrow and 0 <= c < width and merged[r, c]     # noqa: E731
    dr = (-1, -1, 0, 1, 1, 1, 0, -1)
    dc = (0, 1, 1, 1, 0, -1, -1, -1)
    start_r = int(np.nonzero(merged.any(axis=1))[0][0])
    start_c = int(np.nonzero(merged[start_r])[0][0])
    back = next(d for d in range(8) if not inside(start_r + dr[d], start_c + dc[d]))
    r, c, steps, total, previous, first, first_step, closed = start_r, start_c, 0, 0.0, -1, -1, None, False

    def edges(latitude):
        m_lat, m_lon = earth.metres_per_degree(latitude, WGS84)
        return m_lat * pixel, float(k) * (m_lon * pixel)

    while steps < 100000:
        moved = False
        for turn in range(1, 9):
            code = (back + turn) % 8
            nr, nc = r + dr[code], c + dc[code]
            if not inside(nr, nc):
                continue
            if steps > 0 and (r, c) == (start_r, start_c) and (nr, nc) == first_step:
                closed = True
                break
            if steps == 0:
                first_step = (nr, nc)
            ns, ew = edges(north + ((float(r) + float(nr)) / 2.0 + 0.5) * -pixel)
            if code % 2 == 0:
                total += 0.980 * (ns if code in (0, 4) else ew)
            else:
                total += (1.406 / 1.41421356) * math.sqrt(ns * ns + ew * ew)
            if previous >= 0 and code % 2 != previous % 2:
                total -= 0.091 * 0.5 * (ns + ew)
            if first < 0:
                first = code
            previous, r, c, back, moved, steps = code, nr, nc, (code + 4) % 8, True, steps + 1
            break
        if not moved:
            break
    if not closed or steps < 3:
        return 0.0
    if previous % 2 != first % 2:
        ns, ew = edges(north + (float(start_r) + 0.5) * -pixel)
        total -= 0.091 * 0.5 * (ns + ew)
    return total


def _ragged(rng, nrow, ncol):
    """A ragged catchment: smoothed noise above 0, its largest 8-connected piece, off the frame."""
    from scipy import ndimage
    field = ndimage.gaussian_filter(rng.standard_normal((nrow, ncol)), (2.0, 6.0))
    mask = field > 0.0
    mask[0, :] = mask[-1, :] = False
    mask[:, 0] = mask[:, -1] = False
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == 1 + int(np.argmax(sizes))


def _cells_m(mask, north, west, k, pixel=PIXEL):
    nrow, ncol = mask.shape
    seq, cells, _ = _grid(mask, north, west, pixel)
    start = int(cells.min())
    return vscells.cells_length_m(seq, nrow, ncol, 0, 0, north, -pixel, vscells.global_col_of_col0(west, pixel), pixel,
                                  pixel, start // ncol, start % ncol, k, 10 * cells.size + 100, WGS84)


@pytest.mark.parametrize("north", (0.5, 30.0, 62.0, 75.0, -48.0))
def test_the_cells_are_traced_as_the_reference_traces_them(north):
    rng = np.random.default_rng(int(abs(north) * 10))
    for _ in range(4):
        mask = _ragged(rng, 40, 160)
        west = float(rng.uniform(-170.0, 170.0))
        for k in (1, 2, 3, 5):
            status, value = _cells_m(mask, north, west, k)
            assert status == vscells.VS_OK
            assert value == _reference_on_cells_m(mask, north, west, k), (north, k)


@pytest.mark.parametrize("north", (3.0, 30.0, 49.0, 62.0, 75.0, 81.0))
def test_the_perimeter_is_taken_between_two_merges(north):
    # (1 - w) P(k0) + w P(k0 + 1), k0 and w at the middle of the catchment's rows; where a pixel is wider than
    # tall (within 6.59 degrees of the equator) P is the weighted trace on the pixels themselves
    rng = np.random.default_rng(7)
    mask = _ragged(rng, 50, 300)
    west = 25.0
    rows = np.nonzero(mask.any(axis=1))[0]
    middle = north + ((float(rows[0]) + float(rows[-1]) + 1.0) / 2.0) * -PIXEL
    k0, w = vscells.merge_at_latitude(middle, PIXEL, PIXEL, WGS84)
    ns, ew = vscells.pixel_edges_m(middle, PIXEL, PIXEL, WGS84)
    assert k0 == max(1, math.floor(ns / ew)) and w == min(max(ns / ew - k0, 0.0), 1.0)
    p0 = _reference_on_cells_m(mask, north, west, k0)
    want = p0 if w == 0.0 else (1.0 - w) * p0 + w * _reference_on_cells_m(mask, north, west, k0 + 1)
    assert _perimeter_m(mask, north, west) == want
    if north == 3.0:
        assert (k0, w) == (1, 0.0)


@pytest.mark.parametrize("north", (0.5, -4.0))
def test_a_rectangle_of_pixels_has_the_weighted_length_of_its_trace(north):
    # near the equator, where P is the trace on the pixels: it goes round the centres, n - 1 steps a side, and a
    # rectangle has no corner between a step along a row or column and a diagonal one
    mask = np.zeros((12, 15), bool)
    mask[2:9, 3:13] = True                                  # 7 rows by 10 columns
    expected = 0.0
    for row in (2, 8):                                      # the top and the bottom rows: 9 steps each
        _, ew = vscells.pixel_edges_m(north + (row + 0.5) * -PIXEL, PIXEL, PIXEL, WGS84)
        expected += 9 * 0.980 * ew
    for row in range(2, 8):                                 # down the right column and up the left one
        ns, _ = vscells.pixel_edges_m(north + (row + 1.0) * -PIXEL, PIXEL, PIXEL, WGS84)
        expected += 2 * 0.980 * ns
    assert _perimeter_m(mask, north) == pytest.approx(expected, rel=1e-13)


def test_one_or_two_pixels_have_no_perimeter():
    mask = np.zeros((6, 6), bool)
    mask[2, 2] = True
    assert _cells_m(mask, 10.0, 10.0, 1) == (vscells.VS_OK, 0.0)
    mask[3, 3] = True                                       # two pixels touching at a corner: one catchment
    assert _cells_m(mask, 10.0, 10.0, 1) == (vscells.VS_OK, 0.0)
    assert _perimeter_m(mask, 10.0) == 0.0


def test_a_hole_is_not_part_of_the_perimeter():
    full = np.zeros((30, 30), bool)
    full[4:26, 5:27] = True
    holed = full.copy()
    holed[10:16, 11:19] = False
    assert _perimeter_m(holed, 50.0) == _perimeter_m(full, 50.0)


def _ellipse(rng, north, cells_wide):
    ns, ew = vscells.pixel_edges_m(north, PIXEL, PIXEL, WGS84)
    a = cells_wide * ns
    b = a * rng.uniform(0.4, 1.0)
    angle = rng.uniform(0.0, math.pi)
    nrow, ncol = int(2.2 * a / ns) + 8, int(2.2 * a / ew) + 8
    y = (nrow / 2.0 - np.arange(nrow)[:, None]) * ns + rng.uniform(0.0, ns)
    x = (np.arange(ncol)[None, :] - ncol / 2.0) * ew + rng.uniform(0.0, ew)
    u, v = x * math.cos(angle) + y * math.sin(angle), -x * math.sin(angle) + y * math.cos(angle)
    h = ((a - b) / (a + b)) ** 2
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0, math.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def test_known_ellipses_read_alike_at_every_latitude():
    # on the pixels the weighted trace reads some 10 % long at 80 degrees; on cells close to square it reads
    # as at the equator, at every latitude
    rng = np.random.default_rng(2026)
    medians = []
    for north in (0.0, 45.0, 60.0, 75.0, 80.0):
        ratios = []
        for _ in range(12):
            mask, true_m = _ellipse(rng, north, rng.uniform(30.0, 60.0))
            ratios.append(_perimeter_m(mask, north + mask.shape[0] / 2.0 * PIXEL) / true_m)
        medians.append(float(np.median(ratios)))
    assert all(0.975 < value < 1.008 for value in medians), medians
    assert max(medians) - min(medians) < 0.02, medians


def test_no_step_where_k_changes():
    rng = np.random.default_rng(11)
    mask = _ragged(rng, 40, 300)
    for crossing in (60.0554, 70.5439):                    # dy / dx = 2 and 3 on WGS84
        values = [_perimeter_m(mask, crossing + 0.002 * i) for i in range(-20, 21)]
        steps = np.abs(np.diff(values)) / np.array(values[:-1])
        assert float(steps.max()) < 1e-3, crossing


def test_the_cells_start_at_the_global_columns_that_are_multiples_of_k():
    rng = np.random.default_rng(5)
    mask = _ragged(rng, 40, 300)
    north = 72.0                                            # k0 = 3: cells of 3 and of 4 columns
    base = _perimeter_m(mask, north, west=30.0)
    assert _perimeter_m(mask, north, west=30.0 + 12 * PIXEL) == base
    shifted = _perimeter_m(mask, north, west=30.0 + PIXEL)
    assert shifted != base and shifted == pytest.approx(base, rel=0.05)
    # the shape moved one column east in its array, on the grid that starts one column west of the shifted one:
    # the same global columns as the shifted grid, the same perimeter
    east = np.zeros((mask.shape[0], mask.shape[1] + 1), bool)
    east[:, 1:] = mask
    assert _perimeter_m(east, north, west=30.0) == shifted
    # a grid west of 180 W (global columns below 0) and the same grid 1200 columns (100 k0 (k0 + 1)) east
    assert vscells.global_col_of_col0(-180.4, PIXEL) < 0
    assert _perimeter_m(mask, north, west=-180.4) == _perimeter_m(mask, north, west=-179.4)


def test_a_merit_hydro_window_keeps_its_global_columns_wherever_it_starts():
    """MERIT Hydro's column c, whose west edge lies half a pixel west of c pixels from 180 W, is global column c - 1
    in every window, and a column of a grid whose edges lie on 180 W is itself; so one catchment keeps its cells, and
    its perimeter, in MERIT windows that start at other columns.  (Rounded to the nearest whole number instead, about
    half of such windows would take column c.)"""
    from flowmorph.grid import north_up

    def west_as_read(west):                                 # the west edge as the package takes a header
        return north_up((west, PIXEL, 0.0, 72.0, 0.0, -PIXEL))[0]

    rng = np.random.default_rng(11)
    for c in (int(value) for value in rng.integers(0, 432000, 5000)):
        assert vscells.global_col_of_col0(west_as_read(-180.0 + (c - 0.5) * PIXEL), PIXEL) == c - 1
        assert vscells.global_col_of_col0(west_as_read(-180.0 + c * PIXEL), PIXEL) == c
    mask = _ragged(rng, 40, 300)
    north = 72.0                                            # k0 = 3
    base = _perimeter_m(mask, north, west=west_as_read(-180.0 + (186000 - 0.5) * PIXEL))
    for s in range(1, 8):
        wider = np.zeros((mask.shape[0], mask.shape[1] + s), bool)
        wider[:, s:] = mask
        assert _perimeter_m(wider, north, west=west_as_read(-180.0 + (186000 - s - 0.5) * PIXEL)) == base, s


def test_toward_the_poles_the_merge_is_held_at_89_5_degrees():
    held = vscells.merge_at_latitude(89.5, PIXEL, PIXEL, WGS84)
    assert 100 < held[0] < 120
    for latitude in (89.6, 89.999999, 90.0, -90.0, 1000.0, float("nan")):
        assert vscells.merge_at_latitude(latitude, PIXEL, PIXEL, WGS84) == held, latitude


def test_the_perimeter_of_made_up_catchments_matches_the_reference():
    """Cells two to nine columns wide, grids west of 180 W and catchments that touch the grid's edges: the package
    against the reference doubles (tests/reference/expected_perimeter_synthetic.npz), bit for bit."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with np.load(os.path.join(here, "reference", "expected_perimeter_synthetic.npz")) as archive:
        masks, transforms, want, ncol = archive["masks"], archive["transforms"], archive["perimeter_m"], int(archive["ncol"])
    k_seen = set()
    for packed, t, expected in zip(masks, transforms, want):
        mask = np.unpackbits(packed, axis=1)[:, :ncol].astype(bool)
        nrow = mask.shape[0]
        seq, cells, _ = _grid(mask, float(t[3]), float(t[0]), PIXEL)
        start, last = int(cells.min()), int(cells.max())
        col0 = vscells.global_col_of_col0(float(t[0]), PIXEL)
        status, got = vscells.perimeter_m(seq, nrow, ncol, 0, 0, float(t[3]), float(t[5]), col0, PIXEL, PIXEL,
                                          start // ncol, last // ncol, start // ncol, start % ncol,
                                          10 * cells.size + 100, WGS84)
        assert status == vscells.VS_OK
        assert np.float64(got).view(np.int64) == np.float64(expected).view(np.int64)
        middle = float(t[3]) + ((start // ncol + last // ncol + 1) / 2.0) * float(t[5])
        k_seen.add(vscells.merge_at_latitude(middle, PIXEL, PIXEL, WGS84)[0])
    assert min(k_seen) <= 2 and max(k_seen) >= 8


def test_the_start_must_be_the_topmost_leftmost_pixel_and_the_trace_close():
    mask = np.zeros((6, 6), bool)
    mask[1:4, 1:4] = True
    seq, cells, _ = _grid(mask, 10.0)
    col0 = vscells.global_col_of_col0(10.0, PIXEL)
    run = lambda r, c, k, cap: vscells.cells_length_m(seq, 6, 6, 0, 0, 10.0, -PIXEL, col0, PIXEL, PIXEL, r, c, k,  # noqa: E731
                                                      cap, WGS84)[0]
    assert run(0, 0, 1, 100) == vscells.VS_BAD_START       # not in the catchment
    assert run(1, 1, 0, 100) == vscells.VS_BAD_START       # no cells
    assert run(1, 1, 1, 100) == vscells.VS_OK
    assert run(1, 2, 1, 100) == vscells.VS_BAD_START       # on the top row, not the leftmost
    assert run(3, 1, 1, 100) == vscells.VS_BAD_START       # on the bottom row
    assert run(1, 1, 1, 5) == vscells.VS_NOT_CLOSED
    # a ring round a hole: a start on the rim of the hole is refused (the trace from it would go round the hole only)
    ring = np.zeros((10, 10), bool)
    ring[1:9, 1:9] = True
    ring[3:7, 3:7] = False
    seq, _, _ = _grid(ring, 10.0)
    ring_run = lambda r, c: vscells.cells_length_m(seq, 10, 10, 0, 0, 10.0, -PIXEL, col0, PIXEL, PIXEL, r, c, 1,  # noqa: E731
                                                   1000, WGS84)[0]
    assert ring_run(2, 2) == vscells.VS_BAD_START and ring_run(1, 1) == vscells.VS_OK
    # cells of three columns from global column 0 (west 180 W): (1, 0) and (1, 2) in one cell with (1, 1) empty, a
    # block below; the start (1, 2) gives the same cells as (1, 0) and is refused (the pixel left of it in its cell)
    cells3 = np.zeros((6, 6), bool)
    cells3[1, 0] = cells3[1, 2] = True
    cells3[2:5, 0:3] = True
    seq3, _, _ = _grid(cells3, 75.0, west=-180.0)
    col0_3 = vscells.global_col_of_col0(-180.0, PIXEL)
    run3 = lambda r, c: vscells.cells_length_m(seq3, 6, 6, 0, 0, 75.0, -PIXEL, col0_3, PIXEL, PIXEL, r, c, 3, 1000,  # noqa: E731
                                               WGS84)[0]
    assert col0_3 == 0 and run3(1, 2) == vscells.VS_BAD_START and run3(1, 0) == vscells.VS_OK
    # a cell above the start's cell holds a pixel the four pixels next to the start do not see: refused
    above = np.zeros((6, 6), bool)
    above[1, 2] = True
    above[2:5, :] = True
    seq4, _, _ = _grid(above, 75.0, west=-180.0)
    assert vscells.cells_length_m(seq4, 6, 6, 0, 0, 75.0, -PIXEL, col0_3, PIXEL, PIXEL, 2, 0, 3, 1000, WGS84)[0] \
        == vscells.VS_BAD_START


def test_a_catchment_that_a_merge_leaves_two_cells_has_no_perimeter():
    # three pixels along a row at 30 degrees (k0 = 1, w = 0.15): the merge of two columns leaves two cells, whose
    # trace has no length, so P has none though P(1) has one
    line = np.zeros((10, 30), bool)
    line[5, 10:13] = True                                   # column 0 at global column 216000, even
    assert vscells.global_col_of_col0(0.0, PIXEL) % 2 == 0
    seq, _, _ = _grid(line, 30.0, west=0.0)
    col0 = vscells.global_col_of_col0(0.0, PIXEL)
    k1 = vscells.cells_length_m(seq, 10, 30, 0, 0, 30.0, -PIXEL, col0, PIXEL, PIXEL, 5, 10, 1, 1000, WGS84)
    assert k1[0] == vscells.VS_OK and k1[1] > 0.0
    assert _perimeter_m(line, 30.0, west=0.0) == 0.0


# ---- the length, the width and the hull ----

def _plan_form(mask, target, upstream_area_km2, north, west=10.0, pixel=PIXEL):
    nrow, ncol = mask.shape
    _, cells, t = _grid(mask, north, west, pixel)
    target_cell = target[0] * ncol + target[1]
    length_km, far_row, far_col = sphere.basin_length(cells, cells.size, target_cell, ncol, west, pixel, north, -pixel,
                                                      t[0], t[1], t[2], t[3], WGS84)
    width_km = sphere.basin_width(cells, cells.size, target_cell, far_row, far_col, ncol, west, pixel, north, -pixel,
                                  t[0], t[1], t[2], t[3], WGS84)
    status, convexity = sphere.convexity(cells, cells.size, target_cell, np.float32(upstream_area_km2), ncol, west,
                                         pixel, north, -pixel, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], WGS84)
    return length_km, (far_row, far_col), width_km, status, convexity


def _pixel_area_km2(mask, north, pixel=PIXEL):
    rows = np.nonzero(mask)[0]
    return sum(earth.pixel_area_m2(north - (row + 0.5) * pixel, pixel, pixel) for row in rows) / 1.0e6


@pytest.mark.parametrize("north", (1.0, 45.0, 70.0, -20.0))
def test_a_rectangle_is_as_long_as_its_diagonal_and_convex(north):
    mask = np.zeros((40, 60), bool)
    mask[5:35, 10:50] = True                                # 30 rows by 40 columns, the target at its top left
    area = _pixel_area_km2(mask, north)
    length_km, farthest, width_km, status, convexity = _plan_form(mask, (5, 10), area, north)
    assert farthest == (34, 49)
    assert length_km == pytest.approx(earth.distance_m(*_centre(5, 10, north), *_centre(34, 49, north), WGS84) / 1e3,
                                      rel=1e-9)
    # the two other corners are the farthest across the diagonal: the width is twice the height of the triangle
    side_a = earth.distance_m(*_centre(5, 10, north), *_centre(5, 49, north), WGS84) / 1e3
    side_b = earth.distance_m(*_centre(5, 10, north), *_centre(34, 10, north), WGS84) / 1e3
    assert width_km == pytest.approx(2.0 * side_a * side_b / length_km, rel=2e-3)
    assert status == sphere.HULL_OK
    # the hull of the pixels holds the pixels; a rectangle of pixels fills it to the slivers between a parallel and
    # the great circles of its pieces
    assert 1.0 - 1e-6 < convexity <= 1.0


def test_an_l_shape_is_not_convex():
    mask = np.zeros((30, 30), bool)
    mask[2:22, 2:22] = True
    mask[2:12, 12:22] = False                               # a quarter taken out
    area = _pixel_area_km2(mask, 10.0)
    _, _, _, status, convexity = _plan_form(mask, (21, 2), area, 10.0)
    assert status == sphere.HULL_OK
    # the hull is the square less the triangle cut off between the two inner corners' far ends: 7/8 of the square
    assert convexity == pytest.approx(0.75 / 0.875, rel=2e-3)


def test_the_hull_is_the_hull_of_every_corner():
    generator = np.random.default_rng(7)
    for north in (2.0, 48.0, 78.0):
        for _ in range(6):
            size = int(generator.integers(8, 60))
            mask = np.zeros((size + 4, size + 4), bool)
            yy, xx = np.mgrid[0:size + 4, 0:size + 4]
            for _ in range(int(generator.integers(1, 5))):
                cy, cx = generator.uniform(2, size + 1, 2)
                ry, rx = generator.uniform(1.5, size / 2.0, 2)
                mask |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
            mask[0, :] = mask[-1, :] = False
            mask[:, 0] = mask[:, -1] = False
            if mask.sum() < 3:
                continue
            rows, cols = np.nonzero(mask)
            target = (int(rows[0]), int(cols[0]))
            # an area below any hull, so that the ratio is never capped and gives the hull's area back
            small_area = 1.0e-6
            _, _, _, status, convexity = _plan_form(mask, target, small_area, north)
            assert status == sphere.HULL_OK and 0.0 < convexity < 1.0
            hull_km2 = small_area / convexity
            # brute force: every corner of every pixel in the chart about the target, an exact monotone chain,
            # the ring's area by the same great-circle polygon
            nrow, ncol = mask.shape
            t = sphere.grid_tables(10.0, PIXEL, north, -PIXEL, nrow, ncol)
            k = sphere.centre_vector(target[0], target[1], t[0], t[1], t[2], t[3])
            horizontal = math.hypot(k[0], k[1])
            east = (-k[1] / horizontal, k[0] / horizontal, 0.0)
            north_vector = (k[1] * east[2] - k[2] * east[1], k[2] * east[0] - k[0] * east[2],
                            k[0] * east[1] - k[1] * east[0])
            corners = {(int(r) + dr, int(c) + dc) for r, c in zip(rows, cols) for dr in (0, 1) for dc in (0, 1)}
            points = []
            for r, c in corners:
                v = (t[4][r] * t[6][c], t[4][r] * t[7][c], t[5][r])
                towards = v[0] * k[0] + v[1] * k[1] + v[2] * k[2]
                points.append(((v[0] * east[0] + v[1] * east[1] + v[2] * east[2]) / towards,
                               (v[0] * north_vector[0] + v[1] * north_vector[1] + v[2] * north_vector[2]) / towards,
                               r, c))
            points.sort()

            def turn(o, a, b):
                return ((Fraction(a[0]) - Fraction(o[0])) * (Fraction(b[1]) - Fraction(o[1]))
                        - (Fraction(a[1]) - Fraction(o[1])) * (Fraction(b[0]) - Fraction(o[0])))
            lower, upper = [], []
            for p in points:
                while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            for p in reversed(points):
                while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            ring = lower[:-1] + upper[:-1]
            expected_km2 = sphere._corner_polygon_area_km2(
                np.array([p[2] for p in ring], np.int64), np.array([p[3] for p in ring], np.int64), len(ring),
                10.0, PIXEL, north, -PIXEL, t[4], t[5], t[6], t[7], WGS84)
            assert hull_km2 == pytest.approx(expected_km2, rel=1e-6)      # the area is passed as a float32


def test_no_boundary_pixel_gives_no_data():
    t = sphere.grid_tables(10.0, PIXEL, 10.0, -PIXEL, 4, 4)
    cells = np.zeros(1, np.int64)
    assert sphere.basin_length(cells, 0, 5, 4, 10.0, PIXEL, 10.0, -PIXEL, t[0], t[1], t[2], t[3], WGS84) \
        == (-9999.0, -1, -1)
    assert sphere.basin_width(cells, 0, 5, -1, -1, 4, 10.0, PIXEL, 10.0, -PIXEL, t[0], t[1], t[2], t[3], WGS84) == 0.0
    assert sphere.convexity(cells, 0, 5, np.float32(1.0), 4, 10.0, PIXEL, 10.0, -PIXEL, t[0], t[1], t[2], t[3], t[4],
                            t[5], t[6], t[7], WGS84) == (sphere.HULL_OK, -9999.0)


def test_a_grid_across_the_180th_meridian():
    # the same rectangle of pixels on a grid whose longitudes run past 180 and on one far from it: the same numbers
    mask = np.zeros((20, 40), bool)
    mask[3:15, 5:35] = True
    area = _pixel_area_km2(mask, 65.0)
    here = _plan_form(mask, (14, 5), area, 65.0, west=179.99)
    there = _plan_form(mask, (14, 5), area, 65.0, west=20.0)
    assert here[1] == there[1]
    assert here[0] == pytest.approx(there[0], rel=1e-12)
    assert here[2] == pytest.approx(there[2], rel=1e-9)
    assert here[4] == pytest.approx(there[4], rel=1e-9)
    # the perimeter's cells start at the global columns that are multiples of k (2 and 3 at 65 degrees): a grid
    # 7200 columns away has the same cells
    assert _perimeter_m(mask, 65.0, west=179.99) == _perimeter_m(mask, 65.0, west=173.99)


def test_the_target_alone_and_a_one_sided_catchment():
    # the target is the only pixel: it is its own farthest pixel, the length 0 and the width 0
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    length_km, farthest, width_km, status, convexity = _plan_form(mask, (2, 2), _pixel_area_km2(mask, 20.0), 20.0)
    assert (length_km, farthest, width_km) == (0.0, (2, 2), 0.0) and status == sphere.HULL_OK
    assert convexity == pytest.approx(1.0, abs=1e-6)
    # an L: the axis runs from the corner (6, 1) to the top of the leg (2, 10), and the row's far end lies off it, so
    # the width is well above 0; the row alone, whose axis runs along it, has a width of nearly 0
    mask = np.zeros((8, 12), bool)
    mask[6, 1:11] = True
    mask[2:7, 10] = True
    length_km, farthest, width_km, _, _ = _plan_form(mask, (6, 1), 1.0, 0.004)
    assert farthest == (2, 10)
    axis_only = mask.copy()
    axis_only[2:6, 10] = False
    _, _, width_of_the_row, _, _ = _plan_form(axis_only, (6, 1), 1.0, 0.004)
    # pixels on the axis itself: the side a pixel is on is the sign of a sine that is rounding there (1e-17), so
    # the width of a straight line of pixels is not exactly 0 but some 1e-10 km
    assert width_of_the_row < 1.0e-6 and width_km > 0.3


def test_the_walk_gives_every_catchment_what_it_has_alone():
    """boundary_walk on a network of 30 columns draining south into a bottom row that drains east: the catchments
    of the bottom row hold one another and grow column by column (the boundary set is updated, pixels leave it),
    those of the columns are lines that start afresh.  Every target against its own catchment measured alone."""
    import flowmorph
    nrow, ncol, north = 24, 30, 47.0
    d8 = np.full((nrow, ncol), 4, np.uint8)                 # south
    d8[nrow - 1, :] = 1                                     # the bottom row east
    d8[nrow - 1, ncol - 1] = 0                              # the mouth
    count = np.empty((nrow, ncol), np.float32)
    count[:, :] = np.arange(1, nrow + 1)[:, None]
    count[nrow - 1, :] = nrow * np.arange(1, ncol + 1)
    upa = (count * np.float32(0.005)).astype(np.float32)
    transform = (10.0, PIXEL, 0.0, north, 0.0, -PIXEL)
    fm = flowmorph.FlowMorph(d8, upa, transform, min_upstream_area_km2=0.004)
    walked = fm.boundary_walk()
    assert int(fm.is_target.sum()) == nrow * ncol
    west, pixel = fm.transform[0], fm.transform[1]          # the grid as the package took it (1/1200 to 1e-12)
    for row, col in [(0, 0), (5, 7), (nrow - 2, 29), (nrow - 1, 0), (nrow - 1, 1), (nrow - 1, 13), (nrow - 1, 29),
                     (11, 29), (nrow - 1, 28)]:
        mask = np.zeros((nrow, ncol), bool)
        if row == nrow - 1:
            mask[:, :col + 1] = True
        else:
            mask[:row + 1, col] = True
        length_km, _, width_km, status, convexity = _plan_form(mask, (row, col), float(upa[row, col]), north, west,
                                                               pixel)
        assert status == sphere.HULL_OK
        perimeter_m = _perimeter_m(mask, north, west, pixel, edge=PIXEL)
        assert walked["basin_length_km"][row, col] == np.float32(length_km)
        assert walked["basin_width_km"][row, col] == np.float32(width_km)
        assert walked["convexity"][row, col] == np.float32(convexity)
        expected_perimeter = np.float32(perimeter_m / 1000.0) if perimeter_m > 0.0 else np.float32(-9999.0)
        assert walked["perimeter_km"][row, col] == expected_perimeter


def test_a_catchment_next_to_a_pole_is_measured():
    """The perimeter measures the edges of the pixels at their own latitudes, so a catchment in the grid's row next to
    a pole is measured (a perimeter between pixel centres would need the centre of the pixel beyond that row, which
    has none on the Earth there)."""
    import flowmorph
    nrow, ncol = 6, 4
    d8 = np.full((nrow, ncol), 4, np.uint8)                 # south
    d8[nrow - 1, :] = 0                                     # every column its own mouth
    upa = np.tile(np.arange(1, nrow + 1, dtype=np.float32)[:, None], (1, ncol))
    at_the_pole = (10.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    walked = flowmorph.FlowMorph(d8, upa, at_the_pole, min_upstream_area_km2=0.5).boundary_walk()
    assert walked["perimeter_km"][nrow - 1, 0] > 0.0


def test_inputs_are_normalised_and_checked():
    import flowmorph
    from flowmorph.grid import snap_transform
    written = (10.0, 0.00083333, 0.0, 47.0, 0.0, -0.00083333)
    taken = snap_transform(written)
    assert snap_transform(taken) == taken and taken[1] == 0.000833333333
    d8 = np.array([[1, 0]], np.uint8)
    upa = np.array([[1.0, 20.0]], np.float32)
    seq = np.array([0, 1], np.int32)
    upg = np.array([1, 2], np.int64)
    # a target mask of integers: any non-zero value marks a target
    marked = flowmorph.boundary_walk(seq, upg, upa, np.array([0, 2]), (1, 2), written, earth.WGS84)
    assert marked["basin_length_km"][1] > 0.0 and marked["basin_length_km"][0] == np.float32(-9999.0)
    same = flowmorph.boundary_walk(seq, upg, upa, np.array([False, True]), (1, 2), taken, earth.WGS84)
    assert np.array_equal(marked["basin_length_km"], same["basin_length_km"])
    with pytest.raises(ValueError, match="not finite"):
        flowmorph.shape_arithmetic(upa, np.array([[-9999.0, np.inf]], np.float32), np.array([[-9999.0, 5.0]], np.float32),
                                   np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32),
                                   np.ones((1, 2), np.float32), np.array([[False, True]]))
    with pytest.raises(ValueError, match="too small"):
        flowmorph.FlowMorph(d8, upa, (10.0, 1e-13, 0.0, 47.0, 0.0, -1e-13))
