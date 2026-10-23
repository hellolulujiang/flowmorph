"""Step 3: the plan form of every pixel's upstream catchment, attributes [7] .. [10].

    [7]  perimeter_km     km  P: FlowMorph 1.0's perimeter, taken on cells close to square on the ground
                              (:mod:`flowmorph.vscells`): the outer contour traced through the boundary pixels'
                              centres from the topmost, then leftmost, boundary pixel, with the step weights of
                              Vossepoel & Smeulders (1982) for square pixels, on cells of k columns (k from the
                              pixel's height over its width at the middle of the catchment, interpolated between the
                              two whole numbers next to it).  The rims of holes inside the catchment are not part
                              of it.  No data for a catchment of fewer than three boundary pixels.
    [8]  basin_length_km  km  L_b: the length of the great circle from the pixel (the outlet of its catchment) to
                              the boundary pixel at the largest angle from it (Rigon et al. 1996; Willemin 2000).
    [9]  basin_width_km   km  W_b: the span of the boundary pixels across the great circle of L_b: on either
                              side the pixel farthest off it, measured along the great circle to the foot of its
                              perpendicular, the two added (Rigon et al. 1996).  For a broad catchment it can be
                              longer than L_b.
    [10] convexity        -   A / (area of the convex hull of the catchment's pixels), capped at 1 (Willemin 2000).
                              A is the pixel's upstream area.  The hull is that of the pixels themselves, of all
                              their corners, convex on the sphere with great-circle sides.

[8] to [10] are measured on the sphere of the geodetic latitudes and longitudes, with lengths and areas on the earth
model (:mod:`flowmorph.sphere`, :mod:`flowmorph.earth`; the WGS84 ellipsoid by default).  All four come from one set
of boundary pixels: the pixels of the catchment with at least one of their eight neighbours outside it (or outside
the grid).  Written where the upstream area is at least the target threshold, -9999 elsewhere.

HOW.  The targets are taken in rank order, upstream first.  A run of targets each of whose catchment holds the one
before is a stream, walked with one boundary set grown step by step: the pixels new to the catchment are the ranks
added below and above the old run, and those of them with a neighbour outside go in; the boundary pixels next to a
new pixel are checked again and dropped when they are no longer on the boundary.  The first target of a stream
builds its set from every rank of its run.  The set is the same set of pixels however it is built, and none of the
four attributes depends on the order it is held in (among boundary pixels at the same angle, L_b and W_b take
the one first in the grid's row-major order).
"""

import numpy as np
from numba import njit

from . import sphere, vscells
from .grid import north_up, pixel_edges_deg

NODATA = -9999.0
BOUNDARY_NAMES = ("perimeter_km", "basin_length_km", "basin_width_km", "convexity")

# what stopped the walk (the status of :func:`_boundary_walk`): 1 and 2 are faults of the input; 3 to 5 are limits of
# the method (the chart's reach, the poles, the pieces a hull's sides are held in), which no river basin on Earth
# reaches
_WALK_OK = 0
_WALK_CONTOUR_BAD_START = 1
_WALK_CONTOUR_NOT_CLOSED = 2
_WALK_HULL_BEYOND_CHART = 3
_WALK_HULL_ROUND_A_POLE = 4
_WALK_HULL_TOO_MANY_PIECES = 5

# the eight neighbours, clockwise from north
_NEIGHBOUR_ROW = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_NEIGHBOUR_COL = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


@njit
def _in_range(seq, row, col, nrow, ncol, first_rank, last_rank):
    if row < 0 or row >= nrow or col < 0 or col >= ncol:
        return False
    rank = seq[row * ncol + col]
    return rank >= 0 and rank >= first_rank and rank <= last_rank


@njit
def _is_outer_boundary(seq, row, col, nrow, ncol, first_rank, last_rank):
    for direction in range(8):
        if not _in_range(seq, row + _NEIGHBOUR_ROW[direction], col + _NEIGHBOUR_COL[direction], nrow, ncol,
                         first_rank, last_rank):
            return True
    return False


# ---- the boundary set: the pixels in an array, and each pixel's place in it (-1 when out) ----

@njit
def _set_add(cells, size, place, cell):
    if place[cell] >= 0:
        return size
    cells[size] = cell
    place[cell] = size
    return size + 1


@njit
def _set_remove(cells, size, place, cell):
    position = place[cell]
    if position < 0:
        return size
    last_cell = cells[size - 1]
    cells[position] = last_cell
    place[last_cell] = position
    place[cell] = -1
    return size - 1


@njit
def _boundary_walk(seq, nrow, ncol, cell_of_rank, target_cell, target_rank, target_first_rank, target_upa,
                   west, x_resolution, north, y_resolution, global_col_of_col0, pixel_width_deg, pixel_height_deg,
                   model):
    """(status, the target it stopped at, the four attributes of every target); the targets in rank order, rising."""
    target_count = target_cell.size
    outputs = np.full((target_count, 4), np.float32(NODATA), dtype=np.float32)
    (cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col,
     cos_latitude_of_row_edge, sin_latitude_of_row_edge, cos_longitude_of_col_edge,
     sin_longitude_of_col_edge) = sphere.grid_tables(west, x_resolution, north, y_resolution, nrow, ncol)
    cells = np.empty(max(cell_of_rank.size, 1), dtype=np.int64)
    place = np.full(nrow * ncol, -1, dtype=np.int64)
    size = 0
    previous_first = -1
    previous_last = -1
    for target in range(target_count):
        first_rank = target_first_rank[target]
        last_rank = target_rank[target]
        nested = previous_first >= 0 and first_rank <= previous_first and last_rank >= previous_last
        if not nested:
            # the first target of a stream: the set built from every rank of its run
            for index in range(size):
                place[cells[index]] = -1
            size = 0
            for rank in range(first_rank, last_rank + 1):
                cell = cell_of_rank[rank]
                if _is_outer_boundary(seq, cell // ncol, cell % ncol, nrow, ncol, first_rank, last_rank):
                    size = _set_add(cells, size, place, cell)
        else:
            # the next target downstream: the ranks new below and above the old run
            for rank in range(first_rank, previous_first):
                cell = cell_of_rank[rank]
                if _is_outer_boundary(seq, cell // ncol, cell % ncol, nrow, ncol, first_rank, last_rank):
                    size = _set_add(cells, size, place, cell)
            for rank in range(previous_last + 1, last_rank + 1):
                cell = cell_of_rank[rank]
                if _is_outer_boundary(seq, cell // ncol, cell % ncol, nrow, ncol, first_rank, last_rank):
                    size = _set_add(cells, size, place, cell)
            # the old boundary pixels next to a pixel new to the catchment: still on the boundary?
            index = size - 1
            while index >= 0:
                cell = cells[index]
                row = cell // ncol
                col = cell % ncol
                needs_recheck = False
                for direction in range(8):
                    neighbour_row = row + _NEIGHBOUR_ROW[direction]
                    neighbour_col = col + _NEIGHBOUR_COL[direction]
                    if neighbour_row < 0 or neighbour_row >= nrow or neighbour_col < 0 or neighbour_col >= ncol:
                        continue
                    neighbour_rank = seq[neighbour_row * ncol + neighbour_col]
                    if neighbour_rank < 0:
                        continue
                    if first_rank <= neighbour_rank <= last_rank and \
                            (neighbour_rank < previous_first or neighbour_rank > previous_last):
                        needs_recheck = True
                        break
                if needs_recheck and not _is_outer_boundary(seq, row, col, nrow, ncol, first_rank, last_rank):
                    size = _set_remove(cells, size, place, cell)
                index -= 1
        previous_first = first_rank
        previous_last = last_rank

        this_cell = target_cell[target]
        # [8] L_b, and the farthest boundary pixel for [9]
        length_km, farthest_row, farthest_col = sphere.basin_length(
            cells, size, this_cell, ncol, west, x_resolution, north, y_resolution, cos_latitude_of_row,
            sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col, model)
        outputs[target, 1] = np.float32(length_km)
        # [9] W_b
        if farthest_row >= 0:
            outputs[target, 2] = np.float32(sphere.basin_width(
                cells, size, this_cell, farthest_row, farthest_col, ncol, west, x_resolution, north, y_resolution,
                cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col, model))
        # [7] P: 1.0's perimeter on cells close to square on the ground, from the topmost, then leftmost, boundary
        # pixel (its row the catchment's first; the lowest boundary pixel's row its last)
        if size >= 3:
            start = cells[0]
            last = cells[0]
            for index in range(1, size):
                if cells[index] < start:
                    start = cells[index]
                if cells[index] > last:
                    last = cells[index]
            perimeter_status, perimeter_m = vscells.perimeter_m(
                seq, nrow, ncol, first_rank, last_rank, north, y_resolution, global_col_of_col0, pixel_width_deg,
                pixel_height_deg, start // ncol, last // ncol, start // ncol, start % ncol, size * 10 + 100, model)
            if perimeter_status != vscells.VS_OK:
                return (_WALK_CONTOUR_BAD_START if perimeter_status == vscells.VS_BAD_START
                        else _WALK_CONTOUR_NOT_CLOSED), target, outputs
            if perimeter_m > 0.0:
                outputs[target, 0] = np.float32(perimeter_m / 1000.0)
        # [10] the convexity
        hull_status, convexity = sphere.convexity(
            cells, size, this_cell, target_upa[target], ncol, west, x_resolution, north, y_resolution,
            cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col,
            cos_latitude_of_row_edge, sin_latitude_of_row_edge, cos_longitude_of_col_edge,
            sin_longitude_of_col_edge, model)
        if hull_status != sphere.HULL_OK:
            if hull_status == sphere.HULL_BEYOND_CHART:
                return _WALK_HULL_BEYOND_CHART, target, outputs
            if hull_status == sphere.HULL_ROUND_A_POLE:
                return _WALK_HULL_ROUND_A_POLE, target, outputs
            return _WALK_HULL_TOO_MANY_PIECES, target, outputs
        outputs[target, 3] = np.float32(convexity)
    return _WALK_OK, -1, outputs


_WALK_FAILURES = {
    _WALK_CONTOUR_BAD_START: "the trace of its outer contour did not start at its topmost-leftmost pixel",
    _WALK_CONTOUR_NOT_CLOSED: "the trace of its outer contour did not close within 10 steps per boundary pixel "
                              "plus 100",
    _WALK_HULL_BEYOND_CHART: "its convex hull reaches more than 80 degrees from the pixel",
    _WALK_HULL_ROUND_A_POLE: "its convex hull rings a pole",
    _WALK_HULL_TOO_MANY_PIECES: "the sides of its convex hull have too many pieces to hold",
}


def boundary_walk(seq_dfs, upg, upa, is_target, shape, transform, model, pixel_edges=None):
    """The four attributes of step 3, each a float32 array over the pixels (-9999 where not written).

    ``transform`` places the pixel centres and corners.  ``pixel_edges`` is the pixel's width and height in degrees
    for the lengths of its edges in the perimeter ([7]), as in 1.0: :func:`flowmorph.grid.pixel_edges_deg` of the
    transform when not given (1/1200 both on MERIT Hydro, as the C code takes them).

    Raises ValueError when a catchment cannot be measured.  A contour that does not close is a fault of the input.
    A convex hull that reaches more than 80 degrees (some 8,900 km) from its pixel, or that rings a pole, is beyond
    what the gnomonic chart of the hull measures: a limit of the method, which no river basin on Earth reaches (on a
    grid of the Earth it most often means that the transform is not the grid's)."""
    nrow, ncol = shape
    given_transform = transform
    transform = north_up(transform)               # as the C code takes it (a transform already taken so stays as
    #                                               it is); a rotated grid is refused, as the FlowMorph object does
    seq = np.ascontiguousarray(seq_dfs, dtype=np.int32).reshape(-1)
    network = seq >= 0
    cells = np.flatnonzero(network)
    ranks = seq[network].astype(np.int64)
    cell_of_rank = np.empty(ranks.size, dtype=np.int64)
    cell_of_rank[ranks] = cells
    targets = np.flatnonzero(np.asarray(is_target, dtype=bool).reshape(-1) & network)
    target_rank = seq[targets].astype(np.int64)
    order = np.argsort(target_rank, kind="stable")
    targets = targets[order]
    target_rank = target_rank[order]
    target_first_rank = target_rank - np.asarray(upg).reshape(-1)[targets].astype(np.int64) + 1
    target_upa = np.asarray(upa, dtype=np.float32).reshape(-1)[targets]
    # the pixel's size from the transform as given (put back on 1/N, not rounded to 1e-12), as the FlowMorph object
    # takes it
    pixel_width_deg, pixel_height_deg = pixel_edges_deg(given_transform) if pixel_edges is None else pixel_edges
    status, stopped_at, outputs = _boundary_walk(seq, nrow, ncol, cell_of_rank, targets.astype(np.int64),
                                                 target_rank, target_first_rank, target_upa, float(transform[0]),
                                                 float(transform[1]), float(transform[3]), float(transform[5]),
                                                 vscells.global_col_of_col0(transform[0], pixel_width_deg),
                                                 float(pixel_width_deg), float(pixel_height_deg), model)
    if status != _WALK_OK:
        cell = int(targets[stopped_at])
        raise ValueError(f"the catchment of the pixel at row {cell // ncol}, column {cell % ncol} cannot be "
                         f"measured: {_WALK_FAILURES[status]}")
    result = {}
    for index, name in enumerate(BOUNDARY_NAMES):
        values = np.full(seq.size, np.float32(NODATA), dtype=np.float32)
        values[targets] = outputs[:, index]
        result[name] = values
    return result


__all__ = ["boundary_walk", "BOUNDARY_NAMES"]
