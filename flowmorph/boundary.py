"""Step 3: the plan form of every pixel's upstream catchment, attributes [7] .. [10].

    [7]  perimeter_km     km  P: the catchment's outer boundary traced pixel by pixel (Moore neighbourhood), each step
                              weighted after Vossepoel & Smeulders (1982) on the real pixel edges at the step's
                              latitude: 0.980 x the edge for a step along a row or a column, (1.406 / sqrt 2) x the
                              pixel's diagonal for a diagonal step, and -0.091 x the mean edge for every corner, a
                              corner being a step along a row or column next to a diagonal step (either order), the
                              last step and the first included.  The trace starts at the topmost, then leftmost,
                              boundary pixel and stops when it is back there and its next step would repeat its first
                              (Jacob's stopping criterion).  No data when it does not come back.
    [8]  basin_length_km  km  L_par: the distance from the pixel (the outlet of its catchment) to the farthest pixel of
                              the boundary (Rigon et al. 1996; Willemin 2000).
    [9]  basin_width_km   km  L_perp: the span of the boundary pixels across the direction of L_par, in a plane centred
                              on the outlet (Rigon et al. 1996).  For a broad catchment it can be longer than L_par.
    [10] convexity        -   A / (area of the convex hull of the catchment's pixels), capped at 1 (Willemin 2000).
                              A is the pixel's upstream area.  The hull is that of the pixels themselves: the hull of
                              the four corners of each vertex pixel of the hull of the boundary pixel centres (or of
                              the pixels at the two ends of the centres' line, when they all lie on one).  Its area is
                              exact for its straight edges in longitude and latitude.

All four come from one set of boundary pixels: the pixels of the catchment with at least one of their eight
neighbours outside it (or outside the grid).  Distances, edges and hull areas are on the earth model
(:mod:`flowmorph.earth`, the WGS84 ellipsoid by default).  Written where the upstream area is at least the target
threshold, -9999 elsewhere.

HOW.  The targets are taken in rank order, upstream first.  A run of targets each of whose catchment holds the one
before is a stream, walked with one boundary set grown step by step: the pixels new to the catchment are the ranks
added below and above the old run, and those of them with a neighbour outside go in; the boundary pixels next to a
new pixel are checked again and dropped when they are no longer on the boundary.  The first target of a stream
builds its set from every rank of its run.  The set is the same set of pixels however it is built, and none of the
four attributes depends on the order it is held in (among boundary pixels equally far from the outlet, L_par takes
the one first in the grid's row-major order).
"""

import math

import numpy as np
from numba import njit

from .earth import distance_m, lonlat_polygon_area_m2, metres_per_degree

NODATA = -9999.0
BOUNDARY_NAMES = ("perimeter_km", "basin_length_km", "basin_width_km", "convexity")

# the eight neighbours, clockwise from north: even codes step along a column or a row, odd codes diagonally
_NEIGHBOUR_ROW = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_NEIGHBOUR_COL = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

PERIMETER_WEIGHT_EVEN_STEP = 0.980
PERIMETER_WEIGHT_ODD_STEP = 1.406 / 1.41421356
PERIMETER_WEIGHT_CORNER = 0.091


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


# ---- the boundary set: the cells in an array, and each cell's place in it (-1 when out) ----

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


# ---- the four attributes of one target ----

@njit
def _basin_length(cells, size, target_cell, ncol, west, x_resolution, north, y_resolution, model):
    """L_par in km, and the farthest boundary cell (-1 when none is farther than 0)."""
    if size < 1:
        return NODATA, -1
    target_latitude = north + (target_cell // ncol + 0.5) * y_resolution
    target_longitude = west + (target_cell % ncol + 0.5) * x_resolution
    largest = 0.0
    farthest = -1
    for index in range(size):
        cell = cells[index]
        d = distance_m(target_latitude, target_longitude, north + (cell // ncol + 0.5) * y_resolution,
                       west + (cell % ncol + 0.5) * x_resolution, model)
        if d > largest or (d == largest and farthest >= 0 and cell < farthest):
            largest = d
            farthest = cell
    return largest / 1000.0, farthest


@njit
def _basin_width(cells, size, target_cell, farthest_cell, ncol, west, x_resolution, north, y_resolution, model):
    if size < 2 or farthest_cell < 0:
        return 0.0
    target_longitude = west + (target_cell % ncol + 0.5) * x_resolution
    target_latitude = north + (target_cell // ncol + 0.5) * y_resolution
    farthest_longitude = west + (farthest_cell % ncol + 0.5) * x_resolution
    farthest_latitude = north + (farthest_cell // ncol + 0.5) * y_resolution
    metres_latitude, metres_longitude = metres_per_degree(target_latitude, model)
    axis_x = (farthest_longitude - target_longitude) * metres_longitude
    axis_y = (farthest_latitude - target_latitude) * metres_latitude
    axis_length = math.sqrt(axis_x * axis_x + axis_y * axis_y)
    if axis_length <= 0.0:
        return 0.0
    perpendicular_x = -axis_y / axis_length
    perpendicular_y = axis_x / axis_length
    projection_min = 1e30
    projection_max = -1e30
    for index in range(size):
        cell = cells[index]
        point_x = (west + (cell % ncol + 0.5) * x_resolution - target_longitude) * metres_longitude
        point_y = (north + (cell // ncol + 0.5) * y_resolution - target_latitude) * metres_latitude
        projection = point_x * perpendicular_x + point_y * perpendicular_y
        if projection < projection_min:
            projection_min = projection
        if projection > projection_max:
            projection_max = projection
    return (projection_max - projection_min) / 1000.0


@njit
def _edges_m(latitude, edge_x_deg, edge_y_deg, model):
    metres_latitude, metres_longitude = metres_per_degree(latitude, model)
    return metres_latitude * edge_y_deg, metres_longitude * edge_x_deg


@njit
def _moore_perimeter(cells, size, seq, first_rank, last_rank, nrow, ncol, north, y_resolution, edge_x_deg, edge_y_deg,
                     model):
    if size < 3:
        return NODATA
    start = cells[0]
    for index in range(1, size):
        if cells[index] < start:
            start = cells[index]
    start_row = start // ncol
    start_col = start % ncol
    backtrack = -1
    for direction in range(8):
        if not _in_range(seq, start_row + _NEIGHBOUR_ROW[direction], start_col + _NEIGHBOUR_COL[direction], nrow,
                         ncol, first_rank, last_rank):
            backtrack = direction
            break
    if backtrack < 0:
        return NODATA
    perimeter = 0.0
    row = start_row
    col = start_col
    previous_code = -1
    first_code = -1
    step_count = 0
    step_count_max = size * 10 + 100
    first_step_row = -1
    first_step_col = -1
    came_back = False
    while True:
        step_found = False
        for turn in range(1, 9):
            code = (backtrack + turn) % 8
            next_row = row + _NEIGHBOUR_ROW[code]
            next_col = col + _NEIGHBOUR_COL[code]
            if not _in_range(seq, next_row, next_col, nrow, ncol, first_rank, last_rank):
                continue
            if step_count > 0 and row == start_row and col == start_col and next_row == first_step_row \
                    and next_col == first_step_col:
                came_back = True
                break
            if step_count == 0:
                first_step_row = next_row
                first_step_col = next_col
            # the edges at the latitude halfway along the step
            step_latitude = north + ((row + next_row) / 2.0 + 0.5) * y_resolution
            north_south_edge, east_west_edge = _edges_m(step_latitude, edge_x_deg, edge_y_deg, model)
            if code % 2 == 0:
                edge = north_south_edge if (code == 0 or code == 4) else east_west_edge
                perimeter += PERIMETER_WEIGHT_EVEN_STEP * edge
            else:
                diagonal = math.sqrt(north_south_edge * north_south_edge + east_west_edge * east_west_edge)
                perimeter += PERIMETER_WEIGHT_ODD_STEP * diagonal
            if previous_code >= 0 and (code % 2) != (previous_code % 2):
                perimeter -= PERIMETER_WEIGHT_CORNER * 0.5 * (north_south_edge + east_west_edge)
            if first_code < 0:
                first_code = code
            previous_code = code
            row = next_row
            col = next_col
            backtrack = (code + 4) % 8
            step_found = True
            step_count += 1
            break
        if not step_found:
            break
        if came_back or step_count >= step_count_max:
            break
    if step_count < 3 or not came_back:
        return NODATA
    # the trace ends where it started: its last step and its first are consecutive too
    if previous_code >= 0 and first_code >= 0 and (previous_code % 2) != (first_code % 2):
        north_south_edge, east_west_edge = _edges_m(north + (start_row + 0.5) * y_resolution, edge_x_deg,
                                                    edge_y_deg, model)
        perimeter -= PERIMETER_WEIGHT_CORNER * 0.5 * (north_south_edge + east_west_edge)
    return perimeter / 1000.0 if perimeter > 0.0 else NODATA


@njit
def _point_before(x1, y1, x2, y2):
    return x1 < x2 or (x1 == x2 and y1 < y2)


@njit
def _convex_hull(x, y):
    """Andrew's monotone chain on the points (x, y): the indices of the hull's corners, anticlockwise, or an empty
    array when there are fewer than three."""
    n = x.size
    if n < 3:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(y, kind="mergesort")
    order = order[np.argsort(x[order], kind="mergesort")]
    distinct = np.empty(n, dtype=np.int64)
    distinct_count = 1
    distinct[0] = order[0]
    for position in range(1, n):
        point = order[position]
        kept = distinct[distinct_count - 1]
        if x[point] != x[kept] or y[point] != y[kept]:
            distinct[distinct_count] = point
            distinct_count += 1
    if distinct_count < 3:
        return np.empty(0, dtype=np.int64)
    hull = np.empty(2 * distinct_count, dtype=np.int64)
    hull_count = 0
    for position in range(distinct_count):
        point = distinct[position]
        while hull_count >= 2:
            origin = hull[hull_count - 2]
            first = hull[hull_count - 1]
            cross = ((x[first] - x[origin]) * (y[point] - y[origin])
                     - (y[first] - y[origin]) * (x[point] - x[origin]))
            if cross <= 0.0:
                hull_count -= 1
            else:
                break
        hull[hull_count] = point
        hull_count += 1
    lower_end = hull_count + 1
    for position in range(distinct_count - 2, -1, -1):
        point = distinct[position]
        while hull_count >= lower_end:
            origin = hull[hull_count - 2]
            first = hull[hull_count - 1]
            cross = ((x[first] - x[origin]) * (y[point] - y[origin])
                     - (y[first] - y[origin]) * (x[point] - x[origin]))
            if cross <= 0.0:
                hull_count -= 1
            else:
                break
        hull[hull_count] = point
        hull_count += 1
    hull_count -= 1
    if hull_count < 3:
        return np.empty(0, dtype=np.int64)
    return hull[:hull_count].copy()


@njit
def _convexity(cells, size, target_cell, upstream_area_km2, ncol, west, x_resolution, north, y_resolution, model):
    if size < 1:
        return NODATA
    target_longitude = west + (target_cell % ncol + 0.5) * x_resolution
    target_latitude = north + (target_cell // ncol + 0.5) * y_resolution
    metres_latitude, metres_longitude = metres_per_degree(target_latitude, model)
    x = np.empty(size, dtype=np.float64)
    y = np.empty(size, dtype=np.float64)
    longitude = np.empty(size, dtype=np.float64)
    latitude = np.empty(size, dtype=np.float64)
    for index in range(size):
        cell = cells[index]
        longitude[index] = west + (cell % ncol + 0.5) * x_resolution
        latitude[index] = north + (cell // ncol + 0.5) * y_resolution
        x[index] = (longitude[index] - target_longitude) * metres_longitude
        y[index] = (latitude[index] - target_latitude) * metres_latitude
    # the two ends of the centres in (x, y) order: the ends of their line when they all lie on one
    lowest = 0
    highest = 0
    for index in range(1, size):
        if _point_before(x[index], y[index], x[lowest], y[lowest]):
            lowest = index
        if _point_before(x[highest], y[highest], x[index], y[index]):
            highest = index
    hull = _convex_hull(x, y)
    if hull.size >= 3:
        vertices = hull
    else:
        vertices = np.array([lowest, highest], dtype=np.int64)
    # the corners of the vertex pixels, in the same plane as the centres
    half_longitude = 0.5 * abs(x_resolution)
    half_latitude = 0.5 * abs(y_resolution)
    corner_count = 4 * vertices.size
    corner_x = np.empty(corner_count, dtype=np.float64)
    corner_y = np.empty(corner_count, dtype=np.float64)
    corner_longitude = np.empty(corner_count, dtype=np.float64)
    corner_latitude = np.empty(corner_count, dtype=np.float64)
    longitude_offsets = (-half_longitude, half_longitude, -half_longitude, half_longitude)
    latitude_offsets = (-half_latitude, -half_latitude, half_latitude, half_latitude)
    position = 0
    for vertex_index in range(vertices.size):
        vertex = vertices[vertex_index]
        for corner in range(4):
            corner_longitude[position] = longitude[vertex] + longitude_offsets[corner]
            corner_latitude[position] = latitude[vertex] + latitude_offsets[corner]
            corner_x[position] = (corner_longitude[position] - target_longitude) * metres_longitude
            corner_y[position] = (corner_latitude[position] - target_latitude) * metres_latitude
            position += 1
    pixel_hull = _convex_hull(corner_x, corner_y)
    if pixel_hull.size < 3:
        return NODATA
    hull_km2 = lonlat_polygon_area_m2(corner_longitude[pixel_hull], corner_latitude[pixel_hull], model) / 1.0e6
    if hull_km2 > 0.0 and upstream_area_km2 > 0.0:
        convexity = np.float64(upstream_area_km2) / hull_km2
        return 1.0 if convexity > 1.0 else convexity
    return NODATA


@njit
def _boundary_walk(seq, nrow, ncol, cell_of_rank, target_cell, target_rank, target_first_rank, target_upa,
                   west, x_resolution, north, y_resolution, edge_x_deg, edge_y_deg, model):
    """The four attributes of every target; the targets in rank order, rising."""
    target_count = target_cell.size
    outputs = np.full((target_count, 4), np.float32(NODATA), dtype=np.float32)
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
        length, farthest = _basin_length(cells, size, this_cell, ncol, west, x_resolution, north, y_resolution,
                                         model)
        outputs[target, 1] = np.float32(length)
        if farthest >= 0:
            outputs[target, 2] = np.float32(_basin_width(cells, size, this_cell, farthest, ncol, west,
                                                         x_resolution, north, y_resolution, model))
        outputs[target, 0] = np.float32(_moore_perimeter(cells, size, seq, first_rank, last_rank, nrow, ncol,
                                                         north, y_resolution, edge_x_deg, edge_y_deg, model))
        outputs[target, 3] = np.float32(_convexity(cells, size, this_cell, target_upa[target], ncol, west,
                                                   x_resolution, north, y_resolution, model))
    return outputs


def boundary_walk(seq_dfs, upg, upa, is_target, shape, transform, model, pixel_edges=None):
    """The four attributes of step 3, each a float32 array over the cells (-9999 where not written).

    ``transform`` places the pixel centres; ``pixel_edges`` (width, height in degrees) are the pixel's edges in the
    perimeter, the transform's pixel size when not given (:func:`flowmorph.grid.pixel_edges_deg`)."""
    if pixel_edges is None:
        pixel_edges = (abs(float(transform[1])), abs(float(transform[5])))
    nrow, ncol = shape
    seq = np.ascontiguousarray(seq_dfs, dtype=np.int32).reshape(-1)
    network = seq >= 0
    cells = np.flatnonzero(network)
    ranks = seq[network].astype(np.int64)
    cell_of_rank = np.empty(ranks.size, dtype=np.int64)
    cell_of_rank[ranks] = cells
    targets = np.flatnonzero(np.asarray(is_target).reshape(-1) & network)
    target_rank = seq[targets].astype(np.int64)
    order = np.argsort(target_rank, kind="stable")
    targets = targets[order]
    target_rank = target_rank[order]
    target_first_rank = target_rank - np.asarray(upg).reshape(-1)[targets].astype(np.int64) + 1
    target_upa = np.asarray(upa, dtype=np.float32).reshape(-1)[targets]
    outputs = _boundary_walk(seq, nrow, ncol, cell_of_rank, targets.astype(np.int64), target_rank,
                             target_first_rank, target_upa, float(transform[0]), float(transform[1]),
                             float(transform[3]), float(transform[5]), float(pixel_edges[0]), float(pixel_edges[1]),
                             model)
    result = {}
    for index, name in enumerate(BOUNDARY_NAMES):
        values = np.full(seq.size, np.float32(NODATA), dtype=np.float32)
        values[targets] = outputs[:, index]
        result[name] = values
    return result


__all__ = ["boundary_walk", "BOUNDARY_NAMES"]
