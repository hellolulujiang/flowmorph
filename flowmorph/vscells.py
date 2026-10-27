"""The perimeter of a catchment: the weighted Moore trace, taken on cells close to square on the ground.

``the trace``        the outer contour traced through the centres of the catchment's boundary pixels (Moore
                     neighbourhood, from the topmost, then leftmost, pixel, done by the stopping rule of Jacob: back on
                     the start pixel and about to make the first step again), each step weighted as Vossepoel &
                     Smeulders (1982) weigh it for square pixels: 0.980 a step along a row or a column, 1.406 /
                     1.41421356 of the diagonal for a diagonal step, and -0.091 of the mean of the two edges for a
                     corner, where a step along a row or a column meets a diagonal one (the last step and the first
                     included).  The edges are those of a pixel at the latitude halfway along the step, on the earth
                     model.  No constant.
``the cells``        the weights were made for straight lines on a square grid.  A pixel of a longitude-latitude grid
                     is a = dy / dx times as tall as it is wide on the ground, about 1 / cos(latitude) on a grid with
                     the same angular step in both (2 at 60 degrees, 3.9 at 75, 5.8 at 80), and there the trace on
                     the pixels reads long: on smooth shapes of known perimeter by 2 % at 60 degrees, 7.6 % at 75 and
                     10 % at 80.  So the columns are merged k at a time into cells and the trace and the weights are
                     taken on the cells: a cell is in the catchment when one of its k pixels is, and its east-west
                     edge is k dx.  The cells start at the columns of the global grid that are multiples of k
                     (:func:`global_col_of_col0`; MERIT Hydro's column c is global column c - 1).
``continuous``       a is taken at the middle of the catchment's rows; k0 = max(1, floor(a)),
                     w = min(max(a - k0, 0), 1), and P = (1 - w) P(k0) + w P(k0 + 1), so that P has no step where k
                     changes.  Where a <= 1 (within 6.59 degrees of the equator on WGS84, with the same angular step in
                     both) P is the trace on the pixels themselves.  A catchment that a merge leaves one or two cells
                     (a few pixels) has no P, as a catchment of one or two pixels has none.

On smooth shapes of known perimeter it reads at every latitude as the trace on the pixels does at the equator: 0.98
of the perimeter for radii of 10 to 25 pixels, 0.994 for 30 to 80.  The merge measures on cells close to square on
the ground, which fills the notches of a divide narrower than a cell; where the k cells start moves P by a median
0.4 to 1.1 %, with no bias.  The pixel's width and height in degrees are those of the grid put back on 1/N
(:func:`flowmorph.grid.pixel_edges_deg`): 1/1200 both on MERIT Hydro.
"""

import math

import numpy as np
from numba import njit

from .earth import metres_per_degree

WEIGHT_EVEN_STEP = 0.980
WEIGHT_ODD_STEP = 1.406 / 1.41421356            # the square root of 2 taken to eight decimals
WEIGHT_CORNER = 0.091

VS_OK = 0
VS_BAD_START = 1        # k < 1, or the start is not in the catchment or not its topmost-leftmost pixel
VS_NOT_CLOSED = 2       # no closed trace within the cap

# the eight neighbours, clockwise from north: even codes step along a column or a row, odd codes step diagonally
_NEIGHBOUR_ROW = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_NEIGHBOUR_COL = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


@njit
def pixel_edges_m(latitude_deg, pixel_width_deg, pixel_height_deg, model):
    """The north-south and the east-west edge (m) of a pixel at a latitude, on the earth model."""
    metres_per_degree_latitude, metres_per_degree_longitude = metres_per_degree(latitude_deg, model)
    return metres_per_degree_latitude * pixel_height_deg, metres_per_degree_longitude * pixel_width_deg


@njit
def merge_at_latitude(latitude_deg, pixel_width_deg, pixel_height_deg, model):
    """(k0, w) at a latitude: k0 = max(1, floor(a)), w = min(max(a - k0, 0), 1), a = dy / dx; beyond 89.5 degrees (and
    for a latitude that is no number) the merge of 89.5, so that k stays bounded (a pixel's width goes to 0 toward a
    pole; MERIT Hydro ends at 85 N and 60 S)."""
    merge_latitude_deg = latitude_deg
    if not (abs(merge_latitude_deg) <= 89.5):
        merge_latitude_deg = -89.5 if merge_latitude_deg < 0.0 else 89.5
    north_south_edge_m, east_west_edge_m = pixel_edges_m(merge_latitude_deg, pixel_width_deg, pixel_height_deg, model)
    aspect = north_south_edge_m / east_west_edge_m
    k0 = int(math.floor(aspect))
    if k0 < 1:
        k0 = 1
    w = aspect - float(k0)
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    return k0, w


@njit
def _in_range(seq, row, col, nrow, ncol, first_rank, last_rank):
    if row < 0 or row >= nrow or col < 0 or col >= ncol:
        return False
    rank = seq[row * ncol + col]
    return rank >= 0 and rank >= first_rank and rank <= last_rank


@njit
def _cell_is_inside(seq, nrow, ncol, first_rank, last_rank, global_col_of_col0, row, cell, k):
    """The cell (row, cell) of k columns holds a pixel of the catchment [first_rank, last_rank] (only the cell's columns
    that lie on the grid are asked)."""
    if row < 0 or row >= nrow:
        return False
    first_col = cell * k - global_col_of_col0
    end_col = first_col + k
    if first_col < 0:
        first_col = 0
    if end_col > ncol:
        end_col = ncol
    for col in range(first_col, end_col):
        if _in_range(seq, row, col, nrow, ncol, first_rank, last_rank):
            return True
    return False


@njit
def cells_length_m(seq, nrow, ncol, first_rank, last_rank, north, y_resolution, global_col_of_col0, pixel_width_deg,
                   pixel_height_deg, start_row, start_col, k, step_count_max, model):
    """(status, length in m) of the trace on cells of k columns, from the topmost, then leftmost, pixel of the
    catchment.  The length is 0 when the trace on the cells made fewer than three steps (one or two cells)."""
    if k < 1 or not _in_range(seq, start_row, start_col, nrow, ncol, first_rank, last_rank):
        return VS_BAD_START, 0.0
    start_cell = (start_col + global_col_of_col0) // k
    # the topmost-leftmost pixel: no cell of the catchment in the row above next to its cell or left of its cell in its
    # row, and no pixel of the catchment left of it in its own cell (the trace below also refuses a cell above the
    # start or left of it, which it meets wherever else they lie)
    if (_cell_is_inside(seq, nrow, ncol, first_rank, last_rank, global_col_of_col0, start_row - 1, start_cell - 1, k)
            or _cell_is_inside(seq, nrow, ncol, first_rank, last_rank, global_col_of_col0, start_row - 1, start_cell, k)
            or _cell_is_inside(seq, nrow, ncol, first_rank, last_rank, global_col_of_col0, start_row - 1,
                               start_cell + 1, k)
            or _cell_is_inside(seq, nrow, ncol, first_rank, last_rank, global_col_of_col0, start_row, start_cell - 1,
                               k)):
        return VS_BAD_START, 0.0
    first_col_of_cell = start_cell * k - global_col_of_col0
    for col in range(first_col_of_cell if first_col_of_cell > 0 else 0, start_col):
        if _in_range(seq, start_row, col, nrow, ncol, first_rank, last_rank):
            return VS_BAD_START, 0.0
    backtrack_direction = -1
    for direction in range(8):
        if not _cell_is_inside(seq, nrow, ncol, first_rank, last_rank, global_col_of_col0,
                               start_row + _NEIGHBOUR_ROW[direction], start_cell + _NEIGHBOUR_COL[direction], k):
            backtrack_direction = direction
            break
    if backtrack_direction < 0:
        return VS_BAD_START, 0.0

    perimeter_m = 0.0
    current_row = start_row
    current_cell = start_cell
    previous_code = -1
    first_code = -1
    step_count = 0
    first_step_row = -1
    first_step_cell = -1
    trace_came_back_to_start = False
    while True:
        step_found = False
        for turn in range(1, 9):
            code = (backtrack_direction + turn) % 8
            next_row = current_row + _NEIGHBOUR_ROW[code]
            next_cell = current_cell + _NEIGHBOUR_COL[code]
            if not _cell_is_inside(seq, nrow, ncol, first_rank, last_rank, global_col_of_col0, next_row, next_cell,
                                   k):
                continue
            if (step_count > 0 and current_row == start_row and current_cell == start_cell
                    and next_row == first_step_row and next_cell == first_step_cell):
                trace_came_back_to_start = True
                break
            if next_row < start_row or (next_row == start_row and next_cell < start_cell):
                return VS_BAD_START, 0.0    # a cell above the start or left of it: not the topmost-leftmost pixel
            if step_count == 0:
                first_step_row = next_row
                first_step_cell = next_cell
            # the edges at the latitude halfway along the step; a cell is k pixels wide
            step_latitude = north + ((float(current_row) + float(next_row)) / 2.0 + 0.5) * y_resolution
            north_south_edge_m, east_west_edge_m = pixel_edges_m(step_latitude, pixel_width_deg, pixel_height_deg,
                                                                 model)
            east_west_edge_m = float(k) * east_west_edge_m
            if code % 2 == 0:
                step_edge_m = north_south_edge_m if (code == 0 or code == 4) else east_west_edge_m
                perimeter_m += WEIGHT_EVEN_STEP * step_edge_m
            else:
                diagonal_m = math.sqrt(north_south_edge_m * north_south_edge_m + east_west_edge_m * east_west_edge_m)
                perimeter_m += WEIGHT_ODD_STEP * diagonal_m
            # a corner: a step along a row or column next to a diagonal step
            if previous_code >= 0 and (code % 2) != (previous_code % 2):
                perimeter_m -= WEIGHT_CORNER * 0.5 * (north_south_edge_m + east_west_edge_m)
            if first_code < 0:
                first_code = code
            previous_code = code
            current_row = next_row
            current_cell = next_cell
            backtrack_direction = (code + 4) % 8
            step_found = True
            step_count += 1
            break
        if not step_found:
            break           # back at the start, or a cell alone
        if step_count >= step_count_max:
            break

    if not trace_came_back_to_start:
        if step_count == 0:
            return VS_OK, 0.0               # one cell: no length
        return VS_NOT_CLOSED, 0.0
    if step_count < 3:
        return VS_OK, 0.0                   # two cells: no length
    # the trace ends where it started: its last step and its first are consecutive too
    if previous_code >= 0 and first_code >= 0 and (previous_code % 2) != (first_code % 2):
        north_south_edge_m, east_west_edge_m = pixel_edges_m(north + (float(start_row) + 0.5) * y_resolution,
                                                             pixel_width_deg, pixel_height_deg, model)
        east_west_edge_m = float(k) * east_west_edge_m
        perimeter_m -= WEIGHT_CORNER * 0.5 * (north_south_edge_m + east_west_edge_m)
    return VS_OK, (perimeter_m if perimeter_m > 0.0 else 0.0)


@njit
def perimeter_m(seq, nrow, ncol, first_rank, last_rank, north, y_resolution, global_col_of_col0, pixel_width_deg,
                pixel_height_deg, first_row, last_row, start_row, start_col, step_count_max, model):
    """(status, P in m): (1 - w) P(k0) + w P(k0 + 1), k0 and w at the middle of the catchment's rows (the edge halfway
    between the top edge of its first row and the bottom edge of its last).  P is 0 (no data) when a trace it needs
    has no length: a catchment that the merge leaves one or two cells (a few pixels), so that P is continuous in
    latitude for every catchment of three cells or more on both merges, and absent below."""
    middle_latitude = north + ((float(first_row) + float(last_row) + 1.0) / 2.0) * y_resolution
    k0, w = merge_at_latitude(middle_latitude, pixel_width_deg, pixel_height_deg, model)
    status, length_k0_m = cells_length_m(seq, nrow, ncol, first_rank, last_rank, north, y_resolution,
                                         global_col_of_col0, pixel_width_deg, pixel_height_deg, start_row, start_col,
                                         k0, step_count_max, model)
    if status != VS_OK:
        return status, 0.0
    if not length_k0_m > 0.0:
        return VS_OK, 0.0
    if w == 0.0:
        return VS_OK, length_k0_m
    status, length_k1_m = cells_length_m(seq, nrow, ncol, first_rank, last_rank, north, y_resolution,
                                         global_col_of_col0, pixel_width_deg, pixel_height_deg, start_row, start_col,
                                         k0 + 1, step_count_max, model)
    if status != VS_OK:
        return status, 0.0
    if not length_k1_m > 0.0:
        return VS_OK, 0.0
    return VS_OK, (1.0 - w) * length_k0_m + w * length_k1_m


def global_col_of_col0(west, pixel_width_deg):
    """The column of the global grid (counted from 180 W) of the grid's column 0: the whole pixels between 180 W and
    its west edge, a quarter of a pixel added and the result rounded down.  A grid whose edges lie on 180 W
    (HydroSHEDS) gives its column; MERIT Hydro's column c, whose west edge lies half a pixel west of it, gives c - 1
    in every window.  (Rounded to the nearest whole number instead, MERIT Hydro's half pixel would be decided by the
    last bits of the header.)  Every window gives the same global columns on a grid whose pixel edges lie on 180 W or
    half a pixel off it (MERIT Hydro, HydroSHEDS); on a grid offset by another fraction of a pixel windows may differ
    by a column.  The columns count in the window's own longitudes: the same pixels given 360 degrees apart start
    their cells at other columns where k does not divide 432,000 (k of 7 or more: on MERIT Hydro north of 81.8
    degrees, where no land reaches 180 E)."""
    return int(math.floor((float(west) + 180.0) / float(pixel_width_deg) + 0.25))


__all__ = ["perimeter_m", "cells_length_m", "merge_at_latitude", "pixel_edges_m", "global_col_of_col0",
           "VS_OK", "VS_BAD_START", "VS_NOT_CLOSED"]
