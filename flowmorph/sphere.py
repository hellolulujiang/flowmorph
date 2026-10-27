"""The plan form of a catchment on the earth: the basin length, the basin width and the convex hull.

Every centre and every corner of a pixel is a point of the sphere of the geodetic latitudes and longitudes, and a
length is that of a great circle of that sphere on the earth model (:mod:`flowmorph.earth`), added up in pieces of
at most 5 km.

``basin length``  L_b.  The farthest pixel of the boundary is the one at the largest angle from the target (the
                  smallest cosine of the two unit vectors; on equal cosines the first in the grid's row-major
                  order), and L_b is the length of the great circle to it.
``basin width``   W_b.  The span of the boundary pixels across the great circle of L_b (the axis): of the
                  pixels on either side of it, the one at the largest angle off the axis (by the sine of that
                  angle, the pixel's unit vector against the axis's pole; on equal sines the first in row-major
                  order), measured along the great circle to the foot of its perpendicular on the axis; W_b is
                  the sum of the two.  A pixel on the axis is on neither side.
``convex hull``   the hull of the catchment's pixels themselves, of all their corners.  On the sphere the convex
                  hull of points within a hemisphere is their convex hull in a gnomonic chart, which maps great
                  circles to straight lines.  A meridian is a great circle, so of the corners on one column edge
                  only the topmost and the bottommost can be corners of the hull: two corners a column edge, from
                  the topmost and the bottommost boundary pixel of the columns on either side, give the hull of
                  every pixel.  Its area is taken on the earth model with great-circle sides, every side cut into
                  pieces of at most 5 km straight in longitude and latitude.

Measured instead in one plane centred on the target, with the target's east-west scale everywhere, a pixel 20
degrees poleward of a target at 30 N comes out 1.35 times too far east or west, the width of a large northern river
some 30 % short, and the hull of a large high-latitude catchment some 1 % off.  On the 1,929 catchments of the
bundled basin (729 km2 at 24 N) the sphere and that plane differ in the width by less than 0.2 % for all but 8 and
in the hull by less than 0.02 % for all; the length agrees to float32 rounding but for 8, where another boundary
pixel is the farthest (by up to 0.4 %), and there the axis turns and the width with it (by -15 to +6 %).

The grid's tables (:func:`grid_tables`) hold the cosine and the sine of the latitude of every row's centre and edge
and of the longitude of every column's centre and edge, so that a centre or a corner is a unit vector without a call
to cos or sin.  The centres' tables run one pixel beyond the grid on every side (index row + 1, column + 1).
"""

import math

import numpy as np
from numba import njit

from .clib import fma, remainder, sincos
from .earth import distance_m, lonlat_polygon_area_m2

NODATA = -9999.0
PIECE_M = 5000.0                           # a length is the sum of great-circle pieces of at most this
RADIUS_FOR_PIECES_M = 6400000.0            # the radius the number of pieces is taken with: above any of the earth's
CHART_MIN_COSINE = 0.17364817766693041     # cos 80 degrees: a point of a chart lies within this of its centre
_TO_RADIANS = math.pi / 180.0
_TO_DEGREES = 180.0 / math.pi
_INT32_MAX = 2147483647

HULL_OK = 0
HULL_BEYOND_CHART = 1                      # a corner lies more than 80 degrees from the target
HULL_ROUND_A_POLE = 2                      # the hull rings a pole
HULL_TOO_MANY_PIECES = 3                   # the pieces of the hull's sides cannot be held


# ---- the grid ----

@njit
def grid_tables(west, x_resolution, north, y_resolution, nrow, ncol):
    """The eight tables of a grid: (cos, sin) of the latitude of the row centres -1 .. nrow (index row + 1), of the
    longitude of the column centres -1 .. ncol (index col + 1), of the latitude of the row edges 0 .. nrow and of the
    longitude of the column edges 0 .. ncol.  Edge r is the top of row r, edge c the left of column c."""
    cos_latitude_of_row = np.empty(nrow + 2, dtype=np.float64)
    sin_latitude_of_row = np.empty(nrow + 2, dtype=np.float64)
    for row in range(-1, nrow + 1):
        latitude = north + (float(row) + 0.5) * y_resolution
        sine, cosine = sincos(latitude * _TO_RADIANS)
        cos_latitude_of_row[row + 1] = cosine
        sin_latitude_of_row[row + 1] = sine
    cos_longitude_of_col = np.empty(ncol + 2, dtype=np.float64)
    sin_longitude_of_col = np.empty(ncol + 2, dtype=np.float64)
    for col in range(-1, ncol + 1):
        longitude = west + (float(col) + 0.5) * x_resolution
        sine, cosine = sincos(longitude * _TO_RADIANS)
        cos_longitude_of_col[col + 1] = cosine
        sin_longitude_of_col[col + 1] = sine
    cos_latitude_of_row_edge = np.empty(nrow + 1, dtype=np.float64)
    sin_latitude_of_row_edge = np.empty(nrow + 1, dtype=np.float64)
    for row_edge in range(nrow + 1):
        latitude = north + float(row_edge) * y_resolution
        sine, cosine = sincos(latitude * _TO_RADIANS)
        cos_latitude_of_row_edge[row_edge] = cosine
        sin_latitude_of_row_edge[row_edge] = sine
    cos_longitude_of_col_edge = np.empty(ncol + 1, dtype=np.float64)
    sin_longitude_of_col_edge = np.empty(ncol + 1, dtype=np.float64)
    for col_edge in range(ncol + 1):
        longitude = west + float(col_edge) * x_resolution
        sine, cosine = sincos(longitude * _TO_RADIANS)
        cos_longitude_of_col_edge[col_edge] = cosine
        sin_longitude_of_col_edge[col_edge] = sine
    return (cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col,
            cos_latitude_of_row_edge, sin_latitude_of_row_edge, cos_longitude_of_col_edge, sin_longitude_of_col_edge)


@njit(inline="always")
def centre_vector(row, col, cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col):
    """The unit vector of the centre of pixel (row, col)."""
    cos_latitude = cos_latitude_of_row[row + 1]
    return (cos_latitude * cos_longitude_of_col[col + 1], cos_latitude * sin_longitude_of_col[col + 1],
            sin_latitude_of_row[row + 1])


@njit(inline="always")
def _dot(a0, a1, a2, b0, b1, b2):
    return a0 * b0 + a1 * b1 + a2 * b2


@njit(inline="always")
def _cross(a0, a1, a2, b0, b1, b2):
    return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0


@njit(inline="always")
def _piece_count_of_angle(angle):
    pieces = np.int64(math.ceil(angle * RADIUS_FOR_PIECES_M / PIECE_M))
    return pieces if pieces >= 1 else np.int64(1)


# ---- L_b ----

@njit
def great_circle_length_m(first_latitude, first_longitude, f0, f1, f2, second_latitude, second_longitude, s0, s1, s2,
                          model):
    """The length on the earth model of the great circle between two points given by their latitudes and longitudes
    (degrees) and their unit vectors, in equal pieces of at most 5 km."""
    c0, c1, c2 = _cross(f0, f1, f2, s0, s1, s2)
    angle = math.atan2(math.sqrt(_dot(c0, c1, c2, c0, c1, c2)), _dot(f0, f1, f2, s0, s1, s2))
    piece_count = _piece_count_of_angle(angle)
    if piece_count == 1:
        return distance_m(first_latitude, first_longitude, second_latitude, second_longitude, model)
    sine_of_angle = math.sin(angle)
    length_m = 0.0
    previous_latitude = first_latitude
    previous_longitude = first_longitude
    for piece in range(1, piece_count + 1):
        latitude = second_latitude
        longitude = second_longitude
        if piece < piece_count:
            fraction = float(piece) / float(piece_count)
            weight_first = math.sin((1.0 - fraction) * angle) / sine_of_angle
            weight_second = math.sin(fraction * angle) / sine_of_angle
            x = weight_first * f0 + weight_second * s0
            y = weight_first * f1 + weight_second * s1
            z = weight_first * f2 + weight_second * s2
            latitude = math.atan2(z, math.sqrt(x * x + y * y)) * _TO_DEGREES
            # next to the first point's longitude (a grid's own longitudes can pass 180)
            longitude = first_longitude + remainder(math.atan2(y, x) * _TO_DEGREES - first_longitude, 360.0)
        length_m += distance_m(previous_latitude, previous_longitude, latitude, longitude, model)
        previous_latitude = latitude
        previous_longitude = longitude
    return length_m


@njit
def basin_length(cells, size, target_cell, ncol, west, x_resolution, north, y_resolution, cos_latitude_of_row,
                 sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col, model):
    """(L_b in km, the farthest boundary pixel's row and column); (no data, -1, -1) without a boundary pixel."""
    if size < 1:
        return NODATA, -1, -1
    target_row = target_cell // ncol
    target_col = target_cell % ncol
    t0, t1, t2 = centre_vector(target_row, target_col, cos_latitude_of_row, sin_latitude_of_row,
                               cos_longitude_of_col, sin_longitude_of_col)
    best_cosine = 2.0
    best_row = _INT32_MAX
    best_col = _INT32_MAX
    for index in range(size):
        cell = cells[index]
        row = cell // ncol
        col = cell % ncol
        p0, p1, p2 = centre_vector(row, col, cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col,
                                   sin_longitude_of_col)
        cosine = _dot(p0, p1, p2, t0, t1, t2)
        # farther: a smaller cosine, or the same cosine and the pixel first in row-major order
        if cosine != best_cosine:
            farther = cosine < best_cosine
        else:
            farther = row < best_row or (row == best_row and col < best_col)
        if farther:
            best_cosine = cosine
            best_row = row
            best_col = col
    f0, f1, f2 = centre_vector(best_row, best_col, cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col,
                               sin_longitude_of_col)
    length_m = great_circle_length_m(north + (float(target_row) + 0.5) * y_resolution,
                                     west + (float(target_col) + 0.5) * x_resolution, t0, t1, t2,
                                     north + (float(best_row) + 0.5) * y_resolution,
                                     west + (float(best_col) + 0.5) * x_resolution, f0, f1, f2, model)
    return length_m / 1000.0, best_row, best_col


# ---- W_b ----

@njit
def _leg_to_axis_m(row, col, pole0, pole1, pole2, west, x_resolution, north, y_resolution, cos_latitude_of_row,
                   sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col, model):
    """The length from the centre of (row, col) to the foot of its perpendicular on the axis of the pole: the great
    circle through the centre and the pole meets the axis there, at the centre less its part along the pole."""
    c0, c1, c2 = centre_vector(row, col, cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col,
                               sin_longitude_of_col)
    sine = _dot(c0, c1, c2, pole0, pole1, pole2)
    foot0 = c0 - sine * pole0
    foot1 = c1 - sine * pole1
    foot2 = c2 - sine * pole2
    foot_length = math.sqrt(_dot(foot0, foot1, foot2, foot0, foot1, foot2))
    if not (foot_length > 0.0):
        return 0.0                         # the centre at the pole itself: no catchment reaches 90 degrees across
    foot0 /= foot_length
    foot1 /= foot_length
    foot2 /= foot_length
    centre_latitude = north + (float(row) + 0.5) * y_resolution
    centre_longitude = west + (float(col) + 0.5) * x_resolution
    foot_latitude = math.atan2(foot2, math.sqrt(foot0 * foot0 + foot1 * foot1)) * _TO_DEGREES
    foot_longitude = centre_longitude + remainder(math.atan2(foot1, foot0) * _TO_DEGREES - centre_longitude, 360.0)
    return great_circle_length_m(centre_latitude, centre_longitude, c0, c1, c2, foot_latitude, foot_longitude,
                                 foot0, foot1, foot2, model)


@njit
def basin_width(cells, size, target_cell, farthest_row, farthest_col, ncol, west, x_resolution, north, y_resolution,
                cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col, model):
    """W_b in km: 0 with fewer than two boundary pixels, without a farthest pixel, or when the farthest pixel is
    the target itself."""
    if size < 2 or farthest_row < 0:
        return 0.0
    t0, t1, t2 = centre_vector(target_cell // ncol, target_cell % ncol, cos_latitude_of_row, sin_latitude_of_row,
                               cos_longitude_of_col, sin_longitude_of_col)
    f0, f1, f2 = centre_vector(farthest_row, farthest_col, cos_latitude_of_row, sin_latitude_of_row,
                               cos_longitude_of_col, sin_longitude_of_col)
    # the pole of the axis target -> farthest: a unit vector at right angles to both
    pole0, pole1, pole2 = _cross(t0, t1, t2, f0, f1, f2)
    pole_length = math.sqrt(_dot(pole0, pole1, pole2, pole0, pole1, pole2))
    if not (pole_length > 0.0):
        return 0.0
    pole0 /= pole_length
    pole1 /= pole_length
    pole2 /= pole_length
    largest_sine = 0.0
    largest_row = -1
    largest_col = -1
    smallest_sine = 0.0
    smallest_row = -1
    smallest_col = -1
    for index in range(size):
        cell = cells[index]
        row = cell // ncol
        col = cell % ncol
        p0, p1, p2 = centre_vector(row, col, cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col,
                                   sin_longitude_of_col)
        sine = _dot(p0, p1, p2, pole0, pole1, pole2)
        if sine > 0.0 and (sine > largest_sine or largest_row < 0
                           or (sine == largest_sine
                               and (row < largest_row or (row == largest_row and col < largest_col)))):
            largest_sine = sine
            largest_row = row
            largest_col = col
        elif sine < 0.0 and (sine < smallest_sine or smallest_row < 0
                             or (sine == smallest_sine
                                 and (row < smallest_row or (row == smallest_row and col < smallest_col)))):
            smallest_sine = sine
            smallest_row = row
            smallest_col = col
    width_m = 0.0
    if largest_row >= 0:
        width_m += _leg_to_axis_m(largest_row, largest_col, pole0, pole1, pole2, west, x_resolution, north,
                                  y_resolution, cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col,
                                  sin_longitude_of_col, model)
    if smallest_row >= 0:
        width_m += _leg_to_axis_m(smallest_row, smallest_col, pole0, pole1, pole2, west, x_resolution, north,
                                  y_resolution, cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col,
                                  sin_longitude_of_col, model)
    return width_m / 1000.0


# ---- the convex hull ----

@njit(inline="always")
def _two_sum(first, second):
    total = first + second
    second_virtual = total - first
    first_virtual = total - second_virtual
    return total, (first - first_virtual) + (second - second_virtual)


@njit
def _orientation_sign_exact(ox, oy, fx, fy, sx, sy):
    """The sign of the orientation of (origin, first, second) in the chart, exact for the given doubles (Shewchuk
    1997): the sum of the six products of the coordinates, each an exact pair (fma), added up as an expansion of
    non-overlapping doubles, whose sign is that of its largest part."""
    factor_a = (ox, -ox, fx, -fx, sx, -sx)
    factor_b = (fy, sy, sy, oy, oy, fy)
    expansion = np.zeros(32, dtype=np.float64)
    expansion_count = 0
    for product_index in range(6):
        product = factor_a[product_index] * factor_b[product_index]
        product_error = fma(factor_a[product_index], factor_b[product_index], -product)
        for part_index in range(2):
            carry = product_error if part_index == 0 else product
            kept = 0
            for index in range(expansion_count):
                total, error = _two_sum(carry, expansion[index])
                if error != 0.0:
                    expansion[kept] = error
                    kept += 1
                carry = total
            if carry != 0.0:
                expansion[kept] = carry
                kept += 1
            expansion_count = kept
    if expansion_count == 0:
        return 0
    return 1 if expansion[expansion_count - 1] > 0.0 else -1


@njit(inline="always")
def _chart_orientation_sign(ox, oy, fx, fy, sx, sy):
    """> 0 anticlockwise, < 0 clockwise, 0 in line: the determinant in doubles when it is clear of its rounding
    bound, the exact sign otherwise (a near-collinear triple could otherwise lose a corner of the hull)."""
    left = (fx - ox) * (sy - oy)
    right = (fy - oy) * (sx - ox)
    determinant = left - right
    bound = 3.3306690738754716e-16 * (abs(left) + abs(right))
    if determinant > bound:
        return 1
    if -determinant > bound:
        return -1
    return _orientation_sign_exact(ox, oy, fx, fy, sx, sy)


@njit
def _corner_polygon_area_km2(polygon_rows, polygon_cols, polygon_count, west, x_resolution, north, y_resolution,
                             cos_latitude_of_row_edge, sin_latitude_of_row_edge, cos_longitude_of_col_edge,
                             sin_longitude_of_col_edge, model):
    """The area, km^2, on the earth model of a polygon of corners (row edges and column edges) whose sides are great
    circles; -1 when the pieces cannot be held, -2 for a ring round a pole."""
    if polygon_count < 3:
        return 0.0
    point_total = 0
    for corner_index in range(polygon_count):
        following = (corner_index + 1) % polygon_count
        from_cos = cos_latitude_of_row_edge[polygon_rows[corner_index]]
        f0 = from_cos * cos_longitude_of_col_edge[polygon_cols[corner_index]]
        f1 = from_cos * sin_longitude_of_col_edge[polygon_cols[corner_index]]
        f2 = sin_latitude_of_row_edge[polygon_rows[corner_index]]
        to_cos = cos_latitude_of_row_edge[polygon_rows[following]]
        t0 = to_cos * cos_longitude_of_col_edge[polygon_cols[following]]
        t1 = to_cos * sin_longitude_of_col_edge[polygon_cols[following]]
        t2 = sin_latitude_of_row_edge[polygon_rows[following]]
        c0, c1, c2 = _cross(f0, f1, f2, t0, t1, t2)
        angle = math.atan2(math.sqrt(_dot(c0, c1, c2, c0, c1, c2)), _dot(f0, f1, f2, t0, t1, t2))
        point_total += _piece_count_of_angle(angle)
    if point_total > _INT32_MAX:
        return -1.0
    longitudes = np.empty(point_total, dtype=np.float64)
    latitudes = np.empty(point_total, dtype=np.float64)
    point_index = 0
    for corner_index in range(polygon_count):
        following = (corner_index + 1) % polygon_count
        from_cos = cos_latitude_of_row_edge[polygon_rows[corner_index]]
        f0 = from_cos * cos_longitude_of_col_edge[polygon_cols[corner_index]]
        f1 = from_cos * sin_longitude_of_col_edge[polygon_cols[corner_index]]
        f2 = sin_latitude_of_row_edge[polygon_rows[corner_index]]
        to_cos = cos_latitude_of_row_edge[polygon_rows[following]]
        t0 = to_cos * cos_longitude_of_col_edge[polygon_cols[following]]
        t1 = to_cos * sin_longitude_of_col_edge[polygon_cols[following]]
        t2 = sin_latitude_of_row_edge[polygon_rows[following]]
        c0, c1, c2 = _cross(f0, f1, f2, t0, t1, t2)
        angle = math.atan2(math.sqrt(_dot(c0, c1, c2, c0, c1, c2)), _dot(f0, f1, f2, t0, t1, t2))
        pieces = _piece_count_of_angle(angle)
        # the corner itself, from the grid's lines
        from_latitude = north + float(polygon_rows[corner_index]) * y_resolution
        from_longitude = west + float(polygon_cols[corner_index]) * x_resolution
        latitudes[point_index] = from_latitude
        longitudes[point_index] = from_longitude
        point_index += 1
        sine_of_angle = math.sin(angle)
        for piece in range(1, pieces):
            fraction = float(piece) / float(pieces)
            weight_from = math.sin((1.0 - fraction) * angle) / sine_of_angle
            weight_to = math.sin(fraction * angle) / sine_of_angle
            x = weight_from * f0 + weight_to * t0
            y = weight_from * f1 + weight_to * t1
            z = weight_from * f2 + weight_to * t2
            latitudes[point_index] = math.atan2(z, math.sqrt(x * x + y * y)) * _TO_DEGREES
            longitudes[point_index] = from_longitude + remainder(math.atan2(y, x) * _TO_DEGREES - from_longitude,
                                                                 360.0)
            point_index += 1
    # a ring round a pole turns its longitudes by a whole turn, and the zonal sum then gives another area: no
    # catchment encloses a pole, so such a hull is refused rather than measured wrong
    winding_deg = 0.0
    for index in range(point_index):
        winding_deg += remainder(longitudes[(index + 1) % point_index] - longitudes[index], 360.0)
    if abs(winding_deg) > 180.0:
        return -2.0
    return lonlat_polygon_area_m2(longitudes, latitudes, model) / 1.0e6


@njit
def convexity(cells, size, target_cell, upstream_area_km2, ncol, west, x_resolution, north, y_resolution,
              cos_latitude_of_row, sin_latitude_of_row, cos_longitude_of_col, sin_longitude_of_col,
              cos_latitude_of_row_edge, sin_latitude_of_row_edge, cos_longitude_of_col_edge,
              sin_longitude_of_col_edge, model):
    """(status, convexity): A over the area of the convex hull on the sphere of the catchment's pixels, capped at 1;
    no data without a boundary pixel or when the hull has no area.  status is HULL_OK or one of the refusals."""
    if size < 1:
        return HULL_OK, NODATA
    first_col = _INT32_MAX
    last_col = -1
    for index in range(size):
        col = cells[index] % ncol
        if col < first_col:
            first_col = col
        if col > last_col:
            last_col = col
    column_count = last_col - first_col + 1
    # the topmost and the bottommost boundary pixel of every column: the topmost and the bottommost pixel of the
    # catchment there (a pixel with nothing of the catchment above it is on the boundary)
    top_row_of_column = np.full(column_count, _INT32_MAX, dtype=np.int64)
    bottom_row_of_column = np.full(column_count, -1, dtype=np.int64)
    for index in range(size):
        cell = cells[index]
        row = cell // ncol
        column_index = cell % ncol - first_col
        if row < top_row_of_column[column_index]:
            top_row_of_column[column_index] = row
        if row > bottom_row_of_column[column_index]:
            bottom_row_of_column[column_index] = row
    # at every column edge, the topmost and the bottommost corner of the pixels on either side
    corner_rows = np.empty(2 * (column_count + 1), dtype=np.int64)
    corner_cols = np.empty(2 * (column_count + 1), dtype=np.int64)
    corner_count = 0
    for edge_index in range(column_count + 1):
        top_row = _INT32_MAX
        bottom_row = -1
        for side in range(2):
            column_index = edge_index - 1 + side
            if column_index < 0 or column_index >= column_count or bottom_row_of_column[column_index] < 0:
                continue
            if top_row_of_column[column_index] < top_row:
                top_row = top_row_of_column[column_index]
            if bottom_row_of_column[column_index] > bottom_row:
                bottom_row = bottom_row_of_column[column_index]
        if bottom_row < 0:
            continue
        corner_rows[corner_count] = top_row                    # the top edge of the topmost pixel
        corner_cols[corner_count] = first_col + edge_index
        corner_count += 1
        corner_rows[corner_count] = bottom_row + 1             # the bottom edge of the bottommost pixel
        corner_cols[corner_count] = first_col + edge_index
        corner_count += 1
    # the gnomonic chart about the target: east = (-y, x, 0) / |(x, y)|, north = centre x east
    k0, k1, k2 = centre_vector(target_cell // ncol, target_cell % ncol, cos_latitude_of_row, sin_latitude_of_row,
                               cos_longitude_of_col, sin_longitude_of_col)
    horizontal = math.sqrt(k0 * k0 + k1 * k1)
    if horizontal > 0.0:
        east0 = -k1 / horizontal
        east1 = k0 / horizontal
    else:
        east0 = 0.0
        east1 = 1.0
    east2 = 0.0
    north0, north1, north2 = _cross(k0, k1, k2, east0, east1, east2)
    corner_x = np.empty(corner_count, dtype=np.float64)
    corner_y = np.empty(corner_count, dtype=np.float64)
    for corner_index in range(corner_count):
        cos_latitude = cos_latitude_of_row_edge[corner_rows[corner_index]]
        v0 = cos_latitude * cos_longitude_of_col_edge[corner_cols[corner_index]]
        v1 = cos_latitude * sin_longitude_of_col_edge[corner_cols[corner_index]]
        v2 = sin_latitude_of_row_edge[corner_rows[corner_index]]
        towards_centre = _dot(v0, v1, v2, k0, k1, k2)
        if not (towards_centre >= CHART_MIN_COSINE):
            return HULL_BEYOND_CHART, NODATA
        corner_x[corner_index] = _dot(v0, v1, v2, east0, east1, east2) / towards_centre
        corner_y[corner_index] = _dot(v0, v1, v2, north0, north1, north2) / towards_centre
    # the order of the monotone chain: x, then y, then the row edge, then the column edge
    order = np.argsort(corner_cols[:corner_count], kind="mergesort")
    order = order[np.argsort(corner_rows[:corner_count][order], kind="mergesort")]
    order = order[np.argsort(corner_y[order], kind="mergesort")]
    order = order[np.argsort(corner_x[order], kind="mergesort")]
    sorted_x = np.empty(corner_count, dtype=np.float64)
    sorted_y = np.empty(corner_count, dtype=np.float64)
    sorted_rows = np.empty(corner_count, dtype=np.int64)
    sorted_cols = np.empty(corner_count, dtype=np.int64)
    distinct_count = 0
    for position in range(corner_count):
        corner_index = order[position]
        if distinct_count > 0 and sorted_x[distinct_count - 1] == corner_x[corner_index] \
                and sorted_y[distinct_count - 1] == corner_y[corner_index] \
                and sorted_rows[distinct_count - 1] == corner_rows[corner_index] \
                and sorted_cols[distinct_count - 1] == corner_cols[corner_index]:
            continue
        sorted_x[distinct_count] = corner_x[corner_index]
        sorted_y[distinct_count] = corner_y[corner_index]
        sorted_rows[distinct_count] = corner_rows[corner_index]
        sorted_cols[distinct_count] = corner_cols[corner_index]
        distinct_count += 1
    if distinct_count < 3:
        return HULL_OK, NODATA
    # Andrew's monotone chain: the lower chain left to right, the upper chain right to left; points on a side dropped
    lower = np.empty(distinct_count, dtype=np.int64)
    lower_count = 0
    for vertex in range(distinct_count):
        while lower_count >= 2 and _chart_orientation_sign(
                sorted_x[lower[lower_count - 2]], sorted_y[lower[lower_count - 2]],
                sorted_x[lower[lower_count - 1]], sorted_y[lower[lower_count - 1]],
                sorted_x[vertex], sorted_y[vertex]) <= 0:
            lower_count -= 1
        lower[lower_count] = vertex
        lower_count += 1
    upper = np.empty(distinct_count, dtype=np.int64)
    upper_count = 0
    for vertex in range(distinct_count - 1, -1, -1):
        while upper_count >= 2 and _chart_orientation_sign(
                sorted_x[upper[upper_count - 2]], sorted_y[upper[upper_count - 2]],
                sorted_x[upper[upper_count - 1]], sorted_y[upper[upper_count - 1]],
                sorted_x[vertex], sorted_y[vertex]) <= 0:
            upper_count -= 1
        upper[upper_count] = vertex
        upper_count += 1
    polygon_count = (lower_count - 1) + (upper_count - 1)
    if polygon_count < 3:
        return HULL_OK, NODATA
    polygon_rows = np.empty(polygon_count, dtype=np.int64)
    polygon_cols = np.empty(polygon_count, dtype=np.int64)
    position = 0
    for index in range(lower_count - 1):
        polygon_rows[position] = sorted_rows[lower[index]]
        polygon_cols[position] = sorted_cols[lower[index]]
        position += 1
    for index in range(upper_count - 1):
        polygon_rows[position] = sorted_rows[upper[index]]
        polygon_cols[position] = sorted_cols[upper[index]]
        position += 1
    hull_km2 = _corner_polygon_area_km2(polygon_rows, polygon_cols, polygon_count, west, x_resolution, north,
                                        y_resolution, cos_latitude_of_row_edge, sin_latitude_of_row_edge,
                                        cos_longitude_of_col_edge, sin_longitude_of_col_edge, model)
    if hull_km2 == -2.0:
        return HULL_ROUND_A_POLE, NODATA
    if hull_km2 < 0.0:
        return HULL_TOO_MANY_PIECES, NODATA
    if hull_km2 > 0.0 and upstream_area_km2 > 0.0:
        ratio = np.float64(upstream_area_km2) / hull_km2
        return HULL_OK, (1.0 if ratio > 1.0 else ratio)
    return HULL_OK, NODATA


__all__ = ["grid_tables", "centre_vector", "great_circle_length_m", "basin_length", "basin_width", "convexity",
           "HULL_OK", "HULL_BEYOND_CHART", "HULL_ROUND_A_POLE", "HULL_TOO_MANY_PIECES"]
