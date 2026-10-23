"""The transform of a grid, taken as the C code takes it.

A GeoTIFF header carries the pixel size and the origin in decimal, and the last bits of those numbers are noise
(1/1200 written as 0.00083333, say).  The C code puts them back before it uses them, and so does this package, so
that both place every pixel centre at the same double:

1. the pixel size goes back on 1/N, N a divisor of MERIT Hydro's 1200 or HydroSHEDS' 3600 pixels per degree, when it
   is within 1e-5 of it (relative);
2. the origin goes back on the grid of half pixels of that size, when it is within 1e-4 of a half pixel of it;
3. every term is rounded to 1e-12 of a degree.

A grid of another resolution keeps its size; its origin still goes back on its own grid of half pixels when it is
within 1e-4 of a half pixel of it (step 2), and step 3 applies.  The centre of pixel (row, col) is
then (transform[0] + (col + 0.5) transform[1], transform[3] + (row + 0.5) transform[5]).  :func:`pixel_edges_deg`
gives the size of step 1 without the rounding of step 3: the edges of a pixel as the perimeter ([7],
:mod:`flowmorph.vscells`) takes them, as in FlowMorph 1.0.

A file is written with :func:`exact_transform`, steps 1 and 2 without step 3: the rounding to 1e-12 leaves 1/1200 as
0.000833333333, and a grid of MERIT Hydro written with it would carry that size in its header.
"""

import math

_LARGEST_ROUNDING_IN_A_HEADER_RELATIVE = 1.0e-5
_BASE_PIXELS_PER_DEGREE = (1200, 3600)


def _pixel_size_on_the_exact_fraction(pixel_size):
    if not math.isfinite(pixel_size) or pixel_size == 0.0:
        return pixel_size
    sign = -1.0 if pixel_size < 0.0 else 1.0
    magnitude = abs(pixel_size)
    pixels_per_degree = 1.0 / magnitude
    if not pixels_per_degree >= 1.0 or pixels_per_degree > 1.0e15:
        return pixel_size
    for base in _BASE_PIXELS_PER_DEGREE:
        divisor_factor = math.floor(base / pixels_per_degree + 0.5)
        if divisor_factor >= 1 and base % divisor_factor == 0:
            candidate = base // divisor_factor
            if candidate < 1:
                continue
            candidate_size = 1.0 / candidate
            if abs(magnitude - candidate_size) <= _LARGEST_ROUNDING_IN_A_HEADER_RELATIVE * magnitude:
                return sign * candidate_size
    return pixel_size


def _origin_on_the_exact_grid(origin, pixel_size):
    if not math.isfinite(origin) or not math.isfinite(pixel_size) or pixel_size == 0.0:
        return origin
    half_pixel = abs(pixel_size) / 2.0
    half_pixels_from_zero = origin / half_pixel
    if not math.isfinite(half_pixels_from_zero):
        return origin
    half_pixels_rounded = float(math.floor(half_pixels_from_zero + 0.5))
    if abs(half_pixels_from_zero - half_pixels_rounded) > 1.0e-4:
        return origin
    return half_pixels_rounded * half_pixel


def snap_transform(transform):
    """The six terms of a transform (GDAL order) as the C code uses them."""
    terms = [float(value) for value in transform]
    terms[1] = _pixel_size_on_the_exact_fraction(terms[1])
    terms[5] = _pixel_size_on_the_exact_fraction(terms[5])
    terms[0] = _origin_on_the_exact_grid(terms[0], terms[1])
    terms[3] = _origin_on_the_exact_grid(terms[3], terms[5])
    return tuple(float(math.floor(value * 1.0e12 + 0.5)) / 1.0e12 for value in terms)


def exact_transform(transform):
    """The six terms of a transform on the exact grid, for the header of a file: the pixel size back on 1/N and the
    origin back on the grid of half pixels (steps 1 and 2 of :func:`snap_transform`), not rounded to 1e-12, so that
    a grid of MERIT Hydro is written with 1/1200 to the last bit.  On a grid of 1/N of a degree, N a divisor of 1200
    or 3600, whose origin lies on its grid of half pixels (MERIT Hydro's and HydroSHEDS' do), it undoes the rounding
    of a transform :func:`snap_transform` gave: the pixel size is 1/N exactly, and the origin a whole number of half
    pixels, within a few units in the last place of the value it came from."""
    terms = [float(value) for value in transform]
    terms[1] = _pixel_size_on_the_exact_fraction(terms[1])
    terms[5] = _pixel_size_on_the_exact_fraction(terms[5])
    terms[0] = _origin_on_the_exact_grid(terms[0], terms[1])
    terms[3] = _origin_on_the_exact_grid(terms[3], terms[5])
    return tuple(terms)


def north_up(transform):
    """The six terms of a transform as the C code uses them (:func:`snap_transform`), for the entries that take a
    transform from the caller: six finite numbers, no rotation or shear (the third and fifth terms 0), a positive
    width and a negative height that the rounding to 1e-12 of a degree does not take to 0.  ValueError otherwise:
    the C code's drivers refuse such a grid, and the arithmetic of this package would take it for a north-up one."""
    terms = [float(value) for value in transform]
    if len(terms) != 6 or not all(math.isfinite(value) for value in terms):
        raise ValueError("the transform must be six finite numbers")
    if terms[2] != 0.0 or terms[4] != 0.0 or not terms[1] > 0.0 or not terms[5] < 0.0:
        raise ValueError("the grid must be north up with no rotation")
    snapped = snap_transform(terms)
    if not (snapped[1] > 0.0 and snapped[5] < 0.0):
        raise ValueError("the grid must be north up, with pixels of at least 1e-12 of a degree")
    return snapped


def pixel_edges_deg(transform):
    """The width and height of a pixel in degrees for the lengths of its edges: the size put back on 1/N (step 1
    above) and not rounded, so 1/1200 exactly on MERIT Hydro's grid, as the perimeter of FlowMorph 1.0 took it."""
    return (abs(_pixel_size_on_the_exact_fraction(float(transform[1]))),
            abs(_pixel_size_on_the_exact_fraction(float(transform[5]))))


__all__ = ["snap_transform", "exact_transform", "north_up", "pixel_edges_deg"]
