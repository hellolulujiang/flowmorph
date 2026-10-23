"""Lengths and areas on the earth, as the C code of FlowMorph takes them.

Two models.  ``"wgs84"`` (the default of FlowMorph 1.1) is the WGS84 ellipsoid: the metres of one degree from its
meridian and prime-vertical radii, a great-circle distance on the authalic sphere scaled to the ellipsoid along the
bearing at the mid-latitude, and polygon areas from the ellipsoid's zone function.  ``"sphere"`` is the authalic
sphere of radius 6 371 007.181 m, on which the FullBasin v1.0 layers were measured.

The slope of :mod:`flowmorph.slope` does not depend on the model: it takes the metres of one degree from the fixed
series :func:`degree_metres_x` and :func:`degree_metres_y`, as the C code does.

The functions the kernels call are compiled with numba, and those that depend on the model take it as an integer
(:data:`WGS84` or :data:`SPHERE`); :func:`model_code` turns the name into the integer.
"""

import math

import numpy as np
from numba import njit

from .clib import sincos

WGS84 = 0
SPHERE = 1
EARTH_MODELS = ("wgs84", "sphere")

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
R_AUTHALIC = 6371007.181

_GAUSS_LEGENDRE_NODES = np.array([
    -0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
    0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363])
_GAUSS_LEGENDRE_WEIGHTS = np.array([
    0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
    0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763])


def model_code(earth):
    """The integer the compiled functions take for a model name."""
    if earth not in EARTH_MODELS:
        raise ValueError(f"unknown earth model {earth!r}; use one of {EARTH_MODELS}")
    return WGS84 if earth == "wgs84" else SPHERE


@njit
def degree_metres_x(latitude_deg):
    """Metres of one degree of longitude at a latitude: the fixed three-term series of the slope."""
    radlat = latitude_deg * (math.pi / 180.0)
    return 111412.84 * math.cos(radlat) + (-93.5) * math.cos(3.0 * radlat) + 0.118 * math.cos(5.0 * radlat)


@njit
def degree_metres_y(latitude_deg):
    """Metres of one degree of latitude at a latitude: the fixed four-term series of the slope."""
    radlat = latitude_deg * (math.pi / 180.0)
    return (111132.92 + (-559.82) * math.cos(2.0 * radlat)
            + 1.175 * math.cos(4.0 * radlat)
            + (-0.0023) * math.cos(6.0 * radlat))


@njit
def metres_per_degree(latitude_deg, model):
    """The metres of one degree of latitude and of longitude at a latitude, on the model."""
    to_rad = math.pi / 180.0
    phi = latitude_deg * to_rad
    if model == SPHERE:
        return R_AUTHALIC * to_rad, R_AUTHALIC * to_rad * math.cos(phi)
    sin_phi, cos_phi = sincos(phi)                      # one call for the two, as the C binary on macOS takes them
    s2 = sin_phi * sin_phi
    w = math.sqrt(1.0 - WGS84_E2 * s2)
    meridian_radius = WGS84_A * (1.0 - WGS84_E2) / (w * w * w)
    prime_vertical_radius = WGS84_A / w
    return meridian_radius * to_rad, prime_vertical_radius * cos_phi * to_rad


@njit
def shortest_longitude_difference(longitude_from, longitude_to):
    """lon_to - lon_from taken the short way round: one turn is added or taken off when the difference passes
    180 degrees either way, so two longitudes less than a turn apart give a difference in [-180, 180]."""
    difference = longitude_to - longitude_from
    if not math.isfinite(difference):
        return difference
    if difference > 180.0:
        return difference - 360.0
    if difference < -180.0:
        return difference + 360.0
    return difference


@njit
def distance_with_cosines_m(latitude_1, longitude_1, latitude_2, longitude_2, cos_latitude_1, cos_latitude_2, model):
    """:func:`distance_m` with the cosines of the two latitudes given (a grid keeps them in a table, one a row)."""
    to_rad = math.pi / 180.0
    dlat = (latitude_2 - latitude_1) * to_rad
    dlon = shortest_longitude_difference(longitude_1, longitude_2) * to_rad
    la1 = latitude_1 * to_rad
    la2 = latitude_2 * to_rad
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0)
         + cos_latitude_1 * cos_latitude_2 * math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    # rounding must not carry a above 1 (a point and the point opposite it: one unit in the last place above 1 at
    # one latitude in 25).  Below 0 a cannot come by rounding (both terms are squares times cosines of latitudes,
    # none negative): a cosine of a latitude beyond a pole was given, and the distance is then not a number, as in
    # the C code
    if a > 1.0:
        a = 1.0
    d_sphere = R_AUTHALIC * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    if model == SPHERE or d_sphere <= 0.0:
        return d_sphere
    phim = 0.5 * (la1 + la2)
    # the sine and the cosine of one angle together, as the C code's compiler takes them (clib.sincos): taken apart
    # the sine differs in its last bit at some latitudes (26 degrees), and this length is under every plan-form
    # length (metres_per_degree takes them together)
    sin_phim, cos_phim = sincos(phim)
    s2 = sin_phim * sin_phim
    w = math.sqrt(1.0 - WGS84_E2 * s2)
    meridian_radius = WGS84_A * (1.0 - WGS84_E2) / (w * w * w)
    prime_vertical_radius = WGS84_A / w
    ex = prime_vertical_radius * cos_phim * dlon
    ey = meridian_radius * dlat
    sx = R_AUTHALIC * cos_phim * dlon
    sy = R_AUTHALIC * dlat
    num = math.sqrt(ex * ex + ey * ey)
    den = math.sqrt(sx * sx + sy * sy)
    return d_sphere * (num / den) if den > 0.0 else d_sphere


@njit
def distance_m(latitude_1, longitude_1, latitude_2, longitude_2, model):
    """Distance between two points in metres: the great circle on the authalic sphere, and on the ellipsoid that
    length scaled by the ratio of the ellipsoidal to the spherical arc along the bearing at the mid-latitude.

    Taken in one piece it is exact for a line of a few kilometres and drifts from the geodesic on a long one (by
    2e-9 at 10 km, 5e-8 at 50 km, 2.5e-4 at 3,700 km), so the lengths of :mod:`flowmorph.sphere` add it up in
    pieces of at most 5 km along the great circle."""
    to_rad = math.pi / 180.0
    return distance_with_cosines_m(latitude_1, longitude_1, latitude_2, longitude_2,
                                   math.cos(latitude_1 * to_rad), math.cos(latitude_2 * to_rad), model)


@njit
def _wgs84_zone_function(latitude_rad):
    """F(lat) of the zone area between the equator and a latitude: a^2 (1 - e^2) F / 2 per radian of longitude."""
    e = math.sqrt(WGS84_E2)
    sl = math.sin(latitude_rad)
    return sl / (1.0 - WGS84_E2 * sl * sl) + (1.0 / (2.0 * e)) * math.log((1.0 + e * sl) / (1.0 - e * sl))


@njit
def lonlat_polygon_area_m2(longitudes, latitudes, model):
    """Area in m^2 of a polygon whose edges are straight in longitude and latitude (a ring, not closed).

    Green's theorem: the sum over the edges of the longitude step times the mean, along the edge, of the zone area
    from the equator.  On the sphere that mean is exact in closed form; on the ellipsoid an eight-point
    Gauss-Legendre rule, exact to rounding for an edge of any basin.  A rectangle of pixels comes out as the sum of
    their ellipsoidal areas, so a hull that holds pixels is not smaller than they are.
    """
    n = longitudes.size
    if n < 3:
        return 0.0
    deg2rad = math.pi / 180.0
    ellipsoid_zone_factor = WGS84_A * WGS84_A * (1.0 - WGS84_E2) * 0.5
    total = 0.0
    for vertex in range(n):
        following = (vertex + 1) % n
        longitude_step_rad = shortest_longitude_difference(longitudes[vertex], longitudes[following]) * deg2rad
        latitude_from_rad = latitudes[vertex] * deg2rad
        latitude_to_rad = latitudes[following] * deg2rad
        middle_latitude_rad = 0.5 * (latitude_from_rad + latitude_to_rad)
        half_span_rad = 0.5 * (latitude_to_rad - latitude_from_rad)
        mean_zone_area = 0.0
        if model == SPHERE:
            if abs(half_span_rad) < 1.0e-8:
                sin_ratio = 1.0 - half_span_rad * half_span_rad / 6.0
            else:
                sin_ratio = math.sin(half_span_rad) / half_span_rad
            mean_zone_area = R_AUTHALIC * R_AUTHALIC * math.sin(middle_latitude_rad) * sin_ratio
        else:
            for node in range(8):
                node_latitude_rad = middle_latitude_rad + half_span_rad * _GAUSS_LEGENDRE_NODES[node]
                mean_zone_area += (0.5 * _GAUSS_LEGENDRE_WEIGHTS[node] * ellipsoid_zone_factor
                                   * _wgs84_zone_function(node_latitude_rad))
        total += longitude_step_rad * mean_zone_area
    return abs(total)


@njit
def pixel_area_m2(latitude_deg, x_resolution_deg, y_resolution_deg):
    """Area in m^2 of one longitude-latitude pixel on the WGS84 ellipsoid (the exact zone strip)."""
    l1 = (latitude_deg - abs(y_resolution_deg) / 2.0) * math.pi / 180.0
    l2 = (latitude_deg + abs(y_resolution_deg) / 2.0) * math.pi / 180.0
    dx = abs(x_resolution_deg) * math.pi / 180.0
    return WGS84_A * WGS84_A * (1.0 - WGS84_E2) * dx * 0.5 * (_wgs84_zone_function(l2) - _wgs84_zone_function(l1))


__all__ = [
    "WGS84", "SPHERE", "EARTH_MODELS", "model_code",
    "degree_metres_x", "degree_metres_y", "metres_per_degree", "shortest_longitude_difference",
    "distance_m", "distance_with_cosines_m", "lonlat_polygon_area_m2", "pixel_area_m2",
]
