"""Step 4: attributes [11] .. [16], pixel by pixel from steps 2 and 3 and the upstream area; no traversal.

    A      upstream area of the pixel (km^2);  P  perimeter [7] (km);  L  basin length L_par [8] (km)

    [11] relief                m  max_elv - min_elv   (Strahler 1952)
    [12] hypsometric_integral  -  (mean_elv - min_elv) / (max_elv - min_elv)   (Strahler 1952, in the
                                  elevation-relief ratio form); no data when the relief is 0
    [13] elongation_ratio      -  (2 / L) sqrt(A / pi)   (Schumm's 1956 formula, with L_par as the length)
    [14] gravelius             -  P / (2 sqrt(pi A))   (Gravelius 1914)
    [15] circularity           -  4 pi A / P^2   (Miller 1953); gravelius = 1 / sqrt(circularity)
    [16] lemniscate_ratio      -  P_lemniscate / P   (Chorley, Malm & Pogorzelski 1957)

[16].  The ideal lemniscate loop with the basin's area A and longest diameter L has k = pi L^2 / (4 A) (their eq. 1)
and perimeter P_lemniscate = 2 L E(K), K^2 = 1 - 1/k^2 (eqs. 2-3), E the complete elliptic integral of the second
kind.  A catchment whose outlet is not at its tip can be fuller than the circle of diameter L, k < 1; the same
curve rho = L cos(k theta) then has perimeter (2 L / k) E(K'), K'^2 = 1 - k^2, and stays one loop through the
outlet while k >= 1/2.  For k < 1/2 it crosses itself and the pixel is written as no data.
"""

import numpy as np
from scipy.special import ellipe

NODATA = -9999.0
SHAPE_NAMES = ("relief", "hypsometric_integral", "elongation_ratio", "gravelius", "circularity", "lemniscate_ratio")


def _lemniscate_ratio(length_km, area_km2, perimeter_km):
    """[16] for arrays of float64 with A > 0, L > 0 and P > 0; no data where k < 1/2."""
    k = np.pi * length_km * length_km / (4.0 * area_km2)
    ratio = np.full(k.shape, NODATA, dtype=np.float64)
    at_least_one = k >= 1.0
    half_to_one = (k >= 0.5) & ~at_least_one
    modulus_squared = np.zeros(k.shape, dtype=np.float64)
    factor = np.zeros(k.shape, dtype=np.float64)
    modulus_squared[at_least_one] = 1.0 - 1.0 / (k[at_least_one] * k[at_least_one])
    factor[at_least_one] = 2.0
    modulus_squared[half_to_one] = 1.0 - k[half_to_one] * k[half_to_one]
    factor[half_to_one] = 2.0 / k[half_to_one]
    computed = at_least_one | half_to_one
    # E(1) = 1: K^2 rounds to 1 only for k above about 1e8
    elliptic = np.where(modulus_squared < 1.0, ellipe(np.minimum(modulus_squared, 1.0)), 1.0)
    ratio[computed] = factor[computed] * length_km[computed] * elliptic[computed] / perimeter_km[computed]
    return ratio


def shape_arithmetic(upa, perimeter_km, basin_length_km, mean_elv, min_elv, max_elv, is_target):
    """The six attributes of step 4, each a float32 array of the shape of the inputs (-9999 where not written)."""
    nodata32 = np.float32(NODATA)
    area = np.asarray(upa, dtype=np.float32)
    perimeter = np.asarray(perimeter_km, dtype=np.float32)
    length = np.asarray(basin_length_km, dtype=np.float32)
    mean_elevation = np.asarray(mean_elv, dtype=np.float32)
    min_elevation = np.asarray(min_elv, dtype=np.float32)
    max_elevation = np.asarray(max_elv, dtype=np.float32)
    target = np.asarray(is_target, dtype=bool)
    result = {name: np.full(area.shape, nodata32, dtype=np.float32) for name in SHAPE_NAMES}

    area64 = area.astype(np.float64)
    perimeter64 = perimeter.astype(np.float64)
    length64 = length.astype(np.float64)
    perimeter_is_there = target & (perimeter != nodata32) & (perimeter > 0.0)
    length_is_there = target & (length != nodata32) & (length > 0.0)

    where = perimeter_is_there
    result["circularity"][where] = (4.0 * np.pi * area64[where] / (perimeter64[where] * perimeter64[where])
                                    ).astype(np.float32)
    result["gravelius"][where] = (perimeter64[where] / (2.0 * np.sqrt(np.pi * area64[where]))).astype(np.float32)
    where = length_is_there
    result["elongation_ratio"][where] = ((2.0 / length64[where]) * np.sqrt(area64[where] / np.pi)
                                         ).astype(np.float32)
    where = length_is_there & perimeter_is_there
    result["lemniscate_ratio"][where] = _lemniscate_ratio(length64[where], area64[where], perimeter64[where]
                                                          ).astype(np.float32)
    where = target & (min_elevation != nodata32) & (max_elevation != nodata32)
    result["relief"][where] = max_elevation[where] - min_elevation[where]
    where = where & (mean_elevation != nodata32)
    relief64 = max_elevation.astype(np.float64) - min_elevation.astype(np.float64)
    where = where & (relief64 > 0.0)
    result["hypsometric_integral"][where] = ((mean_elevation[where].astype(np.float64)
                                              - min_elevation[where].astype(np.float64)) / relief64[where]
                                             ).astype(np.float32)
    return result


__all__ = ["shape_arithmetic", "SHAPE_NAMES"]
