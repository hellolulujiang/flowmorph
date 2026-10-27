"""Step 2: the elevation and slope of every pixel's upstream catchment, attributes [1] .. [6].

    [1] mean_elv   m    mean elevation of the catchment, weighted by pixel area
    [2] min_elv    m    lowest elevation in the catchment
    [3] max_elv    m    highest elevation in the catchment
    [4] elv_std    m    standard deviation of elevation, weighted by pixel area (population)
    [5] mean_slp   deg  mean of the local slope (step 1) over the catchment, weighted by pixel area
    [6] slp_std    deg  standard deviation of the local slope, weighted by pixel area (population)

Every pixel counts with its area a (the WGS84 area of a pixel of its row, kept as float32): mean = sum(a z) / sum(a),
std = sqrt(sum(a z^2) / sum(a) - mean^2), 0 when that is not above 0, no data when fewer than two pixels are in the
catchment.  On a longitude-latitude grid a pixel shrinks with the cosine of its latitude.  The elevation and the
slope must cover the network: a network pixel without an elevation or without a slope (-9999 or not finite) is
refused.  The standard deviation is the difference of two moments held in doubles, which resolves a centimetre to
2 % and nothing below a few millimetres: a level surface of ten million pixels at 3,800 m comes out between 0 and
2 mm.  Written where the upstream area is at least the target threshold (10 km^2), -9999 elsewhere.

HOW.  The catchment of p is the run of ranks [seq_dfs(p) - upg(p) + 1, seq_dfs(p)], and every pixel of it but p
has a lower rank than p.  So the ranks are taken from 0 up; when the pixel of rank r is reached, the catchments summed
and not yet taken that lie in its run are exactly those of the pixels flowing into it, on the top of a stack: they
are taken off and added to r's own values, and r's sums go on the stack for the pixel r flows into.  A pit's sums
are finished and are not put back.
"""

import numpy as np
from numba import njit

NODATA = -9999.0
TERRAIN_NAMES = ("mean_elv", "min_elv", "max_elv", "elv_std", "mean_slp", "slp_std")


@njit
def _upstream_terrain(elevation_of_rank, slope_of_rank, weight_of_rank, first_rank_of_rank, is_target_of_rank,
                      is_pit_of_rank):
    pixel_count = elevation_of_rank.size
    nodata32 = np.float32(NODATA)
    outputs = np.full((pixel_count, 6), nodata32, dtype=np.float32)
    stack_rank = np.empty(pixel_count, dtype=np.int64)
    stack_count = np.empty(pixel_count, dtype=np.int64)
    stack_weight_sum = np.empty(pixel_count, dtype=np.float64)
    stack_elevation_sum = np.empty(pixel_count, dtype=np.float64)
    stack_elevation_square_sum = np.empty(pixel_count, dtype=np.float64)
    stack_slope_sum = np.empty(pixel_count, dtype=np.float64)
    stack_slope_square_sum = np.empty(pixel_count, dtype=np.float64)
    stack_elevation_min = np.empty(pixel_count, dtype=np.float32)
    stack_elevation_max = np.empty(pixel_count, dtype=np.float32)
    top = 0
    deepest = 0
    for rank in range(pixel_count):
        elevation = elevation_of_rank[rank]
        slope = slope_of_rank[rank]
        # the pixel's own: an elevation and a slope (every pixel of the network holds both), weighted by its area
        weight = np.float64(weight_of_rank[rank])
        weight_sum = weight
        elevation_sum = weight * np.float64(elevation)
        elevation_square_sum = weight * np.float64(elevation) * np.float64(elevation)
        elevation_min = elevation
        elevation_max = elevation
        count = 1
        slope_sum = weight * np.float64(slope)
        slope_square_sum = weight * np.float64(slope) * np.float64(slope)
        first_rank = first_rank_of_rank[rank]
        while top > 0 and stack_rank[top - 1] >= first_rank:
            top -= 1
            elevation_sum += stack_elevation_sum[top]
            elevation_square_sum += stack_elevation_square_sum[top]
            slope_sum += stack_slope_sum[top]
            slope_square_sum += stack_slope_square_sum[top]
            count += stack_count[top]
            weight_sum += stack_weight_sum[top]
            if stack_elevation_min[top] < elevation_min:
                elevation_min = stack_elevation_min[top]
            if stack_elevation_max[top] > elevation_max:
                elevation_max = stack_elevation_max[top]
        if is_target_of_rank[rank] and count > 0 and weight_sum > 0.0:
            area_sum = weight_sum                  # the sum of the areas divides
            mean_elevation = elevation_sum / area_sum
            mean_slope = slope_sum / area_sum
            outputs[rank, 0] = np.float32(mean_elevation)
            outputs[rank, 1] = elevation_min
            outputs[rank, 2] = elevation_max
            outputs[rank, 4] = np.float32(mean_slope)
            if count >= 2:
                elevation_variance = elevation_square_sum / area_sum - mean_elevation * mean_elevation
                outputs[rank, 3] = np.float32(np.sqrt(elevation_variance)) if elevation_variance > 0.0 else 0.0
                slope_variance = slope_square_sum / area_sum - mean_slope * mean_slope
                outputs[rank, 5] = np.float32(np.sqrt(slope_variance)) if slope_variance > 0.0 else 0.0
        if is_pit_of_rank[rank]:
            continue
        stack_rank[top] = rank
        stack_count[top] = count
        stack_weight_sum[top] = weight_sum
        stack_elevation_sum[top] = elevation_sum
        stack_elevation_square_sum[top] = elevation_square_sum
        stack_slope_sum[top] = slope_sum
        stack_slope_square_sum[top] = slope_square_sum
        stack_elevation_min[top] = elevation_min
        stack_elevation_max[top] = elevation_max
        top += 1
        if top > deepest:
            deepest = top
    return outputs, top, deepest


def upstream_terrain(seq_dfs, upg, idxs_ds, elevation, slope, is_target, pixel_area):
    """The six attributes of step 2, each a float32 array over the pixels (-9999 where not written).

    ``seq_dfs``, ``upg`` and ``idxs_ds`` are the checked index of :mod:`flowmorph.network`; ``elevation`` and
    ``slope`` (step 1) are float arrays over the pixels; ``is_target`` marks the pixels written; ``pixel_area`` is
    the area of every pixel (any unit, the same for all; float32), the weight of its values.
    """
    seq_dfs = np.ascontiguousarray(seq_dfs).reshape(-1)
    network = seq_dfs >= 0
    ranks = seq_dfs[network].astype(np.int64)
    pixel_count = ranks.size
    elevation = np.ascontiguousarray(elevation, dtype=np.float32).reshape(-1)
    slope = np.ascontiguousarray(slope, dtype=np.float32).reshape(-1)
    network_elevation = elevation[network]
    network_slope = slope[network]
    without_elevation = np.count_nonzero(~np.isfinite(network_elevation) | (network_elevation == np.float32(NODATA)))
    without_slope = np.count_nonzero(~np.isfinite(network_slope) | (network_slope == np.float32(NODATA)))
    if without_elevation or without_slope:
        raise ValueError(f"{without_elevation} pixels of the network have no finite elevation and {without_slope} no "
                         f"finite slope; the elevation and the slope must cover the network")
    pixel_area = np.ascontiguousarray(pixel_area, dtype=np.float32).reshape(-1)
    if pixel_area.size != seq_dfs.size:
        raise ValueError("pixel_area does not match the grid")
    network_area = pixel_area[network]
    if network_area.size and not np.all(np.isfinite(network_area) & (network_area > 0)):
        raise ValueError("pixel_area holds a value that is not a finite, positive area on a pixel of the network")
    elevation_of_rank = np.empty(pixel_count, dtype=np.float32)
    slope_of_rank = np.empty(pixel_count, dtype=np.float32)
    weight_of_rank = np.empty(pixel_count, dtype=np.float32)
    first_rank_of_rank = np.empty(pixel_count, dtype=np.int64)
    is_target_of_rank = np.zeros(pixel_count, dtype=np.bool_)
    is_pit_of_rank = np.zeros(pixel_count, dtype=np.bool_)
    cells = np.flatnonzero(network)
    elevation_of_rank[ranks] = network_elevation
    slope_of_rank[ranks] = network_slope
    weight_of_rank[ranks] = network_area
    first_rank_of_rank[ranks] = ranks - np.asarray(upg).reshape(-1)[network].astype(np.int64) + 1
    is_target_of_rank[ranks] = np.asarray(is_target).reshape(-1)[network]
    is_pit_of_rank[ranks] = np.asarray(idxs_ds).reshape(-1)[network] == cells
    outputs_of_rank, left_on_stack, _ = _upstream_terrain(elevation_of_rank, slope_of_rank, weight_of_rank,
                                                          first_rank_of_rank, is_target_of_rank, is_pit_of_rank)
    if left_on_stack:
        raise ValueError(f"{left_on_stack} catchments were never taken by the pixel they flow into")
    result = {}
    for index, name in enumerate(TERRAIN_NAMES):
        values = np.full(seq_dfs.size, np.float32(NODATA), dtype=np.float32)
        values[cells] = outputs_of_rank[ranks, index]
        result[name] = values
    return result


__all__ = ["upstream_terrain", "TERRAIN_NAMES"]
