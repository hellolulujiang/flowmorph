"""Step 2: the elevation and slope of every pixel's upstream catchment, attributes [1] .. [6].

    [1] mean_elv   m    mean elevation of the catchment
    [2] min_elv    m    lowest elevation in the catchment
    [3] max_elv    m    highest elevation in the catchment
    [4] elv_std    m    standard deviation of elevation (population: divided by n)
    [5] mean_slp   deg  mean of the local slope (step 1) of the catchment's pixels
    [6] slp_std    deg  standard deviation of the local slope

Every pixel of the catchment counts once (a plain mean over pixels, not weighted by pixel area); n counts the pixels
with an elevation and divides both means; std = sqrt(sum of squares / n - mean^2), 0 when that is not above 0,
no data when n < 2.  Written where the upstream area is at least the target threshold (10 km^2), -9999 elsewhere.

HOW.  The pixels upstream of p are the ranks [seq_dfs(p) - upg(p) + 1, seq_dfs(p)], and every one of them has a
lower rank than p.  So the ranks are taken from 0 up; when the pixel of rank r is reached, the catchments summed
and not yet taken that lie in its run are exactly those of the pixels flowing into it, on the top of a stack: they
are taken off and added to r's own values, and r's sums go on the stack for the pixel r flows into.  A pit's sums
are finished and are not put back.  The sums are added in a fixed order, so a given index always gives the same numbers.
"""

import numpy as np
from numba import njit

NODATA = -9999.0
TERRAIN_NAMES = ("mean_elv", "min_elv", "max_elv", "elv_std", "mean_slp", "slp_std")


@njit
def _upstream_terrain(elevation_of_rank, slope_of_rank, first_rank_of_rank, is_target_of_rank, is_pit_of_rank):
    pixel_count = elevation_of_rank.size
    nodata32 = np.float32(NODATA)
    outputs = np.full((pixel_count, 6), nodata32, dtype=np.float32)
    stack_rank = np.empty(pixel_count, dtype=np.int64)
    stack_count = np.empty(pixel_count, dtype=np.int64)
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
        if elevation != nodata32:
            elevation_sum = np.float64(elevation)
            elevation_square_sum = np.float64(elevation) * np.float64(elevation)
            elevation_min = elevation
            elevation_max = elevation
            count = 1
        else:
            elevation_sum = 0.0
            elevation_square_sum = 0.0
            elevation_min = np.float32(1e30)
            elevation_max = np.float32(-1e30)
            count = 0
        if slope != nodata32:
            slope_sum = np.float64(slope)
            slope_square_sum = np.float64(slope) * np.float64(slope)
        else:
            slope_sum = 0.0
            slope_square_sum = 0.0
        first_rank = first_rank_of_rank[rank]
        while top > 0 and stack_rank[top - 1] >= first_rank:
            top -= 1
            elevation_sum += stack_elevation_sum[top]
            elevation_square_sum += stack_elevation_square_sum[top]
            slope_sum += stack_slope_sum[top]
            slope_square_sum += stack_slope_square_sum[top]
            count += stack_count[top]
            if stack_elevation_min[top] < elevation_min:
                elevation_min = stack_elevation_min[top]
            if stack_elevation_max[top] > elevation_max:
                elevation_max = stack_elevation_max[top]
        if is_target_of_rank[rank] and count > 0:
            pixels = np.float64(count)
            mean_elevation = elevation_sum / pixels
            mean_slope = slope_sum / pixels
            outputs[rank, 0] = np.float32(mean_elevation)
            outputs[rank, 1] = elevation_min
            outputs[rank, 2] = elevation_max
            outputs[rank, 4] = np.float32(mean_slope)
            if count >= 2:
                elevation_variance = elevation_square_sum / pixels - mean_elevation * mean_elevation
                outputs[rank, 3] = np.float32(np.sqrt(elevation_variance)) if elevation_variance > 0.0 else 0.0
                slope_variance = slope_square_sum / pixels - mean_slope * mean_slope
                outputs[rank, 5] = np.float32(np.sqrt(slope_variance)) if slope_variance > 0.0 else 0.0
        if is_pit_of_rank[rank]:
            continue
        stack_rank[top] = rank
        stack_count[top] = count
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


def upstream_terrain(seq_dfs, upg, idxs_ds, elevation, slope, is_target):
    """The six attributes of step 2, each a float32 array over the cells (-9999 where not written).

    ``seq_dfs``, ``upg`` and ``idxs_ds`` are the checked index of :mod:`flowmorph.network`; ``elevation`` and
    ``slope`` (step 1) are float arrays over the cells; ``is_target`` marks the pixels written.
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
    bad_slope = np.count_nonzero(~np.isfinite(network_slope))
    if without_elevation or bad_slope:
        raise ValueError(f"{without_elevation} pixels of the network have no finite elevation and {bad_slope} a slope "
                         f"that is not finite; the elevation must cover the network")
    elevation_of_rank = np.empty(pixel_count, dtype=np.float32)
    slope_of_rank = np.empty(pixel_count, dtype=np.float32)
    first_rank_of_rank = np.empty(pixel_count, dtype=np.int64)
    is_target_of_rank = np.zeros(pixel_count, dtype=np.bool_)
    is_pit_of_rank = np.zeros(pixel_count, dtype=np.bool_)
    cells = np.flatnonzero(network)
    elevation_of_rank[ranks] = network_elevation
    slope_of_rank[ranks] = network_slope
    first_rank_of_rank[ranks] = ranks - np.asarray(upg).reshape(-1)[network].astype(np.int64) + 1
    is_target_of_rank[ranks] = np.asarray(is_target).reshape(-1)[network]
    is_pit_of_rank[ranks] = np.asarray(idxs_ds).reshape(-1)[network] == cells
    outputs_of_rank, left_on_stack, _ = _upstream_terrain(elevation_of_rank, slope_of_rank, first_rank_of_rank,
                                                          is_target_of_rank, is_pit_of_rank)
    if left_on_stack:
        raise ValueError(f"{left_on_stack} catchments were never taken by the pixel they flow into")
    result = {}
    for index, name in enumerate(TERRAIN_NAMES):
        values = np.full(seq_dfs.size, np.float32(NODATA), dtype=np.float32)
        values[cells] = outputs_of_rank[ranks, index]
        result[name] = values
    return result


__all__ = ["upstream_terrain", "TERRAIN_NAMES"]
