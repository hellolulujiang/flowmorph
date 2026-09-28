"""The D8 network FlowMorph walks on, and the index of every pixel's catchment.

The flow directions are in the MERIT Hydro convention: a power of two clockwise from east (1 E, 2 SE, 4 S, 8 SW,
16 W, 32 NW, 64 N, 128 NE), 0 a river mouth and 255 an inland depression (both pits: the pixel is the outlet of
its basin), 247 no data.

THE INDEX.  The pixels of each basin are numbered depth first from the outlet, the larger upstream area first at
every confluence (ties: the smaller cell index), and the numbering is then turned within the basin so that the
outlet takes the highest rank of the basin's run.  That is FlowTopo's ``seq_dfs``.  A pixel's catchment, the pixel
included, is then the run of ranks

    [seq_dfs(p) - upg(p) + 1, seq_dfs(p)]

where ``upg`` is the number of pixels upstream of p, p included.  Every question FlowMorph asks of a catchment
("is q upstream of p") is two comparisons on that run.

:func:`check_index` checks a given ``seq_dfs`` and ``upg`` against the flow directions before they are relied on:
every pixel of the network has one rank, a pixel's rank is below the rank of the pixel it
flows into, and going through the ranks from 0 up, the runs already seen and not yet taken lie side by side exactly
under the run of the pixel that takes them and all flow into it.  By induction from the sources the run of every
pixel then holds exactly its upstream pixels.
"""

import numpy as np
from numba import njit

D8_NODATA = 247
D8_PITS = (0, 255)
_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
_ROW_OFFSETS = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
_COL_OFFSETS = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int64)


@njit
def _downstream_of_grid(d8, nrow, ncol):
    """The flat index each cell flows into (its own at a pit), -1 off the network; the first bad cell and the
    reason (0 none, 1 a code that is not D8, 2 a flow out of the grid, 3 a flow into a cell off the network)."""
    idxs_ds = np.full(nrow * ncol, -1, dtype=np.int64)
    for row in range(nrow):
        for col in range(ncol):
            cell = row * ncol + col
            code = d8[cell]
            if code == D8_NODATA:
                continue
            if code == 0 or code == 255:
                idxs_ds[cell] = cell
                continue
            direction = -1
            for k in range(8):
                if _CODES[k] == code:
                    direction = k
                    break
            if direction < 0:
                return idxs_ds, cell, 1
            next_row = row + _ROW_OFFSETS[direction]
            next_col = col + _COL_OFFSETS[direction]
            if next_row < 0 or next_row >= nrow or next_col < 0 or next_col >= ncol:
                return idxs_ds, cell, 2
            next_cell = next_row * ncol + next_col
            if d8[next_cell] == D8_NODATA:
                return idxs_ds, cell, 3
            idxs_ds[cell] = next_cell
    return idxs_ds, -1, 0


def downstream_index(d8):
    """The flat index each cell of a 2-D MERIT D8 grid flows into: its own at a pit (0 or 255), -1 off the network
    (247).  A code that is not D8, a flow out of the grid and a flow into a cell off the network are refused: the
    catchment of every pixel must lie inside the grid."""
    d8 = np.asarray(d8)
    if d8.ndim != 2:
        raise ValueError("the flow directions must be a 2-D grid")
    if d8.dtype.kind not in "iu" or (d8.size and (d8.min() < 0 or d8.max() > 255)):
        raise ValueError("the flow directions must be whole numbers 0 .. 255 (MERIT Hydro's D8 codes)")
    nrow, ncol = d8.shape
    idxs_ds, bad_cell, reason = _downstream_of_grid(np.ascontiguousarray(d8, dtype=np.uint8).reshape(-1), nrow, ncol)
    if reason:
        row, col = divmod(int(bad_cell), ncol)
        what = {1: f"has the code {int(d8[row, col])}, which is not a MERIT D8 code",
                2: "flows out of the grid",
                3: "flows into a cell without flow direction (247)"}[reason]
        raise ValueError(f"cell (row {row}, col {col}) {what}; the grid must hold whole basins")
    return idxs_ds


@njit
def _dfs_sequence(idxs_ds, upa):
    """seq_dfs and upg of every cell (-1 and 0 off the network), the pixels ranked and the network pixels."""
    size = idxs_ds.size
    donor_offset = np.zeros(size + 1, dtype=np.int64)
    network_pixel_count = 0
    for cell in range(size):
        downstream = idxs_ds[cell]
        if downstream < 0:
            continue
        network_pixel_count += 1
        if downstream != cell:
            donor_offset[downstream + 1] += 1
    for cell in range(size):
        donor_offset[cell + 1] += donor_offset[cell]
    donors = np.empty(donor_offset[size], dtype=np.int64)
    fill_position = donor_offset[:size].copy()
    for cell in range(size):
        downstream = idxs_ds[cell]
        if downstream >= 0 and downstream != cell:
            donors[fill_position[downstream]] = cell
            fill_position[downstream] += 1
    # the donors of each cell by upstream area, the larger first; a stable insertion sort keeps the smaller index
    # first among equal areas
    for cell in range(size):
        first = donor_offset[cell]
        last = donor_offset[cell + 1]
        for position in range(first + 1, last):
            moving = donors[position]
            moving_area = upa[moving]
            insert = position - 1
            while insert >= first and upa[donors[insert]] < moving_area:
                donors[insert + 1] = donors[insert]
                insert -= 1
            donors[insert + 1] = moving

    seq = np.full(size, -1, dtype=np.int32)
    stack = np.empty(max(network_pixel_count, 1), dtype=np.int64)
    preorder = np.empty(max(network_pixel_count, 1), dtype=np.int64)
    basin_start = 0
    for outlet in range(size):
        if idxs_ds[outlet] != outlet:
            continue
        # the basin's pixels in preorder from the outlet: pushed smallest area first, so the largest comes off first
        stack_top = 0
        stack[stack_top] = outlet
        stack_top += 1
        visited = 0
        while stack_top > 0:
            stack_top -= 1
            cell = stack[stack_top]
            preorder[visited] = cell
            visited += 1
            for position in range(donor_offset[cell + 1] - 1, donor_offset[cell] - 1, -1):
                stack[stack_top] = donors[position]
                stack_top += 1
        # turned within the basin's run: the outlet takes its top, basin_start + visited - 1
        for position in range(visited):
            seq[preorder[position]] = basin_start + visited - 1 - position
        basin_start += visited

    upg = np.zeros(size, dtype=np.int64)
    if basin_start != network_pixel_count:
        return seq, upg, basin_start, network_pixel_count          # cells in a cycle reach no pit
    # upg: the pixels upstream of each pixel, itself included, summed downstream in rank order
    cell_of_rank = np.empty(max(network_pixel_count, 1), dtype=np.int64)
    for cell in range(size):
        if seq[cell] >= 0:
            cell_of_rank[seq[cell]] = cell
    for rank in range(network_pixel_count):
        cell = cell_of_rank[rank]
        upg[cell] += 1
        downstream = idxs_ds[cell]
        if downstream != cell:
            upg[downstream] += upg[cell]
    return seq, upg, basin_start, network_pixel_count


def dfs_sequence(idxs_ds, upa):
    """FlowTopo's ``seq_dfs`` and the upstream pixel count ``upg`` of a network.

    Parameters
    ----------
    idxs_ds : 1-D int array, from :func:`downstream_index`
    upa : 1-D float array of the same size, the upstream area that orders the donors of a confluence; it must be
        finite and positive on every cell of the network.

    Returns
    -------
    seq_dfs : int32 array, the rank of every cell, -1 off the network
    upg : int64 array, 0 off the network
    """
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int64)
    upa = np.ascontiguousarray(upa, dtype=np.float32).reshape(-1)
    if upa.size != idxs_ds.size:
        raise ValueError("upa must hold one value for every cell")
    network_upa = upa[idxs_ds >= 0]
    if network_upa.size and not np.all(np.isfinite(network_upa) & (network_upa > 0)):
        raise ValueError("upa must be finite and positive on every cell of the network: it orders the donors of "
                         "a confluence")
    if np.count_nonzero(idxs_ds >= 0) > np.iinfo(np.int32).max:
        raise ValueError("the network holds more pixels than int32 ranks can number")
    seq, upg, ranked_count, network_pixel_count = _dfs_sequence(idxs_ds, upa)
    if ranked_count != network_pixel_count:
        raise ValueError(f"{network_pixel_count - ranked_count} pixels of the network reach no pit: the flow "
                         f"directions hold a cycle")
    return seq, upg


@njit
def _check_index(seq, upg, idxs_ds, pixel_count):
    """0 when the index holds, else a reason code and the rank or cell it failed at."""
    size = seq.size
    seen = np.zeros(pixel_count, dtype=np.bool_)
    upstream_count_of_rank = np.zeros(pixel_count, dtype=np.int64)
    downstream_rank_of_rank = np.zeros(pixel_count, dtype=np.int64)
    for cell in range(size):
        rank = seq[cell]
        downstream = idxs_ds[cell]
        if downstream < 0:
            if rank != -1 or upg[cell] != 0:
                return 1, cell                                   # off the network with a rank or a count
            continue
        if rank < 0 or rank >= pixel_count:
            return 2, cell                                       # a network pixel without a rank in 0 .. n - 1
        if seen[rank]:
            return 3, cell                                       # a rank given twice
        seen[rank] = True
        if upg[cell] < 1 or upg[cell] > rank + 1:
            return 4, cell                                       # the run would start below rank 0
        downstream_rank = rank if downstream == cell else seq[downstream]
        if downstream != cell and downstream_rank <= rank:
            return 5, cell                                       # a pixel ranked above the pixel it flows into
        upstream_count_of_rank[rank] = upg[cell]
        downstream_rank_of_rank[rank] = downstream_rank
    pending_rank = np.empty(pixel_count, dtype=np.int64)
    pending_downstream = np.empty(pixel_count, dtype=np.int64)
    pending_count = np.empty(pixel_count, dtype=np.int64)
    top = 0
    for rank in range(pixel_count):
        range_first_rank = rank - upstream_count_of_rank[rank] + 1
        next_rank_expected = rank - 1
        range_pixel_count = 1
        while top > 0 and pending_rank[top - 1] >= range_first_rank:
            if pending_rank[top - 1] != next_rank_expected or pending_downstream[top - 1] != rank:
                return 6, rank                                   # the runs do not follow the flow
            next_rank_expected = pending_rank[top - 1] - pending_count[top - 1]
            range_pixel_count += pending_count[top - 1]
            top -= 1
        if next_rank_expected != range_first_rank - 1:
            return 7, rank                                       # upg is not the count of the pixels flowing in
        if downstream_rank_of_rank[rank] == rank:
            continue
        pending_rank[top] = rank
        pending_downstream[top] = downstream_rank_of_rank[rank]
        pending_count[top] = range_pixel_count
        top += 1
    if top != 0:
        return 8, pending_rank[0]                                # runs never taken by the pixel they flow into
    return 0, -1


_INDEX_FAILURES = {
    1: "a cell off the network has a rank or an upstream count",
    2: "a pixel of the network has no rank in 0 .. n - 1",
    3: "a rank is given twice",
    4: "an upstream count would start the run below rank 0",
    5: "a pixel is ranked above the pixel it flows into",
    6: "the runs [seq_dfs - upg + 1, seq_dfs] do not follow the flow",
    7: "an upstream count is not the number of pixels flowing in",
    8: "a run is never taken by the pixel it flows into",
}


def check_index(seq_dfs, upg, idxs_ds):
    """Raise ValueError unless every run [seq_dfs - upg + 1, seq_dfs] holds exactly its pixel's upstream pixels."""
    seq_dfs = np.ascontiguousarray(seq_dfs).reshape(-1)
    upg = np.ascontiguousarray(upg).reshape(-1)
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int64)
    if seq_dfs.size != idxs_ds.size or upg.size != idxs_ds.size:
        raise ValueError("seq_dfs and upg must hold one value for every cell")
    if seq_dfs.dtype.kind not in "iu" or upg.dtype.kind not in "iu":
        raise ValueError("seq_dfs and upg must be whole numbers")
    pixel_count = int(np.count_nonzero(idxs_ds >= 0))
    reason, where = _check_index(seq_dfs.astype(np.int64), upg.astype(np.int64), idxs_ds, pixel_count)
    if reason:
        raise ValueError(f"seq_dfs and upg do not index the network: {_INDEX_FAILURES[reason]} (at {where})")


__all__ = ["D8_NODATA", "D8_PITS", "downstream_index", "dfs_sequence", "check_index"]
