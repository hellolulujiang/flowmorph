"""What each piece promises, on small grids made up for the purpose (the numbers are checked on the real basin in
test_example.py; these only check the rules and the refusals)."""

import math

import numpy as np
import pytest

import flowmorph
from flowmorph import earth
from flowmorph.grid import pixel_edges_deg, snap_transform
from flowmorph.network import check_index, dfs_sequence, downstream_index
from flowmorph.shape import shape_arithmetic

TRANSFORM = (10.0, 1.0 / 1200.0, 0.0, 50.0, 0.0, -1.0 / 1200.0)


def _line_grid():
    """Four cells flowing east into a pit, with a fifth cell draining into the second from the north."""
    d8 = np.full((2, 4), 247, dtype=np.uint8)
    d8[1, :3] = 1        # east
    d8[1, 3] = 0         # river mouth
    d8[0, 1] = 4         # south, into (1, 1)
    return d8


# ---- the network ----

def test_downstream_index_follows_the_codes():
    idxs_ds = downstream_index(_line_grid())
    assert idxs_ds.tolist() == [-1, 5, -1, -1, 5, 6, 7, 7]


@pytest.mark.parametrize("code, reason", [(3, "not a MERIT D8 code"), (16, "flows out of the grid")])
def test_downstream_index_refuses(code, reason):
    d8 = _line_grid()
    d8[1, 0] = code
    with pytest.raises(ValueError, match=reason):
        downstream_index(d8)


def test_a_flow_into_no_data_is_refused():
    d8 = _line_grid()
    d8[0, 1] = 64        # north, into row -1 ... out of the grid
    with pytest.raises(ValueError, match="out of the grid"):
        downstream_index(d8)
    d8[0, 1] = 1         # east, into (0, 2), which has no flow direction
    with pytest.raises(ValueError, match="without flow direction"):
        downstream_index(d8)


def test_the_catchment_is_a_run_of_ranks():
    d8 = _line_grid()
    idxs_ds = downstream_index(d8)
    upa = np.array([0, 1, 0, 0, 1, 3, 4, 5], dtype=np.float32)
    seq, upg = dfs_sequence(idxs_ds, upa)
    check_index(seq, upg, idxs_ds)
    assert upg[7] == 5 and seq[7] == 4          # the outlet holds the top rank and all five pixels
    for cell in np.flatnonzero(seq >= 0):
        run = set(np.flatnonzero((seq >= seq[cell] - upg[cell] + 1) & (seq <= seq[cell])).tolist())
        upstream = {cell}
        grew = True
        while grew:
            grew = False
            for other in np.flatnonzero(idxs_ds >= 0):
                if idxs_ds[other] in upstream and other not in upstream:
                    upstream.add(int(other))
                    grew = True
        assert run == upstream


def test_the_larger_donor_is_visited_first():
    # (1, 1) receives from (1, 0) and (0, 1); the larger area is visited first from the outlet, so it has the
    # higher rank of the two once the numbering is turned (the outlet takes the highest)
    d8 = _line_grid()
    idxs_ds = downstream_index(d8)
    upa = np.array([0, 2, 0, 0, 1, 4, 5, 6], dtype=np.float32)       # (0, 1) larger than (1, 0)
    seq, _ = dfs_sequence(idxs_ds, upa)
    assert seq[1] > seq[4]
    upa[1], upa[4] = 1, 2
    seq, _ = dfs_sequence(idxs_ds, upa)
    assert seq[4] > seq[1]


def test_a_cycle_is_refused():
    d8 = np.array([[1, 16]], dtype=np.uint8)        # two cells pointing at each other
    idxs_ds = downstream_index(d8)
    with pytest.raises(ValueError, match="cycle"):
        dfs_sequence(idxs_ds, np.ones(2, dtype=np.float32))


def test_an_area_that_is_not_finite_is_refused():
    idxs_ds = downstream_index(_line_grid())
    upa = np.array([0, 1, 0, 0, np.nan, 3, 4, 5], dtype=np.float32)
    with pytest.raises(ValueError, match="finite and positive"):
        dfs_sequence(idxs_ds, upa)


def test_a_wrong_index_is_refused():
    idxs_ds = downstream_index(_line_grid())
    upa = np.array([0, 1, 0, 0, 1, 3, 4, 5], dtype=np.float32)
    seq, upg = dfs_sequence(idxs_ds, upa)
    bad_upg = upg.copy()
    bad_upg[5] -= 1
    with pytest.raises(ValueError, match="do not index"):
        check_index(seq, bad_upg, idxs_ds)
    bad_seq = seq.copy()
    bad_seq[[1, 7]] = bad_seq[[7, 1]]
    with pytest.raises(ValueError, match="do not index"):
        check_index(bad_seq, upg, idxs_ds)


# ---- the grid and the earth ----

def test_the_transform_is_snapped_as_the_c_snaps_it():
    header = (147.51708333, 0.00083333, 0.0, -42.41875, 0.0, -0.00083333)
    assert snap_transform(header) == (147.517083333333, 0.000833333333, 0.0, -42.41875, 0.0, -0.000833333333)
    assert pixel_edges_deg(header) == (1.0 / 1200.0, 1.0 / 1200.0)
    assert snap_transform((0.0, 0.3, 0.0, 0.0, 0.0, -0.3)) == (0.0, 0.3, 0.0, 0.0, 0.0, -0.3)


def test_the_hull_area_of_one_pixel_is_its_area():
    latitude = 60.0
    half = 0.5 / 1200.0
    lons = np.array([-half, half, half, -half])
    lats = np.array([latitude - half, latitude - half, latitude + half, latitude + half])
    polygon = earth.lonlat_polygon_area_m2(lons, lats, earth.WGS84)
    pixel = earth.pixel_area_m2(latitude, 1.0 / 1200.0, 1.0 / 1200.0)
    assert abs(polygon / pixel - 1.0) < 1e-9


def test_a_distance_across_the_antimeridian_is_short():
    across = earth.distance_m(0.0, 179.9995, 0.0, -179.9995, earth.WGS84)
    assert across < 200.0
    assert earth.distance_m(10.0, 20.0, 10.0, 20.0, earth.WGS84) == 0.0


# ---- the arithmetic of step 4 ----

def _one(value):
    return np.array([[value]], dtype=np.float32)


def test_the_lemniscate_of_a_circle():
    # k = pi L^2 / (4 A) = 1 is the circle of diameter L; its perimeter is pi L and E(0) = pi / 2
    length = 10.0
    area = math.pi * length * length / 4.0
    perimeter = math.pi * length
    values = shape_arithmetic(_one(area), _one(perimeter), _one(length), _one(100.0), _one(50.0), _one(150.0),
                              np.array([[True]]))
    assert abs(float(values["lemniscate_ratio"][0, 0]) - 1.0) < 1e-6
    assert abs(float(values["circularity"][0, 0]) - 1.0) < 1e-6
    assert abs(float(values["hypsometric_integral"][0, 0]) - 0.5) < 1e-7
    assert float(values["relief"][0, 0]) == 100.0


def test_a_lemniscate_that_crosses_itself_is_no_data():
    # k < 1/2: a short length for the area
    values = shape_arithmetic(_one(1000.0), _one(200.0), _one(10.0), _one(1.0), _one(0.0), _one(2.0),
                              np.array([[True]]))
    assert float(values["lemniscate_ratio"][0, 0]) == -9999.0
    assert float(values["elongation_ratio"][0, 0]) > 0.0


def test_a_flat_catchment_has_no_hypsometric_integral():
    values = shape_arithmetic(_one(20.0), _one(20.0), _one(5.0), _one(7.0), _one(7.0), _one(7.0),
                              np.array([[True]]))
    assert float(values["relief"][0, 0]) == 0.0
    assert float(values["hypsometric_integral"][0, 0]) == -9999.0


def test_nothing_is_written_off_the_targets():
    values = shape_arithmetic(_one(20.0), _one(20.0), _one(5.0), _one(7.0), _one(1.0), _one(9.0),
                              np.array([[False]]))
    assert all(float(grid[0, 0]) == -9999.0 for grid in values.values())


# ---- the object ----

def test_the_object_refuses_a_rotated_or_south_up_grid():
    d8 = _line_grid()
    upa = np.ones(d8.shape, dtype=np.float32)
    with pytest.raises(ValueError, match="north up"):
        flowmorph.FlowMorph(d8, upa, (10.0, 1 / 1200, 0.1, 50.0, 0.0, -1 / 1200))
    with pytest.raises(ValueError, match="north up"):
        flowmorph.FlowMorph(d8, upa, (10.0, 1 / 1200, 0.0, 50.0, 0.0, 1 / 1200))


def test_the_index_must_come_whole():
    d8 = _line_grid()
    upa = np.ones(d8.shape, dtype=np.float32)
    with pytest.raises(ValueError, match="together"):
        flowmorph.FlowMorph(d8, upa, TRANSFORM, seq_dfs=np.zeros(d8.size, dtype=np.int32))


def test_a_small_grid_runs_through():
    d8 = _line_grid()
    upa = np.array([[0, 20, 0, 0], [15, 40, 50, 60]], dtype=np.float32)
    elevation = np.array([[-9999, 30, -9999, -9999], [40, 20, 10, 5]], dtype=np.float32)
    fm = flowmorph.FlowMorph(d8, upa, TRANSFORM, elevation=elevation)
    values = fm.attributes()
    assert set(values) == {"slp"} | set(flowmorph.ATTRIBUTE_NAMES)
    assert float(values["max_elv"][1, 3]) == 40.0 and float(values["min_elv"][1, 3]) == 5.0
    assert float(values["mean_elv"][1, 3]) == pytest.approx(21.0)
    assert np.all(values["mean_elv"][0, [0, 2, 3]] == -9999.0)


def test_a_threshold_that_is_not_a_positive_area_is_refused():
    d8 = _line_grid()
    upa = np.ones(d8.shape, dtype=np.float32)
    for threshold in (float("nan"), 0.0, -9999.0):
        with pytest.raises(ValueError, match="min_upstream_area_km2"):
            flowmorph.FlowMorph(d8, upa, TRANSFORM, min_upstream_area_km2=threshold)


def test_a_grid_in_metres_is_refused():
    d8 = _line_grid()
    upa = np.ones(d8.shape, dtype=np.float32)
    with pytest.raises(ValueError, match="longitude-latitude"):
        flowmorph.FlowMorph(d8, upa, (500000.0, 90.0, 0.0, 4500000.0, 0.0, -90.0))
    with pytest.raises(ValueError, match="longitude-latitude"):
        flowmorph.FlowMorph(d8, upa, TRANSFORM, crs="EPSG:32633")


def test_the_ring_across_the_antimeridian_is_read_from_both_ends(tmp_path):
    import rasterio
    from flowmorph.raster import read_ring
    # a global grid of 10-degree pixels whose value is its column; a grid of the last two columns reads its ring
    # from column 33 and, across 180 degrees, from column 0
    width, height = 36, 18
    values = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    path = tmp_path / "global.tif"
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32",
                       crs="EPSG:4326", transform=rasterio.Affine(10.0, 0.0, -180.0, 0.0, -10.0, 90.0),
                       nodata=-9999.0) as dataset:
        dataset.write(values, 1)
    ring = read_ring(str(path), (2, 2), (160.0, 10.0, 0.0, 30.0, 0.0, -10.0))
    assert ring.shape == (4, 4)
    assert ring[1].tolist() == [33.0, 34.0, 35.0, 0.0]


def test_what_the_object_hands_out_is_read_only_and_the_inputs_are_left_alone():
    d8 = _line_grid()
    upa = np.array([[0, 20, 0, 0], [15, 40, 50, 60]], dtype=np.float32)
    elevation = np.array([[-9999, 30, -9999, -9999], [40, 20, 10, 5]], dtype=np.float32)
    fm = flowmorph.FlowMorph(d8, upa, TRANSFORM, elevation=elevation)
    values = fm.attributes()
    for name in ("slp", "mean_elv", "perimeter_km", "relief"):
        with pytest.raises(ValueError):
            values[name][1, 3] = 0.0
    with pytest.raises(ValueError):
        fm.upa[0, 0] = 1.0
    upa[1, 3] = 1.0                     # the caller's array is still theirs, and the object kept its own copy
    assert float(fm.upa[1, 3]) == 60.0


def test_replacing_an_entry_of_a_returned_dict_changes_nothing_later():
    d8 = _line_grid()
    upa = np.array([[0, 20, 0, 0], [15, 40, 50, 60]], dtype=np.float32)
    elevation = np.array([[-9999, 30, -9999, -9999], [40, 20, 10, 5]], dtype=np.float32)
    fm = flowmorph.FlowMorph(d8, upa, TRANSFORM, elevation=elevation)
    before = {name: grid.copy() for name, grid in fm.attributes().items()}
    terrain = fm.upstream_terrain()
    terrain["mean_elv"] = terrain["mean_elv"] / 1000.0
    boundary = fm.boundary_walk()
    boundary["perimeter_km"] = boundary["perimeter_km"] * 0.0
    fm.shape_arithmetic()["relief"] = None
    after = fm.attributes()
    for name, grid in before.items():
        assert np.array_equal(after[name], grid), name
