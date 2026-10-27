"""What each piece promises, on small grids made up for the purpose (the numbers of the real basin are checked in
test_example.py; these check the rules, the refusals and the formulas on made-up numbers)."""

import math

import numpy as np
import pytest

import flowmorph
from ellipsoid_area import area_of_rows
from flowmorph import earth
from flowmorph.grid import pixel_edges_deg, snap_transform
from flowmorph.network import check_index, dfs_sequence, downstream_index
from flowmorph.shape import shape_arithmetic

TRANSFORM = (10.0, 1.0 / 1200.0, 0.0, 50.0, 0.0, -1.0 / 1200.0)


def _line_grid():
    """Four pixels flowing east into a pit, with a fifth pixel draining into the second from the north."""
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
    d8 = np.array([[1, 16]], dtype=np.uint8)        # two pixels pointing at each other
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

def test_the_transform_is_snapped_to_the_grid_and_rounded():
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


def test_the_entries_called_on_their_own_refuse_a_rotated_grid():
    """The standalone slope and boundary functions require north-up grids, as the FlowMorph object does."""
    from flowmorph.boundary import boundary_walk
    from flowmorph.earth import model_code
    from flowmorph.slope import local_slope
    sheared = (10.0, 0.01, 0.01, 47.0, 0.0, -0.01)
    elevation = np.tile(np.array([0.0, 100.0, 200.0], dtype=np.float32), (3, 1))
    with pytest.raises(ValueError, match="north up"):
        local_slope(elevation, sheared)
    with pytest.raises(ValueError, match="six finite"):
        local_slope(elevation, (10.0, 0.01, 0.0, float("nan"), 0.0, -0.01))
    assert local_slope(elevation, (10.0, 0.01, 0.0, 47.0, 0.0, -0.01)).shape == (3, 3)
    one = np.zeros(1, dtype=np.int64)
    area = np.ones(1, dtype=np.float32)
    target = np.ones(1, dtype=bool)
    model = model_code("wgs84")
    with pytest.raises(ValueError, match="north up"):
        boundary_walk(one, one + 1, area, target, (1, 1), sheared, model)


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
    # the mean is weighted by pixel area: one pixel of the upper row (30 m) and four of the lower (40, 20, 10, 5 m)
    area = fm.pixel_area().astype(np.float64)
    by_area = (area[0, 1] * 30.0 + area[1, 0] * (40.0 + 20.0 + 10.0 + 5.0)) / (area[0, 1] + 4.0 * area[1, 0])
    assert float(values["mean_elv"][1, 3]) == pytest.approx(by_area, rel=1e-6)
    assert by_area != 21.0 and abs(by_area - 21.0) < 1e-3          # the plain mean is 21; the rows are 1/1200 degree apart
    assert np.all(values["mean_elv"][0, [0, 2, 3]] == -9999.0)


def test_the_means_of_the_terrain_are_weighted_by_pixel_area():
    """One column of 2,000 pixels of 0.01 degree running south from 60 N to 40 N, the elevation rising a metre a
    pixel towards the south.  A pixel at 40 N is half as large again as one at 60 N: the mean over the catchment's
    area lies 0.69 degree south of the middle row, 69 m above the plain mean of the pixels."""
    nrow = 2000
    d8 = np.full((nrow, 1), 4, dtype=np.uint8)
    d8[nrow - 1, 0] = 0
    upa = (np.arange(1, nrow + 1, dtype=np.float32) * np.float32(0.6)).reshape(nrow, 1)
    elevation = np.arange(nrow, dtype=np.float32).reshape(nrow, 1)
    fm = flowmorph.FlowMorph(d8, upa, (10.0, 0.01, 0.0, 60.0, 0.0, -0.01), elevation=elevation, min_upstream_area_km2=10.0)
    terrain = fm.upstream_terrain()
    # the weights: the area of every row integrated here, not the package's own, kept as float32 as the package keeps it
    area = area_of_rows(60.0, -0.01, 0.01, nrow).astype(np.float32).astype(np.float64)
    z = np.arange(nrow, dtype=np.float64)
    mean = (area * z).sum() / area.sum()
    std = np.sqrt((area * (z - mean) ** 2).sum() / area.sum())
    assert float(terrain["mean_elv"][nrow - 1, 0]) == pytest.approx(mean, rel=1e-6)
    assert float(terrain["elv_std"][nrow - 1, 0]) == pytest.approx(std, rel=1e-6)
    assert 68.0 < mean - z.mean() < 70.0                       # the plain mean of the pixels is 999.5
    assert float(terrain["min_elv"][nrow - 1, 0]) == 0.0 and float(terrain["max_elv"][nrow - 1, 0]) == nrow - 1
    # the package's areas are those of the WGS84 ellipsoid at the rows' own latitudes, one value a row
    own = fm.pixel_area()
    assert own.dtype == np.float32 and own.shape == (nrow, 1)
    np.testing.assert_allclose(own[:, 0].astype(np.float64), area_of_rows(60.0, -0.01, 0.01, nrow), rtol=2e-7, atol=0)
    assert area[nrow // 2] == pytest.approx(7.96e5, rel=5e-3)   # a pixel of 0.01 degree at 50 N is 0.796 km2


def test_a_network_pixel_without_a_slope_is_refused():
    """Every network pixel needs a finite slope other than -9999, as it needs a valid elevation."""
    from flowmorph.terrain import upstream_terrain
    d8 = np.full((4, 1), 4, dtype=np.uint8)
    d8[3, 0] = 0
    idxs_ds = downstream_index(d8)
    seq, upg = dfs_sequence(idxs_ds, np.ones(4, dtype=np.float32))
    elevation = np.array([40.0, 30.0, 20.0, 10.0], dtype=np.float32)
    area = np.ones(4, dtype=np.float32)
    target = np.ones(4, dtype=bool)
    whole = upstream_terrain(seq, upg, idxs_ds, elevation, np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32), target, area)
    assert float(whole["mean_slp"][3]) == pytest.approx(2.5)
    for missing in (-9999.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="no finite slope"):
            upstream_terrain(seq, upg, idxs_ds, elevation, np.array([4.0, missing, 2.0, 1.0], dtype=np.float32), target,
                             area)


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
