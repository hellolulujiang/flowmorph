"""The bundled basin against the C code of FlowMorph 1.1.

The example is one basin of MERIT Hydro (696 km^2, 110,125 pixels, 2,033 targets) cut from region 5699 of the
1400deg2 partition, with the region's C outputs cut to the same window (tests/reference/expected_c.npz), written by
an earlier state of the C code on MERIT Hydro's elevation, the bundled one.  On the region's own tiles the package
gives the C's numbers bit for bit (README, "Same numbers as the C code").

The C code has changed two groups of layers since the reference was made.  WEIGHTED (below): the means and standard
deviations are weighted by pixel area, where the reference counts every pixel once; they are checked against a
plain recomputation.  PERIMETER (below): the perimeter is taken on cells close to square on the ground;
tests/reference/expected_c_perimeter.npz holds the C code's perimeter of every target of this window (made by
tests/reference/make_expected_c_perimeter.py), and the package must give it bit for bit; the three ratios made from
the perimeter are checked against the reference through the change of the perimeter.  Cutting the window moves its
origin, and with it the last bits of some pixel centres, so here a value may differ from the C's in its last place:
the test asks for the same no-data pattern and a relative difference below 5e-7, but for the five layers of WEIGHTED
and the three ratios of PERIMETER (to 1e-6 through the change of the perimeter).
"""

import math
import os

import numpy as np
import pytest

import flowmorph
from ellipsoid_area import area_of_rows

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")


@pytest.fixture(scope="module")
def example():
    fm = flowmorph.FlowMorph.from_raster(os.path.join(DATA, "dir_example.tif"), os.path.join(DATA, "upa_example.tif"),
                                         os.path.join(DATA, "elv_example.tif"))
    return fm, fm.attributes()


@pytest.fixture(scope="module")
def expected():
    with np.load(os.path.join(HERE, "reference", "expected_c.npz")) as archive:
        return {name: archive[name] for name in archive.files}


def test_the_example_is_one_whole_basin(example):
    fm, _ = example
    assert int((fm.seq_dfs >= 0).sum()) == 110125
    assert int((fm.idxs_ds == np.arange(fm.idxs_ds.size)).sum()) == 1
    assert int(fm.is_target.sum()) == 2033


# the four means and standard deviations of step 2, and the hypsometric integral made from the mean elevation, are
# weighted by pixel area; the reference file counts every pixel once.  They are checked against a plain
# recomputation below, and against the reference only for how little a basin of half a degree changes.
WEIGHTED = ("mean_elv", "elv_std", "mean_slp", "slp_std", "hypsometric_integral")
# the perimeter and the three ratios made from it: the reference holds them as an earlier perimeter gave them
PERIMETER = ("perimeter_km", "gravelius", "circularity", "lemniscate_ratio")


@pytest.mark.parametrize("name", [n for n in ("slp",) + flowmorph.ATTRIBUTE_NAMES if n not in WEIGHTED + PERIMETER])
def test_every_attribute_matches_the_c(example, expected, name):
    fm, attributes = example
    got = attributes[name]
    want = expected[name]
    network = (fm.seq_dfs >= 0).reshape(fm.shape)
    if name == "slp":
        got = np.where(network, got, np.float32(-9999))
    assert got.dtype == np.float32
    assert np.array_equal(got == np.float32(-9999), want == np.float32(-9999))
    written = want != np.float32(-9999)
    np.testing.assert_allclose(got[written], want[written], rtol=5e-7, atol=0)


def test_the_values_are_mostly_bit_for_bit(example, expected):
    _, attributes = example
    for name in flowmorph.ATTRIBUTE_NAMES:
        if name in WEIGHTED + PERIMETER:
            continue
        written = expected[name] != np.float32(-9999)
        differ = int((attributes[name][written] != expected[name][written]).sum())
        assert differ <= 3, (name, differ)


def test_the_perimeter_is_the_c_codes_bit_for_bit(example):
    _, attributes = example
    with np.load(os.path.join(HERE, "reference", "expected_c_perimeter.npz")) as archive:
        want = archive["perimeter_km"]
    got = attributes["perimeter_km"]
    assert got.dtype == np.float32 and int((want != np.float32(-9999)).sum()) == 2033
    assert np.array_equal(got.view(np.int32), want.view(np.int32))


def test_the_perimeter_in_metres_is_the_c_codes_bit_for_bit(example):
    """The doubles before the division by 1000 and the cast to float32: every target's catchment taken as step 3
    takes it (its boundary pixels give the start, the last row and the cap), measured by flowmorph.vscells, against
    the C code's metres of the same catchment."""
    from flowmorph import vscells
    from flowmorph.grid import pixel_edges_deg
    fm, _ = example
    with np.load(os.path.join(HERE, "reference", "expected_c_perimeter.npz")) as archive:
        want, cells = archive["perimeter_m"], archive["target_cell"]
    nrow, ncol = fm.shape
    seq = np.ascontiguousarray(fm.seq_dfs, dtype=np.int32).reshape(-1)
    upg = fm.upg.reshape(-1)
    t = fm.transform
    width, height = pixel_edges_deg(t)
    col0 = vscells.global_col_of_col0(t[0], width)
    got = np.zeros_like(want)
    for index, cell in enumerate(cells):
        last = int(seq[cell])
        first = last - int(upg[cell]) + 1
        inside = ((seq >= first) & (seq <= last)).reshape(nrow, ncol)
        padded = np.pad(inside, 1)
        all_in = np.ones_like(inside)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    all_in &= padded[1 + dr:1 + dr + nrow, 1 + dc:1 + dc + ncol]
        boundary = np.flatnonzero((inside & ~all_in).reshape(-1))
        if boundary.size < 3:
            continue
        start, end = int(boundary.min()), int(boundary.max())
        status, got[index] = vscells.perimeter_m(seq, nrow, ncol, first, last, float(t[3]), float(t[5]), col0, width,
                                                 height, start // ncol, end // ncol, start // ncol, start % ncol,
                                                 10 * boundary.size + 100, fm._model)
        assert status == vscells.VS_OK
    assert np.array_equal(got.view(np.int64), want.view(np.int64))


def test_the_ratios_follow_the_perimeter(example, expected):
    """gravelius = P / (2 sqrt(pi A)), circularity = 4 pi A / P^2 and lemniscate_ratio = P_lemniscate / P: against
    the C code's values of the reference, each scaled by the change of the perimeter (new / old, and its inverse
    square or inverse), to float32 rounding."""
    _, attributes = example
    written = expected["perimeter_km"] != np.float32(-9999)
    scale = attributes["perimeter_km"][written].astype(np.float64) / expected["perimeter_km"][written].astype(np.float64)
    for name, power in (("gravelius", 1.0), ("circularity", -2.0), ("lemniscate_ratio", -1.0)):
        old = expected[name][written]
        there = old != np.float32(-9999)
        assert np.array_equal(attributes[name][written] == np.float32(-9999), ~there), name
        np.testing.assert_allclose(attributes[name][written][there], old[there] * scale[there] ** power, rtol=1e-6,
                                   atol=0, err_msg=name)
    # this basin lies at 42 S, where a pixel is 1.35 times as tall as it is wide: the perimeter is some 4 to 7 % longer
    # than the earlier perimeter of the reference (the shortest polygon between the centres inside and outside)
    assert 1.03 < float(np.median(scale)) < 1.08


def test_the_weighted_terrain_is_what_a_plain_recomputation_gives(example, expected):
    """For 80 targets from the smallest to the outlet: the pixels of the catchment are the ranks
    [seq - upg + 1, seq]; their elevations and slopes averaged in float64 with the pixel areas as weights -- the
    area of every row integrated in the test (tests/ellipsoid_area.py) and kept as float32, as the package keeps
    it -- and the standard deviation taken from the deviations from that mean."""
    fm, attributes = example
    seq = fm.seq_dfs.reshape(-1)
    cells = np.flatnonzero(seq >= 0)
    cell_of_rank = np.empty(cells.size, dtype=np.int64)
    cell_of_rank[seq[cells]] = cells
    nrow, ncol = fm.shape
    area_of_row = area_of_rows(fm.transform[3], fm.transform[5], fm.transform[1], nrow).astype(np.float32)
    np.testing.assert_allclose(fm.pixel_area()[:, 0].astype(np.float64), area_of_row.astype(np.float64), rtol=2e-7, atol=0)
    area = np.repeat(area_of_row.astype(np.float64), ncol)
    elevation = fm.elevation.reshape(-1).astype(np.float64)
    slope = fm.local_slope().reshape(-1).astype(np.float64)
    targets = np.flatnonzero(fm.is_target.reshape(-1))
    targets = targets[np.argsort(fm.upg.reshape(-1)[targets], kind="stable")]
    chosen = targets[np.unique(np.linspace(0, targets.size - 1, 80).astype(np.int64))]
    for cell in chosen:
        last = int(seq[cell])
        members = cell_of_rank[last - int(fm.upg.reshape(-1)[cell]) + 1:last + 1]
        a = area[members]
        for values, mean_name, std_name in ((elevation[members], "mean_elv", "elv_std"), (slope[members], "mean_slp", "slp_std")):
            mean = (a * values).sum() / a.sum()
            std = np.sqrt((a * (values - mean) ** 2).sum() / a.sum())
            assert float(attributes[mean_name].reshape(-1)[cell]) == pytest.approx(mean, rel=2e-6), (mean_name, cell)
            assert float(attributes[std_name].reshape(-1)[cell]) == pytest.approx(std, rel=2e-4, abs=1e-4), (std_name, cell)
        low, high = float(attributes["min_elv"].reshape(-1)[cell]), float(attributes["max_elv"].reshape(-1)[cell])
        if high > low:
            integral = ((a * elevation[members]).sum() / a.sum() - low) / (high - low)
            assert float(attributes["hypsometric_integral"].reshape(-1)[cell]) == pytest.approx(integral, rel=1e-4, abs=1e-6)
    # this basin lies within half a degree of latitude: the weighted values are close to those of before
    for name, limit in (("mean_elv", 0.5), ("elv_std", 0.5), ("mean_slp", 0.02), ("slp_std", 0.02), ("hypsometric_integral", 1e-3)):
        written = expected[name] != np.float32(-9999)
        assert np.array_equal(attributes[name] == np.float32(-9999), ~written), name
        assert float(np.max(np.abs(attributes[name][written] - expected[name][written]))) < limit, name


def test_the_index_given_or_made_gives_the_same_attributes(example):
    fm, attributes = example
    d8, transform = _d8_and_transform()
    given = flowmorph.FlowMorph(d8, fm.upa, transform, elevation=fm.elevation, seq_dfs=fm.seq_dfs, upg=fm.upg)
    for name, values in given.attributes().items():
        assert np.array_equal(values, attributes[name]), name


def test_the_sphere_gives_other_lengths(example):
    fm, attributes = example
    sphere = flowmorph.FlowMorph.from_raster(
        os.path.join(DATA, "dir_example.tif"), os.path.join(DATA, "upa_example.tif"),
        os.path.join(DATA, "elv_example.tif"), earth="sphere")
    lengths = sphere.boundary_walk()["basin_length_km"]
    written = attributes["basin_length_km"] != np.float32(-9999)
    ratio = lengths[written] / attributes["basin_length_km"][written]
    assert np.all(np.abs(ratio - 1.0) < 0.01)
    assert not np.array_equal(lengths[written], attributes["basin_length_km"][written])


def test_writing_and_reading_back(example, tmp_path):
    fm, attributes = example
    paths = fm.write(str(tmp_path), names=["convexity", "slp"])
    assert sorted(os.path.basename(path) for path in paths) == ["convexity.tif", "slp.tif"]
    import rasterio
    with rasterio.open(paths[0]) as dataset:
        assert dataset.nodata == -9999.0
        assert np.array_equal(dataset.read(1), attributes["convexity"])
        # the header on the exact grid, as the input's, not the transform rounded to 1e-12
        written = dataset.transform.to_gdal()
    with rasterio.open(os.path.join(DATA, "dir_example.tif")) as dataset:
        given = dataset.transform.to_gdal()
    assert written[1] == 1.0 / 1200.0 and written[5] == -1.0 / 1200.0
    assert written[2] == 0.0 and written[4] == 0.0
    for term in (0, 3):                              # the origin on the grid of half pixels, to 4 units in the last place
        assert abs(written[term] - given[term]) <= 4 * math.ulp(given[term])
    assert fm.transform[1] != 1.0 / 1200.0          # what the computations take, rounded as the C code's


def test_exact_transform_undoes_the_rounding():
    from flowmorph.grid import exact_transform, snap_transform
    exact = (142.99958333333336, 1.0 / 1200.0, 0.0, -39.00041666666667, 0.0, -1.0 / 1200.0)
    assert snap_transform(exact)[1] == 0.000833333333          # the C code's rounding to 1e-12
    for rounded in (snap_transform(exact), (142.999583333333, 0.000833333333, 0.0, -39.000416666667, 0.0,
                                            -0.000833333333)):
        back = exact_transform(rounded)
        assert back[1] == 1.0 / 1200.0 and back[5] == -1.0 / 1200.0
        assert abs(back[0] - exact[0]) <= 4 * math.ulp(exact[0]) and abs(back[3] - exact[3]) <= 4 * math.ulp(exact[3])
    assert exact_transform((-100.0, 0.0015, 0.0, 40.0, 0.0, -0.0015)) == (-100.0, 0.0015, 0.0, 40.0, 0.0, -0.0015)


def _d8_and_transform():
    import rasterio
    with rasterio.open(os.path.join(DATA, "dir_example.tif")) as dataset:
        return dataset.read(1), dataset.transform.to_gdal()
