"""The bundled basin against the published FullMorph v1.1.

The example is the basin of the FullBasin paper, the Nanxi River in Fujian (basin 8595 of region 43; 729 km^2, 93,432
pixels, 1,929 targets): MERIT Hydro's flow directions and upstream area on the basin's pixels, and MERIT DEM's
elevation on the whole window, which keeps one pixel of margin so that every slope in the basin sees its eight
neighbours.  tests/reference/expected.npz holds the 16 attributes of FullMorph v1.1 cut from region 43 to this basin:
an attribute must have their no-data pattern and differ from them by less than 5e-7 (relative), and nearly every value
is bit for bit.
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
    with np.load(os.path.join(HERE, "reference", "expected.npz")) as archive:
        return {name: archive[name] for name in archive.files}


def test_the_example_is_one_whole_basin(example):
    fm, _ = example
    assert int((fm.seq_dfs >= 0).sum()) == 93432
    assert int((fm.idxs_ds == np.arange(fm.idxs_ds.size)).sum()) == 1
    assert int(fm.is_target.sum()) == 1929


# At this target the two farthest boundary pixels, (246, 38) and (246, 40), mirror each other across its column and
# are at the same angle from it.  Where the system's cos and sin give them equal cosines, the first in row-major order
# is the farthest, as in the program that wrote FullMorph; where their last bits differ, the other may be.  The length
# is the same; the axis turns and the width with it.
MIRRORED, OTHER_WIDTH = (186, 39), np.float32(4.7943816)


@pytest.mark.parametrize("name", flowmorph.ATTRIBUTE_NAMES)
def test_every_attribute_matches_the_reference(example, expected, name):
    _, attributes = example
    got = attributes[name]
    want = expected[name]
    if name == "basin_width_km" and got[MIRRORED] == OTHER_WIDTH:
        got = got.copy()
        got[MIRRORED] = want[MIRRORED]
    assert got.dtype == np.float32
    assert np.array_equal(got == np.float32(-9999), want == np.float32(-9999))
    written = want != np.float32(-9999)
    np.testing.assert_allclose(got[written], want[written], rtol=5e-7, atol=0)


def test_the_values_are_mostly_bit_for_bit(example, expected):
    """Of the 1,929 values of an attribute at most a few differ in the last bits: the sums of the means and standard
    deviations, and the hull, can round otherwise than in the C program that wrote FullMorph."""
    _, attributes = example
    for name in flowmorph.ATTRIBUTE_NAMES:
        written = expected[name] != np.float32(-9999)
        differ = int((attributes[name][written] != expected[name][written]).sum())
        assert differ <= 5, (name, differ)


def test_the_weighted_terrain_is_what_a_plain_recomputation_gives(example):
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
    assert fm.transform[1] != 1.0 / 1200.0          # the computation transform is rounded to 1e-12


def test_exact_transform_undoes_the_rounding():
    from flowmorph.grid import exact_transform, snap_transform
    exact = (142.99958333333336, 1.0 / 1200.0, 0.0, -39.00041666666667, 0.0, -1.0 / 1200.0)
    assert snap_transform(exact)[1] == 0.000833333333          # the transform rounded to 1e-12
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
