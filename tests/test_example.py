"""The bundled basin against its reference values.

The example is one basin of MERIT Hydro (696 km^2, 110,125 pixels, 2,033 targets), cut out of a larger region, with
reference values from that region, cut to the same window (tests/reference/expected.npz).  Cutting the
window moves its origin, and with it the last bits of some pixel centres, so a value here may differ from the
reference in its last place: the test asks for the same no-data pattern and a relative difference below 5e-7.
"""

import os

import numpy as np
import pytest

import flowmorph

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
    assert int((fm.seq_dfs >= 0).sum()) == 110125
    assert int((fm.idxs_ds == np.arange(fm.idxs_ds.size)).sum()) == 1
    assert int(fm.is_target.sum()) == 2033


@pytest.mark.parametrize("name", ("slp",) + flowmorph.ATTRIBUTE_NAMES)
def test_every_attribute_matches_the_reference(example, expected, name):
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
        written = expected[name] != np.float32(-9999)
        differ = int((attributes[name][written] != expected[name][written]).sum())
        assert differ <= 3, (name, differ)


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


def _d8_and_transform():
    import rasterio
    with rasterio.open(os.path.join(DATA, "dir_example.tif")) as dataset:
        return dataset.read(1), dataset.transform.to_gdal()
