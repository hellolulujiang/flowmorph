"""Command line: the slope and the sixteen upstream attributes of a grid, written as GeoTIFFs.

    python -m flowmorph DIR UPA ELV OUTDIR [--ring GLOBAL_ELV] [--seq-dfs SEQ --upg UPG] [--earth sphere]
                                           [--min-area 10]
"""

import argparse
import time

from .api import MIN_UPSTREAM_AREA_KM2, FlowMorph
from .earth import EARTH_MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m flowmorph", description=__doc__.split("\n\n")[0])
    parser.add_argument("dir", help="MERIT Hydro D8 flow directions (Byte, 247 no data), holding whole basins")
    parser.add_argument("upa", help="upstream area in km2 (MERIT Hydro's upa)")
    parser.add_argument("elv", help="elevation in m on the same grid")
    parser.add_argument("outdir", help="directory the seventeen GeoTIFFs are written to")
    parser.add_argument("--ring", help="a larger elevation grid (the global one) for the slope at the grid's edge")
    parser.add_argument("--seq-dfs", help="FlowTopo's seq_dfs of the grid (made from the flow directions if not "
                                          "given)")
    parser.add_argument("--upg", help="the upstream pixel count that goes with --seq-dfs")
    parser.add_argument("--earth", choices=EARTH_MODELS, default="wgs84",
                        help="the earth of the lengths and hull areas (default wgs84)")
    parser.add_argument("--min-area", type=float, default=MIN_UPSTREAM_AREA_KM2,
                        help="the targets' smallest upstream area in km2 (default 10)")
    arguments = parser.parse_args(argv)
    started = time.time()
    flowmorph = FlowMorph.from_raster(arguments.dir, arguments.upa, arguments.elv, ring_path=arguments.ring,
                                      seq_dfs_path=arguments.seq_dfs, upg_path=arguments.upg, earth=arguments.earth,
                                      min_upstream_area_km2=arguments.min_area)
    paths = flowmorph.write(arguments.outdir)
    print(f"{int((flowmorph.seq_dfs >= 0).sum())} pixels, {int(flowmorph.is_target.sum())} targets; "
          f"{len(paths)} files in {arguments.outdir}, {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
