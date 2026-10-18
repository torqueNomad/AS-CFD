#!/bin/bash
# Sequential chain: re-render sway_z0 with flow lines, then run all remaining 2-D planes one after another.
cd "$(dirname "$0")/.."
xvfb-run -a python3 scripts/transient2d.py video cases2d/sway_2d sway_z0
xvfb-run -a python3 scripts/transient2d.py all surge_x0 surge_y0 sway_y0 heave_x0 heave_z0
echo CHAIN_DONE
