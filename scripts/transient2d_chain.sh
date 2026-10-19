#!/bin/bash
# Resumable sequential chain (re-run after a restart: finished planes are skipped via cases2d/done_<plane>).
cd "$(dirname "$0")/.."
if [ ! -f cases2d/done_sway_z0 ] && [ -d cases2d/sway_2d ]; then
  xvfb-run -a python3 scripts/transient2d.py video cases2d/sway_2d sway_z0 && touch cases2d/done_sway_z0
fi
for p in surge_x0 surge_y0 sway_y0 heave_x0 heave_z0; do
  [ -f cases2d/done_$p ] || { xvfb-run -a python3 scripts/transient2d.py all $p && touch cases2d/done_$p; }
done
echo CHAIN_DONE
