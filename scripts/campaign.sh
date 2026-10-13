#!/bin/bash
# Full run campaign (sequential, 4 cores).
cd "$(dirname "$0")/.."
python3 scripts/run_all.py coarse surge 6,4,2,1 900
python3 scripts/run_all.py coarse sway,heave 6,4,2,1 1300
python3 scripts/run_all.py medium surge 6 900
python3 scripts/run_all.py medium sway,heave 6 1300
python3 scripts/run_all.py fine surge 6 900
echo CAMPAIGN_DONE
