# Shell CFD: drag and hydraulic power in surge, sway and heave

This file explains how the results in `results/` were produced, what each source file does, and every assumption made along the way.

## 1. Pipeline overview

```
Shell.STL (CAD, mm, 294 open panels)
   │  scripts/make_envelope.py   → seal holes/openings, cap ends, convert to metres
   ▼
geometry/shell_envelope.stl (watertight, 352 800 triangles)
   │  scripts/setup_case.py      → writes the OpenFOAM case for one axis, mesh level and speed
   │  scripts/mesh_case.sh       → blockMesh + snappyHexMesh (+ boundary layers) + checkMesh
   │  scripts/run_case.sh        → simpleFoam (steady RANS, k-ω SST), 4 MPI ranks
   │  scripts/run_all.py         → loop: 1 mesh per axis, 1 run per speed
   │  scripts/campaign.sh        → the exact run list that was executed
   ▼
cases/<axis>_<mesh>/V<kn>kn/postProcessing/forces/0/force.dat
   │  scripts/postprocess.py     → averages forces, computes Cd and P = D·V, fits Cd(Re), writes tables and plots
   ▼
results/runs_raw.csv, results/drag_power_table.csv, results/fit_coefficients.json, results/figures/*.png
```

Software: OpenFOAM v2512 (ESI/OpenCFD, from the official `dl.openfoam.com` apt repository), Python 3 with trimesh, numpy, scipy, scikit-image and matplotlib.

To reproduce everything:
```bash
apt-get install openfoam2512        # after: curl -s https://dl.openfoam.com/add-debian-repo.sh | bash
pip install trimesh numpy scipy scikit-image matplotlib rtree
python3 scripts/make_envelope.py geometry/Shell_original.stl geometry/shell_envelope.stl
bash scripts/campaign.sh           # ~8 h on 4 cores
python3 scripts/postprocess.py
```

## 2. Source files

### `scripts/make_envelope.py`: geometry sealing
The CAD file is not a solid. It is an assembly of 294 separate panels with 2 mm thick skin. It has open ends, ~300 mm side cut-outs near both ends, through-holes, and internal cross-members. A CFD mesher needs one closed outer surface, and you asked for all holes to be treated as closed (blind).

What the script does:
1. It slices the original mesh every 10 mm along Z (the long axis).
2. For each slice it takes polar coordinates about the body's centreline, using 240 angular bins of 1.5°. It keeps the **largest** radius in each bin, which is the outer skin. Internal members are ignored.
3. Bins with no skin (openings) are filled in two steps: first by linear interpolation along Z from neighbouring slices, then around the circumference.
4. It applies light Gaussian smoothing (σ = 0.8 bin) and lofts the radius grid into a triangle surface.
5. It closes both ends with flat caps and writes the result in metres.

Result: a watertight envelope with volume 7.23 m³ and wetted area 27.56 m². Its bounding box matches the CAD to within 1 mm. About 2.3 % of the outer surface was openings that had to be bridged. `results/figures/envelope_sections_z.png` and `envelope_sections_long.png` overlay the envelope (red) on the original (black).

### `scripts/setup_case.py`: OpenFOAM case generator
`setup_case.py <case> <surge|sway|heave> <coarse|medium|fine> <speed_kn> [iterations]` writes a complete case:
- **Frame and direction.** The body is translated so its bounding-box centre is at the origin. Surge is +Z, sway is +X and heave is +Y. The body moves in the + direction, so the inflow is U = −V·ê.
- **Domain (`blockMeshDict`).** The box extends 1.5 L or 8 m upstream (whichever is larger), 3 L or 15 m downstream, and 6 m beyond the body on each side. Patches are `inlet`, `outlet`, `sides` and `body`.
- **Mesh (`snappyHexMeshDict`).** Background cells are 0.64 m (coarse), 0.48 m (medium) or 0.36 m (fine). The body surface is refined to level 5, giving 20, 15 or 11 mm cells, and feature edges to level 6. There is a near-body box at level 2 and a wake box at level 3 that extends about 0.6 L downstream. Three prism layers are grown on the body (expansion ratio 1.2, final layer 0.35 of the local cell).
- **Projected area.** This is the silhouette area normal to the motion axis, computed by rasterising the surface at 4 mm. It is used only to make Cd dimensionless and does not affect the forces.
- **Physics files.** These cover sea-water properties, the k-ω SST model, boundary conditions, numerical schemes, solver settings and the function objects that record forces, y+ and residuals.

### `scripts/mesh_case.sh` and `scripts/run_case.sh`
These are thin wrappers that source the OpenFOAM v2512 environment. `mesh_case.sh` runs `blockMesh`, `surfaceFeatureExtract`, `decomposePar` (4 ranks, hierarchical), `snappyHexMesh -parallel` and `checkMesh`. `run_case.sh` runs `simpleFoam -parallel`.

### `scripts/run_all.py` and `scripts/campaign.sh`
`run_all.py` generates and meshes once per axis and mesh level. It then makes one run directory per speed that reuses the mesh, and marks finished runs with a `DONE` file so the batch can be restarted. `campaign.sh` is the exact list of runs executed.

### `scripts/postprocess.py`
- Reads `force.dat`, which holds the pressure and viscous force on the body each iteration. It projects the force on the free-stream direction to get the drag D.
- Averages D over the **last 30 %** of iterations. The standard deviation over the same window is reported as `drag_std_N`.
- Computes Cd = D / (½ ρ V² A_proj), P = D·V and Re = V·L_axis/ν.
- Fits Cd = c0 + c1·log10(Re) to each axis's speed sweep. It multiplies by the grid correction (medium-mesh D / coarse-mesh D at 6 kn) and evaluates the fit from 0.5 to 6 kn in 0.5 kn steps.
- Writes the CSV and JSON tables and the plots.

### `openfoam_case_example/`
A copy of the dictionaries (`system/`, `constant/`, `0.orig/`) of the surge, 6 kn, coarse-mesh run, for inspection without running Python.

## 3. Assumptions (and why)

| # | Assumption | Reason / effect |
|---|---|---|
| A1 | **All openings, holes and both open ends are closed (blind) and flush with the surrounding skin.** End openings are capped flat. | Your instruction. Real open holes would add drag, through internal flow and cavity drag, which is not captured. |
| A2 | **The outer surface between known skin is interpolated** (2.3 % of the area), including the ~300 mm side cut-outs near the nose. | The CAD has no skin there, so a smooth bridge is the neutral choice. |
| A3 | **Small features below ~20 mm (coarse mesh) are smoothed out**, such as panel joints and fastener heads. | The surface resolution is 11–20 mm. They usually add a few % of drag, which is not captured. |
| A4 | **Fully submerged, deep and far from walls**, with no free surface, no waves and no seabed. | Your instruction. There is no wave-making drag. Results hold where depth is greater than about 3× the body height. |
| A5 | **Standard sea water at 15 °C, 35 g/kg:** ρ = 1025.9 kg/m³, ν = 1.1883×10⁻⁶ m²/s, from ITTC Recommended Procedure 7.5-02-01-03 (2011). | Your instruction ("standard sea water"). Drag scales with ρ, and the viscous part changes about 2–3 % per 5 °C. |
| A6 | **Steady, constant speed, pure translation along one axis at a time.** No combined motion, no drift angle, no acceleration. | Your request was "in all 3 axes". Added mass and inertia are not included, so the forces apply only at constant velocity. |
| A7 | **Motion direction is +Z, +X, +Y.** The rounded end at high Z is taken as the bow. | Derived from the CAD frame. Sway and heave are close to symmetric. Surge in reverse would put the blunt flat stern first and give higher drag. |
| A8 | **Steady RANS with the k-ω SST turbulence model and wall functions** (`nutUSpaldingWallFunction`, valid for any y+). | The industry standard for submerged bodies at Re 10⁶–10⁷. Accuracy is typically ±5–10 % for attached flow and ±15–30 % for bluff, separated flow. |
| A9 | **Fully turbulent boundary layer from the nose.** | k-ω SST without a transition model. It slightly over-predicts friction at 1 kn (Re 3×10⁶, where some laminar flow is possible). |
| A10 | **Smooth hydraulic surface**, with no roughness and no fouling. | Fouling can raise friction drag by 10–50 % or more. Add a margin for in-service conditions. |
| A11 | **Inflow turbulence intensity 1 %, eddy-viscosity ratio 10.** | Calm sea and tow-tank-like conditions, with little effect on drag. |
| A12 | **Sway and heave flows are unsteady in reality** (vortex shedding). The steady solver shows force oscillations, so the reported drag is the average over the last 30 % of iterations and the spread is listed as `drag_std_N`. | Steady RANS on bluff cross-flow captures the mean drag only approximately. URANS or DES would be needed for shedding forces. |
| A13 | **Hydraulic power = drag × speed** (power delivered to the water). | Shaft or electrical power = P / (propulsive efficiency × motor efficiency), typically ÷0.3–0.6 for small thrusters. |
| A14 | **Intermediate speeds come from a Cd fit.** CFD was run at 1, 2, 4 and 6 kn. Other speeds come from Cd = c0 + c1·log10(Re) per axis. | Cd varies smoothly and weakly with Re at these Re. The fit error at the computed points is listed in the report. |
| A15 | **Grid correction.** The speed sweep uses the coarse mesh, scaled by the medium/coarse drag ratio at 6 kn for each axis. A fine mesh was run for surge. | Limited by the 4-core machine. The coarse, medium and fine spread is reported as discretisation uncertainty. |
| A16 | **Domain blockage** is 0.6 % (surge), 2.3 % (sway) and about 3.8 % (heave), with slip side walls. | Small. Not corrected, and it may over-predict sway and heave drag by up to about 3–5 %. |
| A17 | **Only the body's own drag.** No thrusters, appendages, tether or cable, or interaction effects. | These are not in the CAD. |

## 4. Validation checks performed
- **Mesh quality:** `checkMesh` reports "Mesh OK" for every mesh, and prism layers cover more than 98.9 % of the body (`log.checkMesh`, `log.snappyHexMesh`).
- **Convergence:** all residuals and forces were monitored, and one plot per run is in `results/figures/conv_*.png`. Surge runs met the residual targets (p < 1e-5, U, k and ω < 1e-6) in about 530 iterations, with a drag spread under 0.1 %.
- **y+ on the body** is listed per run in `runs_raw.csv`.
- **Independent friction check:** the CFD viscous drag in surge is compared with the ITTC-1957 line, C_F = 0.075/(log10 Re − 2)², × ½ρV²S with S = 27.56 m².
- **Mesh sensitivity:** coarse, medium and fine meshes for surge, and coarse and medium for sway and heave, all at 6 kn.

## 5. y+ (wall resolution)
`scripts/yplus_report.py` reads the OpenFOAM `yPlus` field on the body (first-cell-centre y+) for every run. It writes:
- `results/yplus_summary.csv`: minimum, maximum, area-weighted mean, the 5th/50th/95th area-weighted percentiles, and the % of wetted area in each wall region (<5 viscous sublayer, 5–30 buffer, 30–300 log layer, >300).
- `results/figures/yplus/hist_<axis>_<mesh>.png`: area-weighted y+ histograms, one curve per speed.
- `results/figures/yplus/yplus_<axis>_<mesh>_<kn>kn.png`: y+ contours on the body at 1 and 6 kn.

How to read it: wall functions are designed for 30 < y+ < 300, and `nutUSpaldingWallFunction` (Spalding's continuous law of the wall) also stays valid in the buffer and viscous regions. y+ rises roughly in proportion to speed. At 6 kn in sway, part of the windward face exceeds y+ 300. At Re about 4×10⁶ the log layer extends beyond y+ 300, so this is acceptable, and the medium-mesh run (cells 25 % smaller) quantifies its effect on drag.
