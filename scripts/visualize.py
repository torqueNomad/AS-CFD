"""CFD flow visualisation for one run (requires reconstructPar output; run under xvfb-run).
Produces, per run:
  flow3d_<tag>.png   : body coloured by pressure coefficient Cp + 3-D streamlines coloured by |U|/V
  slices_<tag>.png   : two mid-plane slices coloured by |U|/V with in-plane streamlines
  cp_<tag>.png       : body surface Cp from four views
Usage: xvfb-run -a python3 visualize.py <run_dir> [<run_dir> ...]"""
import os, sys, json, subprocess, numpy as np
import pyvista as pv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'results', 'figures', 'flow'); os.makedirs(FIG, exist_ok=True)
pv.global_theme.font.color = 'black'
NAMES = ['X (sway)', 'Y (heave)', 'Z (surge)']


def reconstruct(run):
    if not os.path.exists(os.path.join(run, 'constant', 'polyMesh', 'faces')):
        subprocess.run(f'bash -c "source /usr/lib/openfoam/openfoam2512/etc/bashrc; cd {run}; '
                       f'reconstructParMesh -constant > log.reconMesh 2>&1; reconstructPar -latestTime > log.recon 2>&1"',
                       shell=True, check=True)
    open(os.path.join(run, 'case.foam'), 'a').close()


def load(run):
    r = pv.OpenFOAMReader(os.path.join(run, 'case.foam'))
    r.set_active_time_value(r.time_values[-1]); r.enable_all_patch_arrays()
    m = r.read()
    return m['internalMesh'], m['boundary']['body']


def main(run):
    meta = json.load(open(os.path.join(run, 'case.json')))
    reconstruct(run)
    vol, body = load(run)
    V = meta['V']; a = {'sway': 0, 'heave': 1, 'surge': 2}[meta['axis']]
    tag = f"{meta['axis']}_{meta['mesh']}_{meta['kn']:g}kn"
    title = f"{meta['axis'].upper()}  {meta['kn']:g} kn ({V:.2f} m/s)"
    # kinematic pressure p -> Cp = p / (0.5 V^2);  |U|/V
    vol.point_data['U/V'] = np.linalg.norm(vol.point_data['U'], axis=1) / V
    vol.cell_data['U/V'] = np.linalg.norm(vol.cell_data['U'], axis=1) / V
    body = body.cell_data_to_point_data()
    body.point_data['Cp'] = body.point_data['p'] / (0.5 * V ** 2)
    ext = np.array(meta['extents'])
    # crop the volume to a region around the body to speed up filters
    lo = -ext / 2 - 1.5; hi = ext / 2 + 1.5; lo[a] -= 2.5 * ext[a] if a != 2 else 0.8 * ext[a]
    roi = vol.clip_box([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]], invert=False)
    roi_p = roi.cell_data_to_point_data()

    cmapU, cmapP = 'turbo', 'coolwarm'
    cp_rng = [-0.8, 1.0]
    sargs_u = dict(title='|U| / V', n_labels=5, fmt='%.2f', vertical=True, position_x=0.90, position_y=0.25, height=0.5)
    sargs_p = dict(title='Cp', n_labels=5, fmt='%.2f', vertical=True, position_x=0.03, position_y=0.25, height=0.5)

    # ---- 1. 3-D streamlines + Cp body
    seed_c = np.zeros(3); seed_c[a] = ext[a] / 2 + 0.6
    others = [i for i in range(3) if i != a]
    # seed two elliptical rings upstream: one inside the silhouette (wraps around the body), one just outside
    g = []
    for sc in (0.6, 1.1):
        for t in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            p = seed_c.copy(); p[others[0]] = sc * ext[others[0]] / 2 * np.cos(t); p[others[1]] = sc * ext[others[1]] / 2 * np.sin(t); g.append(p)
    src = pv.PolyData(np.array(g))
    sl = roi_p.streamlines_from_source(src, vectors='U', integration_direction='forward',
                                       max_steps=6000, max_length=4 * ext[a] + 15, initial_step_length=0.2)
    pl = pv.Plotter(off_screen=True, window_size=(1800, 1000))
    pl.set_background('white')
    pl.add_mesh(body, scalars='Cp', cmap=cmapP, clim=cp_rng, smooth_shading=True, scalar_bar_args=sargs_p)
    # horizontal (or vertical for heave) mid-plane through the body as a flow-speed backdrop
    bn = 1 if a != 1 else 0
    back = roi_p.slice(normal=np.eye(3)[bn], origin=(0, 0, 0))
    pl.add_mesh(back, scalars='U/V', cmap=cmapU, clim=[0, 1.3], lighting=False, show_scalar_bar=False)
    if sl.n_points:
        pl.add_mesh(sl.tube(radius=0.01), scalars='U/V', cmap=cmapU, clim=[0, 1.3], scalar_bar_args=sargs_u)
    pl.add_text(title + '  —  streamlines coloured by |U|/V, body by Cp', font_size=12)
    cam = {'surge': [(9, 6, 10), (0, 0, -2), (0, 1, 0)], 'sway': [(10, 6, 8), (-2, 0, 0), (0, 1, 0)],
           'heave': [(9, 9, 7), (0, -1.5, 0), (0, 0, 1)]}[meta['axis']]
    pl.camera_position = cam; pl.camera.zoom(1.1)
    pl.screenshot(os.path.join(FIG, f'flow3d_{tag}.png')); pl.close()

    # ---- 2. mid-plane slices with surface-LIC-like streamlines (in-plane)
    pl = pv.Plotter(off_screen=True, shape=(2, 1), window_size=(1800, 1300))
    pl.set_background('white')
    for k, n_ax in enumerate(others):
        normal = np.eye(3)[n_ax]
        s = roi_p.slice(normal=normal, origin=(0, 0, 0))
        pl.subplot(k, 0)
        pl.add_mesh(s, scalars='U/V', cmap=cmapU, clim=[0, 1.3], scalar_bar_args=dict(sargs_u, title=f'|U|/V '))
        # in-plane streamlines: project velocity onto plane, seed along an upstream line
        s.point_data['Uin'] = s.point_data['U'] - np.outer(s.point_data['U'] @ normal, normal)
        line_dir = [i for i in others if i != n_ax][0]
        a0 = np.zeros(3); a0[a] = ext[a] / 2 + 1.2; a0[line_dir] = -ext[line_dir] / 2 - 1.0
        a1 = a0.copy(); a1[line_dir] = ext[line_dir] / 2 + 1.0
        seeds = pv.Line(a0, a1, resolution=40)
        st = s.streamlines_from_source(seeds, vectors='Uin', integration_direction='forward', surface_streamlines=True,
                                       max_steps=8000, max_length=4 * ext[a] + 12, initial_step_length=0.05)
        if st.n_points:
            pl.add_mesh(st, color='white', line_width=1.2, opacity=0.8)
        bs = body.slice(normal=normal, origin=(0, 0, 0))
        pl.add_mesh(bs, color='black', line_width=3)
        pl.add_text(f'{title}  —  slice normal to {NAMES[n_ax]}', font_size=11)
        pl.view_vector(normal, viewup=np.eye(3)[1] if n_ax != 1 else np.eye(3)[0])
        pl.reset_camera(bounds=[*(np.ravel(list(zip(lo + 1.0, hi - 1.0))))]); pl.camera.zoom(1.0)
    pl.screenshot(os.path.join(FIG, f'slices_{tag}.png')); pl.close()

    # ---- 3. Cp on body, four views
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1800, 1100))
    pl.set_background('white')
    views = [((1, 0.6, 1), 'iso bow'), ((-1, 0.6, -1), 'iso stern'), ((0, 1, 0.001), 'top (+Y)'), ((1, 0, 0.001), 'side (+X)')]
    for i, (vv, nm) in enumerate(views):
        pl.subplot(i // 2, i % 2)
        pl.add_mesh(body.copy(), scalars='Cp', cmap=cmapP, clim=cp_rng, smooth_shading=True,
                    scalar_bar_args=dict(title='Cp', vertical=False, position_x=0.25, position_y=0.04, width=0.5),
                    show_scalar_bar=(i == 0))
        pl.add_text(f'{title}: {nm}', font_size=10)
        pl.view_vector(vv, viewup=(0, 1, 0) if nm != 'top (+Y)' else (1, 0, 0)); pl.reset_camera(); pl.camera.zoom(1.3)
    pl.screenshot(os.path.join(FIG, f'cp_{tag}.png')); pl.close()
    print('wrote', tag)


if __name__ == '__main__':
    for r in sys.argv[1:]:
        main(os.path.abspath(r))
