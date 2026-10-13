"""CFD flow visualisation for one run (requires reconstructPar output; run under xvfb-run).
The CFD solution is first interpolated (trilinear, from cell-centre -> point data) onto fine uniform grids:
3-D grid of ~4-5 M points for streamlines, 2-D grids at <= 1 cm spacing for slices. Streamlines are integrated
with adaptive Runge-Kutta 4-5, which gives smooth curves. Nothing here changes the CFD result; it is display only.
Produces, per run (all anti-aliased, 2400 px wide or 200 dpi):
  flow3d_<tag>.png   : body coloured by pressure coefficient Cp + 3-D streamtubes coloured by |U|/V over a mid-plane
  slice_<plane>_<tag>.png : mid-plane slices, |U|/V contours + streamlines (matplotlib), and Cp on the same planes
  cp_<tag>.png       : body surface Cp from four views
Usage: xvfb-run -a python3 visualize.py <run_dir> [<run_dir> ...]"""
import os, sys, json, subprocess, numpy as np
import pyvista as pv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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


def ring_seeds(c, others, ext, scales, n):
    g = []
    for sc in scales:
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            p = c.copy(); p[others[0]] = sc * ext[others[0]] / 2 * np.cos(t); p[others[1]] = sc * ext[others[1]] / 2 * np.sin(t); g.append(p)
    return pv.PolyData(np.array(g))


def image_grid(lo, hi, npts):
    L = np.maximum(np.array(hi) - np.array(lo), 1e-9)
    h = (np.prod(L[L > 1e-6]) / npts) ** (1 / max(1, (L > 1e-6).sum()))
    dims = np.maximum((L / h).astype(int) + 1, 1)
    return pv.ImageData(dimensions=dims, spacing=L / np.maximum(dims - 1, 1), origin=lo)


def section_polys(body, normal, i, j):
    s = body.slice(normal=normal, origin=(0, 0, 0)).strip()
    polys = []
    for k in range(s.n_cells):
        p = s.get_cell(k).points
        if len(p) > 3: polys.append(p[:, [i, j]])
    return polys


def main(run):
    meta = json.load(open(os.path.join(run, 'case.json')))
    reconstruct(run)
    vol, body = load(run)
    V = meta['V']; a = {'sway': 0, 'heave': 1, 'surge': 2}[meta['axis']]
    tag = f"{meta['axis']}_{meta['mesh']}_{meta['kn']:g}kn"
    title = f"{meta['axis'].capitalize()}, {meta['kn']:g} kn ({V:.2f} m/s)"
    others = [i for i in range(3) if i != a]
    vol = vol.cell_data_to_point_data()
    vol.point_data['UV'] = np.linalg.norm(vol.point_data['U'], axis=1) / V
    vol.point_data['Cp'] = vol.point_data['p'] / (0.5 * V ** 2)
    body = body.cell_data_to_point_data()
    body.point_data['Cp'] = body.point_data['p'] / (0.5 * V ** 2)
    body = body.extract_surface(algorithm=None).compute_normals(auto_orient_normals=True)
    ext = np.array(meta['extents'])
    cp_rng = [-0.8, 1.0]

    # ---------------- 3-D view: resample, streamtubes, mid-plane backdrop
    lo = -ext / 2 - 1.2; hi = ext / 2 + 1.2
    hi[a] = ext[a] / 2 + 1.5; lo[a] = -ext[a] / 2 - (max(3.0, 0.8 * ext[a]) if a == 2 else 2.5 * ext[a] + 1.0)
    img = image_grid(lo, hi, 4.5e6).sample(vol, pass_point_data=False)
    U = img.point_data['U'].copy(); U[img.point_data['vtkValidPointMask'] == 0] = 0; img.point_data['U'] = U
    seed_c = np.zeros(3); seed_c[a] = ext[a] / 2 + 1.0
    src = ring_seeds(seed_c, others, ext, (0.35, 0.65, 0.9, 1.15), 40)
    sl = img.streamlines_from_source(src, vectors='U', integrator_type=45, integration_direction='forward',
                                     initial_step_length=0.2, min_step_length=0.01, max_step_length=0.5,
                                     max_steps=20000, max_error=1e-6, max_length=40.0, terminal_speed=1e-4)
    sl.point_data['UV'] = np.linalg.norm(sl.point_data['U'], axis=1) / V
    pl = pv.Plotter(off_screen=True, window_size=(2400, 1350)); pl.set_background('white')
    pl.enable_anti_aliasing('ssaa')
    # backdrop: horizontal mid-plane for surge/sway; for heave a vertical plane just behind the body (camera is on +x)
    bn = 1 if a != 1 else 0
    back = img.slice(normal=np.eye(3)[bn], origin=(0, 0, 0) if a != 1 else (-ext[0] / 2 - 0.02, 0, 0))
    back = back.threshold(0.5, scalars='vtkValidPointMask')
    pl.add_mesh(back, scalars='UV', cmap='turbo', clim=[0, 1.3], lighting=False, show_scalar_bar=False, opacity=0.95)
    pl.add_mesh(body, scalars='Cp', cmap='coolwarm', clim=cp_rng, smooth_shading=True, specular=0.3, specular_power=20,
                scalar_bar_args=dict(title='Cp (body)', n_labels=5, fmt='%.1f', vertical=True, position_x=0.03,
                                     position_y=0.25, height=0.5, title_font_size=28, label_font_size=24))
    if sl.n_points:
        pl.add_mesh(sl.tube(radius=0.007, n_sides=16), scalars='UV', cmap='turbo', clim=[0, 1.3], smooth_shading=True,
                    scalar_bar_args=dict(title='U / V', n_labels=6, fmt='%.1f', vertical=True, position_x=0.91,
                                         position_y=0.25, height=0.5, title_font_size=28, label_font_size=24))
    pl.add_text(title + ': streamlines coloured by local speed U/V, body by pressure coefficient Cp', font_size=14)
    cam = {'surge': [(9, 6, 10), (0, 0, -2.5), (0, 1, 0)], 'sway': [(8, 7, 9), (-2, 0, 0), (0, 1, 0)],
           'heave': [(9, 8, 8), (0, -1.5, 0), (0, 1, 0)]}[meta['axis']]
    pl.camera_position = cam; pl.camera.zoom(1.15)
    pl.screenshot(os.path.join(FIG, f'flow3d_{tag}.png')); pl.close()

    # ---------------- 2-D slices (matplotlib): two mid-planes containing the flow direction
    lab = ['x [m] (sway)', 'y [m] (heave)', 'z [m] (surge)']
    for n_ax in others:
        j = [k for k in others if k != n_ax][0]   # in-plane cross-flow axis
        slo = np.array(lo, float) + 0.05; shi = np.array(hi, float) - 0.05; slo[n_ax] = shi[n_ax] = 0.0
        g2 = image_grid(slo, shi, 1.6e6).sample(vol, pass_point_data=False)
        nx = g2.dimensions
        shape = tuple(int(d) for d in nx if d > 1)
        # ImageData point order is x fastest; build 2-D arrays (rows = slower axis)
        axes_in = [k for k in range(3) if nx[k] > 1]
        def arr(v): return v.reshape(shape[::-1])  # (slow, fast)
        mask = arr(g2.point_data['vtkValidPointMask']) == 0
        Ug = g2.point_data['U']
        uf, us = arr(Ug[:, axes_in[0]]), arr(Ug[:, axes_in[1]])
        UV = np.ma.array(arr(g2.point_data['UV']), mask=mask)
        Cp = np.ma.array(arr(g2.point_data['Cp']), mask=mask)
        # put every field in [cross-flow index, flow-axis index] order
        def cf_(v): return v if axes_in[0] == a else v.T
        UV, Cp, mask_cf = cf_(UV), cf_(Cp), cf_(mask)
        Ua, Uc = cf_(uf if axes_in[0] == a else us), cf_(us if axes_in[0] == a else uf)
        fco = np.linspace(slo[a], shi[a], UV.shape[1]); cco = np.linspace(slo[j], shi[j], UV.shape[0])
        polys = [np.c_[p[:, axes_in.index(a)], p[:, axes_in.index(j)]] for p in section_polys(body, np.eye(3)[n_ax], axes_in[0], axes_in[1])]
        Lf, Lc = shi[a] - slo[a], shi[j] - slo[j]
        horiz = Lc <= 1.2 * Lf   # flow horizontal (left -> right) unless the cross-flow extent is much longer
        if horiz:   # x = downstream distance (-a), y = cross-flow
            xh, yv = -fco[::-1], cco
            fields = [UV[:, ::-1], Cp[:, ::-1]]; Ux, Uy = (-Ua)[:, ::-1], Uc[:, ::-1]
            pmap = lambda p: np.c_[-p[:, 0], p[:, 1]]
            xl, yl = '−' + lab[a] + '   (flow left → right)', lab[j]
            asp = Lc / Lf
        else:       # x = cross-flow, y = flow axis (flow top -> bottom)
            xh, yv = cco, fco
            fields = [UV.T, Cp.T]; Ux, Uy = Uc.T, Ua.T
            pmap = lambda p: np.c_[p[:, 1], p[:, 0]]
            xl, yl = lab[j], lab[a] + '   (flow top → bottom)'
            asp = Lf / Lc
        fig, axs = plt.subplots(2, 1, figsize=(16, 2 * 14 * asp + 3.4), dpi=200, layout='constrained')
        msk = np.ma.getmaskarray(fields[0])
        for ax_, F, cmap, clim, lbl in [(axs[0], fields[0], 'turbo', (0, 1.3), '|U| / V'), (axs[1], fields[1], 'coolwarm', (-1, 1), 'Cp')]:
            cs_ = ax_.contourf(xh, yv, F, levels=np.linspace(*clim, 65), cmap=cmap, extend='both')
            ax_.streamplot(xh, yv, np.where(msk, 0, Ux), np.where(msk, 0, Uy), density=(3.2, 1.6) if horiz else (1.6, 3.2),
                           color='white', linewidth=0.6, arrowsize=0.7)
            for p in polys:
                ax_.add_patch(Polygon(pmap(p), closed=True, fc='#9aa3ad', ec='black', lw=1.2, zorder=5))
            cb = fig.colorbar(cs_, ax=ax_, shrink=0.95, pad=0.01, aspect=18); cb.set_label(lbl, fontsize=13)
            cb.set_ticks(np.linspace(*clim, 6))
            ax_.set_xlim(xh.min(), xh.max()); ax_.set_ylim(yv.min(), yv.max()); ax_.set_aspect('equal')
            ax_.set_xlabel(xl, fontsize=12); ax_.set_ylabel(yl, fontsize=12)
            ax_.set_title(f'{title}: mid-plane {"xyz"[n_ax]} = 0, {lbl}', fontsize=14)
        plt.savefig(os.path.join(FIG, f'slice_{"xyz"[n_ax]}0_{tag}.png')); plt.close()

    # ---------------- Cp on body, four views
    pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(2400, 1500)); pl.set_background('white')
    pl.enable_anti_aliasing('ssaa')
    views = [((1, 0.6, 1), 'iso, from +Z end'), ((-1, 0.6, -1), 'iso, from −Z end'), ((0, 1, 0.001), 'top (+Y)'), ((1, 0, 0.001), 'side (+X)')]
    for i, (vv, nm) in enumerate(views):
        pl.subplot(i // 2, i % 2)
        pl.add_mesh(body.copy(), scalars='Cp', cmap='coolwarm', clim=cp_rng, smooth_shading=True, specular=0.3,
                    scalar_bar_args=dict(title='Cp', vertical=False, position_x=0.25, position_y=0.04, width=0.5,
                                         title_font_size=26, label_font_size=22), show_scalar_bar=(i == 0))
        pl.add_text(f'{title}: {nm}', font_size=12)
        pl.view_vector(vv, viewup=(0, 1, 0) if nm != 'top (+Y)' else (1, 0, 0)); pl.reset_camera(); pl.camera.zoom(1.3)
    pl.screenshot(os.path.join(FIG, f'cp_{tag}.png')); pl.close()
    print('wrote', tag, flush=True)


if __name__ == '__main__':
    for r in sys.argv[1:]:
        main(os.path.abspath(r))
