"""Flow animations (MP4, H.264) for one run. Run under xvfb-run.
  anim3d_<tag>.mp4          : particles travelling along the 3-D streamlines (true relative speed, slowed down),
                              over faint streamlines, Cp-coloured body and the speed mid-plane. Seamless loop.
  anim_slice_<plane>_<tag>.mp4 : particles with fading trails advected by the in-plane velocity on each mid-plane,
                              over the |U|/V contour field.
  anim_cp_<tag>.mp4         : turntable of the body coloured by Cp.
Particle motion uses the converged steady CFD velocity field; it shows where water goes, not unsteady shedding.
Usage: xvfb-run -a python3 animate.py <run_dir> [...]"""
import os, sys, json, numpy as np
import pyvista as pv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import reconstruct, load, image_grid, seeds_for, section_polys

matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'animations'); os.makedirs(OUT, exist_ok=True)
FPS = 30
LAB = ['x [m] (sway)', 'y [m] (heave)', 'z [m] (surge)']
CAM = {'surge': [(9, 6, 10), (0, 0, -2.5), (0, 1, 0)], 'sway': [(8, 7, 9), (-2, 0, 0), (0, 1, 0)],
       'heave': [(9, 8, 8), (0, -1.5, 0), (0, 1, 0)]}


def prepare(run):
    meta = json.load(open(os.path.join(run, 'case.json')))
    reconstruct(run)
    vol, body = load(run)
    V = meta['V']; a = {'sway': 0, 'heave': 1, 'surge': 2}[meta['axis']]
    vol = vol.cell_data_to_point_data()
    vol.point_data['UV'] = np.linalg.norm(vol.point_data['U'], axis=1) / V
    body = body.cell_data_to_point_data()
    body.point_data['Cp'] = body.point_data['p'] / (0.5 * V ** 2)
    body = body.extract_surface(algorithm=None).compute_normals(auto_orient_normals=True)
    ext = np.array(meta['extents'])
    lo = -ext / 2 - 1.2; hi = ext / 2 + 1.2
    hi[a] = ext[a] / 2 + 1.5; lo[a] = -ext[a] / 2 - (max(3.0, 0.8 * ext[a]) if a == 2 else 2.5 * ext[a] + 1.0)
    return meta, vol, body, V, a, ext, lo, hi


def anim3d(run, meta, vol, body, V, a, ext, lo, hi, tag, title):
    others = [i for i in range(3) if i != a]
    img = image_grid(lo, hi, 4.5e6).sample(vol, pass_point_data=False)
    U = img.point_data['U'].copy(); U[img.point_data['vtkValidPointMask'] == 0] = 0; img.point_data['U'] = U
    c = np.zeros(3); c[a] = ext[a] / 2 + 1.0
    sl = img.streamlines_from_source(seeds_for(meta['axis'], c, others, ext), vectors='U', integrator_type=45,
                                     integration_direction='forward', initial_step_length=0.2, min_step_length=0.01,
                                     max_step_length=0.5, max_steps=20000, max_error=1e-6, max_length=40.0,
                                     terminal_speed=1e-4)
    sl.point_data['UV'] = np.linalg.norm(sl.point_data['U'], axis=1) / V
    T = sl.point_data['IntegrationTime']; P = sl.points; UVs = sl.point_data['UV']
    lines = []
    for k in range(sl.n_cells):
        ids = sl.get_cell(k).point_ids
        if len(ids) > 5: lines.append((T[ids], P[ids], UVs[ids]))
    # one loop = particles advance by one spacing DT (seconds of real flow); slow-motion factor = loop_s / DT
    DT, TRAIL, NF = 0.40, 0.10, 120
    bn = 1 if a != 1 else 0
    back = img.slice(normal=np.eye(3)[bn], origin=(0, 0, 0) if a != 1 else (-ext[0] / 2 - 0.02, 0, 0))
    back = back.threshold(0.5, scalars='vtkValidPointMask')
    pl = pv.Plotter(off_screen=True, window_size=(1920, 1088)); pl.set_background('white')
    pl.enable_anti_aliasing('ssaa')
    pl.add_mesh(back, scalars='UV', cmap='turbo', clim=[0, 1.3], lighting=False, show_scalar_bar=False, opacity=0.45)
    pl.add_mesh(body, scalars='Cp', cmap='coolwarm', clim=[-0.8, 1.0], smooth_shading=True, specular=0.3,
                scalar_bar_args=dict(title='Cp (body)', n_labels=5, fmt='%.1f', vertical=True, position_x=0.03,
                                     position_y=0.25, height=0.5, title_font_size=22, label_font_size=18))
    pl.add_mesh(sl.tube(radius=0.003, n_sides=8), color='#6c8086', opacity=0.35)
    pl.add_text(f'{title}: water particles along streamlines (slowed ×{NF / FPS / DT:.0f}), coloured by U/V', font_size=12)
    pl.camera_position = CAM[meta['axis']]; pl.camera.zoom(1.15)
    pl.open_movie(os.path.join(OUT, f'anim3d_{tag}.mp4'), framerate=FPS, quality=8)
    actor = None
    for f in range(NF):
        phase = f / NF * DT
        pts, uv, conn, n0 = [], [], [], 0
        for t, p, u in lines:
            for tau in np.arange(phase, t[-1], DT):
                ts = np.linspace(max(tau - TRAIL, 0), tau, 7)
                seg = np.c_[np.interp(ts, t, p[:, 0]), np.interp(ts, t, p[:, 1]), np.interp(ts, t, p[:, 2])]
                pts.append(seg); uv.append(np.interp(ts, t, u))
                conn.append(np.r_[7, np.arange(n0, n0 + 7)]); n0 += 7
        poly = pv.PolyData(np.vstack(pts), lines=np.concatenate(conn)); poly.point_data['UV'] = np.concatenate(uv)
        if actor is not None: pl.remove_actor(actor)
        actor = pl.add_mesh(poly.tube(radius=0.010, n_sides=10), scalars='UV', cmap='turbo', clim=[0, 1.3],
                            smooth_shading=True,
                            scalar_bar_args=dict(title='U / V', n_labels=6, fmt='%.1f', vertical=True, position_x=0.91,
                                                 position_y=0.25, height=0.5, title_font_size=22, label_font_size=18))
        pl.write_frame()
    pl.close()


def anim_slices(run, meta, vol, body, V, a, ext, lo, hi, tag, title):
    others = [i for i in range(3) if i != a]
    rng = np.random.default_rng(1)
    for n_ax in others:
        j = [k for k in others if k != n_ax][0]
        slo = np.array(lo, float) + 0.05; shi = np.array(hi, float) - 0.05; slo[n_ax] = shi[n_ax] = 0.0
        g2 = image_grid(slo, shi, 1.0e6).sample(vol, pass_point_data=False)
        nx = g2.dimensions; axes_in = [k for k in range(3) if nx[k] > 1]
        shape = tuple(int(d) for d in nx if d > 1)
        arr = lambda v: v.reshape(shape[::-1])
        cf_ = lambda v: v if axes_in[0] == a else v.T          # -> [cross, flow]
        mask = cf_(arr(g2.point_data['vtkValidPointMask']) == 0)
        UVg = np.ma.array(cf_(arr(g2.point_data['UV'])), mask=mask)
        Ug = g2.point_data['U']
        Ua, Uc = cf_(arr(Ug[:, a])), cf_(arr(Ug[:, j]))
        fco = np.linspace(slo[a], shi[a], UVg.shape[1]); cco = np.linspace(slo[j], shi[j], UVg.shape[0])
        polys = [np.c_[p[:, axes_in.index(a)], p[:, axes_in.index(j)]] for p in section_polys(body, np.eye(3)[n_ax], axes_in[0], axes_in[1])]
        Lf, Lc = shi[a] - slo[a], shi[j] - slo[j]
        horiz = Lc <= 1.2 * Lf
        if horiz:
            xh, yv, F = -fco[::-1], cco, UVg[:, ::-1]; Ux, Uy = (-Ua)[:, ::-1], Uc[:, ::-1]; M = mask[:, ::-1]
            pmap = lambda p: np.c_[-p[:, 0], p[:, 1]]; xl, yl = '−' + LAB[a] + '   (flow left → right)', LAB[j]; asp = Lc / Lf
        else:
            xh, yv, F = cco, fco, UVg.T; Ux, Uy = Uc.T, Ua.T; M = mask.T
            pmap = lambda p: np.c_[p[:, 1], p[:, 0]]; xl, yl = LAB[j], LAB[a] + '   (flow top → bottom)'; asp = Lf / Lc
        Ux = np.where(M, 0, Ux); Uy = np.where(M, 0, Uy)
        iu = RegularGridInterpolator((yv, xh), Ux, bounds_error=False, fill_value=0)
        iv = RegularGridInterpolator((yv, xh), Uy, bounds_error=False, fill_value=0)
        im = RegularGridInterpolator((yv, xh), M.astype(float), bounds_error=False, fill_value=1)
        W, H = xh[-1] - xh[0], yv[-1] - yv[0]
        NP, NF, NT = 1800, 300, 12
        dt = 0.0035 * max(W, H) / V          # freestream particle moves 0.35 % of the frame width per frame

        def spawn(n, upstream_frac=0.5):
            p = np.c_[rng.uniform(xh[0], xh[-1], n), rng.uniform(yv[0], yv[-1], n)]
            up = rng.random(n) < upstream_frac
            if horiz: p[up, 0] = xh[0] + rng.uniform(0, 0.02 * W, up.sum())
            else: p[up, 1] = yv[-1] - rng.uniform(0, 0.02 * H, up.sum())
            return p
        P = spawn(NP, 0.0); P[im(P[:, ::-1]) > 0.5] = spawn(int((im(P[:, ::-1]) > 0.5).sum()), 1.0)
        hist = np.repeat(P[None], NT, 0); age = rng.integers(0, 200, NP); life = rng.integers(160, 320, NP)
        vel = lambda q: np.c_[iu(q[:, ::-1]), iv(q[:, ::-1])]
        figw = 16; fig, ax = plt.subplots(figsize=(figw, figw * asp + 1.0), dpi=120, layout='constrained')
        cs_ = ax.contourf(xh, yv, F, levels=np.linspace(0, 1.3, 65), cmap='turbo', extend='both')
        for p in polys: ax.add_patch(Polygon(pmap(p), closed=True, fc='#9aa3ad', ec='black', lw=1.2, zorder=5))
        cb = fig.colorbar(cs_, ax=ax, shrink=0.95, pad=0.01, aspect=18); cb.set_label('|U| / V', fontsize=12); cb.set_ticks(np.linspace(0, 1.3, 6))
        ax.set_xlim(xh[0], xh[-1]); ax.set_ylim(yv[0], yv[-1]); ax.set_aspect('equal')
        ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_title(f'{title}: mid-plane {"xyz"[n_ax]} = 0, particles advected by the CFD velocity field', fontsize=13)
        lc = LineCollection([], linewidths=1.1, zorder=6); ax.add_collection(lc)
        alpha = np.linspace(0.05, 0.95, NT - 1)
        wr = FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264', extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p'])
        with wr.saving(fig, os.path.join(OUT, f'anim_slice_{"xyz"[n_ax]}0_{tag}.mp4'), dpi=120):
            for f in range(NF):
                k1 = vel(P); k2 = vel(P + 0.5 * dt * k1); P = P + dt * k2; age += 1
                dead = (P[:, 0] < xh[0]) | (P[:, 0] > xh[-1]) | (P[:, 1] < yv[0]) | (P[:, 1] > yv[-1]) \
                       | (im(P[:, ::-1]) > 0.5) | (age > life)
                if dead.any():
                    P[dead] = spawn(int(dead.sum())); age[dead] = 0; hist[:, dead] = P[dead]
                hist = np.roll(hist, -1, 0); hist[-1] = P
                segs = np.stack([hist[:-1], hist[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
                cols = np.zeros((len(segs), 4)); cols[:, :3] = 1; cols[:, 3] = np.tile(alpha, NP)
                lc.set_segments(segs); lc.set_color(cols)
                wr.grab_frame()
        plt.close(fig)


def anim_cp(meta, body, tag, title):
    pl = pv.Plotter(off_screen=True, window_size=(1920, 1088)); pl.set_background('white')
    pl.enable_anti_aliasing('ssaa')
    pl.add_mesh(body, scalars='Cp', cmap='coolwarm', clim=[-0.8, 1.0], smooth_shading=True, specular=0.3,
                scalar_bar_args=dict(title='Cp', n_labels=5, fmt='%.1f', vertical=True, position_x=0.9,
                                     position_y=0.25, height=0.5, title_font_size=22, label_font_size=18))
    pl.add_text(f'{title}: surface pressure coefficient Cp', font_size=12)
    pl.open_movie(os.path.join(OUT, f'anim_cp_{tag}.mp4'), framerate=FPS, quality=8)
    NF = 240
    for f in range(NF):
        th = 2 * np.pi * f / NF
        pl.camera_position = [(11 * np.cos(th), 4.5, 11 * np.sin(th)), (0, 0, 0), (0, 1, 0)]
        pl.write_frame()
    pl.close()


def main(run, what):
    meta, vol, body, V, a, ext, lo, hi = prepare(run)
    tag = f"{meta['axis']}_{meta['mesh']}_{meta['kn']:g}kn"
    title = f"{meta['axis'].capitalize()}, {meta['kn']:g} kn ({V:.2f} m/s)"
    if '3d' in what: anim3d(run, meta, vol, body, V, a, ext, lo, hi, tag, title); print('3d', tag, flush=True)
    if 'slice' in what: anim_slices(run, meta, vol, body, V, a, ext, lo, hi, tag, title); print('slices', tag, flush=True)
    if 'cp' in what: anim_cp(meta, body, tag, title); print('cp', tag, flush=True)


if __name__ == '__main__':
    what = os.environ.get('ANIM', '3d,slice,cp').split(',')
    for r in sys.argv[1:]:
        main(os.path.abspath(r), what)
