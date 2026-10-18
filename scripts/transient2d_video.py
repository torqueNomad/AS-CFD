"""Render the 2-D impulsive-start sway animation (speed colours developing from still water).
Reads the pimpleFoam time directories of a case made by transient2d_setup.py, resamples |U|/V on a regular grid and
writes an MP4. Run under xvfb-run (pyvista off-screen is only used for the OpenFOAM reader).
Usage: transient2d_video.py <case_dir> <out.mp4> [t_max] [fps]"""
import os, sys, json
import numpy as np
import pyvista as pv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

X0, X1, Y0, Y1, H = -2.5, 11.5, -1.5, 1.5, 0.006     # view window [m] and grid spacing
CHORD = 1.48                                          # body length along the flow [m]


def main(case, out, t_max=1e9, fps=30):
    meta = json.load(open(os.path.join(case, 'case2d.json')))
    V = meta['V']; outline = np.array(meta['outline'])
    open(os.path.join(case, 'case.foam'), 'a').close()
    r = pv.OpenFOAMReader(os.path.join(case, 'case.foam'))
    times = [t for t in r.time_values if 0 < t <= t_max]
    nx, ny = int((X1 - X0) / H) + 1, int((Y1 - Y0) / H) + 1
    grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(H, H, 1), origin=(X0, Y0, 0.025))
    cmap = plt.get_cmap('turbo').copy()
    fig = plt.figure(figsize=(16, 4.9), dpi=120, layout='constrained')
    ax = fig.add_subplot(111)
    im = ax.imshow(np.zeros((ny, nx)), extent=(X0, X1, Y0, Y1), origin='lower', cmap=cmap, vmin=0, vmax=1.3,
                   interpolation='bilinear', aspect='equal')
    ax.add_patch(Polygon(outline, closed=True, fc='#9aa3ad', ec='black', lw=1.2, zorder=5))
    cb = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.01, aspect=14); cb.set_label('|U| / V', fontsize=12)
    cb.set_ticks(np.linspace(0, 1.3, 6))
    ax.set_xlabel('x [m]  (flow left → right)', fontsize=11); ax.set_ylabel('y [m] (heave)', fontsize=11)
    ax.set_title('Sway, 6 kn (3.09 m/s): flow development from rest, mid-span section, 2-D unsteady RANS (illustration only)', fontsize=12)
    txt = ax.text(0.012, 0.93, '', transform=ax.transAxes, fontsize=13, family='monospace', zorder=9,
                  bbox=dict(fc='white', ec='none', alpha=0.85, pad=3))
    wr = FFMpegWriter(fps=fps, bitrate=7000, codec='libx264',
                      extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p'])
    with wr.saving(fig, out, dpi=120):
        for i, t in enumerate(times):
            r.set_active_time_value(t)
            m = r.read()['internalMesh'].cell_data_to_point_data()
            s = grid.sample(m)
            U = np.asarray(s.point_data['U']); ok = np.asarray(s.point_data['vtkValidPointMask']) > 0
            uv = np.where(ok, np.linalg.norm(U[:, :2], axis=1) / V, np.nan).reshape(ny, nx)
            im.set_data(np.ma.masked_invalid(uv))
            txt.set_text(f't = {t:5.2f} s    distance travelled = {V * t / CHORD:5.1f} body lengths')
            wr.grab_frame()
            if i % 40 == 0: print('frame', i, 'of', len(times), flush=True)


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]),
         float(sys.argv[3]) if len(sys.argv) > 3 else 1e9, int(sys.argv[4]) if len(sys.argv) > 4 else 30)
