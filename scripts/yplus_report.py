"""y+ on the body for every finished run (run under xvfb-run).
y+ is the OpenFOAM `yPlus` function-object field on the body patch (first-cell-centre y+).
Outputs:
  results/yplus_summary.csv          area-weighted statistics and the fraction of wetted area in each wall region
  results/figures/yplus/hist_<axis>_<mesh>.png   area-weighted y+ histograms, one curve per speed
  results/figures/yplus/yplus_<tag>.png          y+ contour on the body (two views)
Wall regions: <5 viscous sublayer, 5-30 buffer layer, 30-300 log layer, >300 outer/over-coarse.
The wall model used (nutUSpaldingWallFunction) is a continuous law of the wall valid across all three inner regions."""
import os, sys, glob, json, numpy as np
import pyvista as pv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import reconstruct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'results', 'figures', 'yplus'); os.makedirs(FIG, exist_ok=True)
BINS = [0, 5, 30, 300, 1e9]


def wpct(x, w, q):
    i = np.argsort(x); c = np.cumsum(w[i]) / w.sum()
    return [float(np.interp(qq / 100, c, x[i])) for qq in q]


def body_yplus(run):
    reconstruct(run)
    r = pv.OpenFOAMReader(os.path.join(run, 'case.foam'))
    r.set_active_time_value(r.time_values[-1]); r.enable_all_patch_arrays()
    b = r.read()['boundary']['body']
    b = b.extract_surface(algorithm=None) if not isinstance(b, pv.PolyData) else b
    area = b.compute_cell_sizes(length=False, volume=False).cell_data['Area']
    return b, np.asarray(b.cell_data['yPlus']), np.asarray(area)


def main():
    rows, hist = [], {}
    for run in sorted(glob.glob(os.path.join(ROOT, 'cases', '*_*', 'V*kn'))):
        if not os.path.exists(os.path.join(run, 'DONE')):
            continue
        m = json.load(open(os.path.join(run, 'case.json')))
        b, y, A = body_yplus(run)
        p = wpct(y, A, [5, 50, 95])
        frac = [A[(y >= lo) & (y < hi)].sum() / A.sum() * 100 for lo, hi in zip(BINS[:-1], BINS[1:])]
        rows.append([m['axis'], m['mesh'], m['kn'], y.min(), y.max(), (y * A).sum() / A.sum(), *p, *frac])
        hist.setdefault((m['axis'], m['mesh']), []).append((m['kn'], y, A))
        tag = f"{m['axis']}_{m['mesh']}_{m['kn']:g}kn"
        if m['kn'] in (1, 6):  # contour pictures at lowest and highest speed
            pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 650)); pl.set_background('white')
            for i, vv in enumerate([(1, 0.6, 1), (-1, -0.6, -1)]):
                pl.subplot(0, i)
                pl.add_mesh(b.copy(), scalars='yPlus', cmap='viridis', clim=[0, 400], log_scale=False,
                            scalar_bar_args=dict(title='y+', vertical=False, position_x=0.25, position_y=0.05, width=0.5),
                            show_scalar_bar=(i == 0))
                pl.add_text(f"{m['axis'].upper()} {m['kn']:g} kn: y+ on body ({'upper/bow side' if i == 0 else 'lower/stern side'})", font_size=10)
                pl.view_vector(vv, viewup=(0, 1, 0)); pl.reset_camera(); pl.camera.zoom(1.4)
            pl.screenshot(os.path.join(FIG, f'yplus_{tag}.png')); pl.close()
        print(tag, 'done', flush=True)
    hdr = 'axis,mesh,speed_kn,yplus_min,yplus_max,yplus_area_avg,yplus_p05,yplus_p50,yplus_p95,' \
          'area_pct_lt5,area_pct_5_30,area_pct_30_300,area_pct_gt300'
    with open(os.path.join(ROOT, 'results', 'yplus_summary.csv'), 'w') as f:
        f.write(hdr + '\n')
        for r in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            f.write(f'{r[0]},{r[1]},{r[2]:g},' + ','.join(f'{v:.1f}' for v in r[3:]) + '\n')
    for (ax, mesh), lst in hist.items():
        plt.figure(figsize=(7, 4)); e = np.logspace(-0.5, 3.5, 60)
        for kn, y, A in sorted(lst, key=lambda t: t[0]):
            h, _ = np.histogram(y, bins=e, weights=A / A.sum() * 100)
            plt.semilogx(np.sqrt(e[1:] * e[:-1]), h, label=f'{kn:g} kn')
        for v in (5, 30, 300): plt.axvline(v, color='grey', ls='--', lw=0.8)
        plt.text(1.2, plt.ylim()[1] * 0.92, 'viscous', fontsize=8); plt.text(9, plt.ylim()[1] * 0.92, 'buffer', fontsize=8)
        plt.text(60, plt.ylim()[1] * 0.92, 'log layer', fontsize=8)
        plt.xlabel('y+ (first cell centre)'); plt.ylabel('% of wetted area per bin'); plt.title(f'y+ distribution on body: {ax}, {mesh} mesh')
        plt.legend(); plt.grid(alpha=.3, which='both'); plt.tight_layout()
        plt.savefig(os.path.join(FIG, f'hist_{ax}_{mesh}.png'), dpi=90); plt.close()
    print(open(os.path.join(ROOT, 'results', 'yplus_summary.csv')).read())


if __name__ == '__main__':
    main()
