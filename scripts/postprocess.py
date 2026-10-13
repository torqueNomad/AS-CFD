"""Collect forces from all runs, build drag / power tables and plots.
Drag D = force on body projected on the free-stream direction, averaged over the last 30 % of iterations.
Hydraulic (tow) power P = D * V.  Cd = D / (0.5 rho V^2 A_proj)."""
import os, glob, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results'); FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)
KN = 0.514444
AXES = ['surge', 'sway', 'heave']
# heave: steady-RANS Cd scatters +-5 % between speeds with no Re trend (wandering separated wake)
CONST_CD = {'heave'}
COL = {'surge': '#1f77b4', 'sway': '#d62728', 'heave': '#2ca02c'}


def load_run(run):
    meta = json.load(open(os.path.join(run, 'case.json')))
    f = os.path.join(run, 'postProcessing', 'forces', '0', 'force.dat')
    d = np.loadtxt(f, comments='#')
    it, F, Fp, Fv = d[:, 0], d[:, 1:4], d[:, 4:7], d[:, 7:10]
    u = np.array(meta['U']) / np.linalg.norm(meta['U'])
    n = len(it); s = slice(int(0.7 * n), n)
    D = F @ u; Dp = Fp @ u; Dv = Fv @ u
    yp = None
    for yf in glob.glob(os.path.join(run, 'postProcessing', 'yPlus', '*', 'yPlus.dat')):
        y = np.loadtxt(yf, comments='#', dtype=str)
        y = np.atleast_2d(y); yp = [float(y[-1][2]), float(y[-1][3]), float(y[-1][4])]  # min max avg
    return dict(meta=meta, it=it, D=D, Dmean=D[s].mean(), Dstd=D[s].std(), Dp=Dp[s].mean(), Dv=Dv[s].mean(),
                Fmean=F[s].mean(0).tolist(), niter=int(it[-1]), yplus=yp,
                drift=abs(D[int(0.7 * n):int(0.85 * n)].mean() - D[int(0.85 * n):].mean()))


def main():
    rows = []
    for run in sorted(glob.glob(os.path.join(ROOT, 'cases', '*_*', 'V*kn'))):
        if not os.path.exists(os.path.join(run, 'DONE')):
            continue
        r = load_run(run); m = r['meta']
        V, A, rho, nu = m['V'], m['Aproj'], m['rho'], m['nu']
        q = 0.5 * rho * V ** 2
        rows.append(dict(axis=m['axis'], mesh=m['mesh'], kn=m['kn'], V=V, Re=V * m['L_axis'] / nu, A=A,
                         D=r['Dmean'], D_std=r['Dstd'], D_drift=r['drift'], D_pressure=r['Dp'], D_viscous=r['Dv'],
                         Cd=r['Dmean'] / (q * A), P=r['Dmean'] * V, niter=r['niter'], yplus=r['yplus'],
                         F_body=r['Fmean']))
        # convergence plot
        plt.figure(figsize=(6, 3.5)); plt.plot(r['it'], r['D'], lw=0.8)
        plt.ylim(0, 2.5 * max(r['Dmean'], 1e-6)); plt.xlabel('iteration'); plt.ylabel('drag [N]')
        plt.title(f"{m['axis']} {m['mesh']} {m['kn']:g} kn: D = {r['Dmean']:.1f} N"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, f"conv_{m['axis']}_{m['mesh']}_{m['kn']:g}kn.png"), dpi=80); plt.close()
    json.dump(rows, open(os.path.join(OUT, 'runs.json'), 'w'), indent=1)
    with open(os.path.join(OUT, 'runs_raw.csv'), 'w') as f:
        f.write('axis,mesh,speed_kn,speed_mps,Re,A_proj_m2,drag_N,drag_std_N,drag_pressure_N,drag_viscous_N,Cd,power_W,iterations,yplus_min,yplus_max,yplus_avg\n')
        for r in sorted(rows, key=lambda r: (r['axis'], r['mesh'], r['kn'])):
            y = r['yplus'] or [np.nan] * 3
            f.write(f"{r['axis']},{r['mesh']},{r['kn']:g},{r['V']:.4f},{r['Re']:.4e},{r['A']:.4f},{r['D']:.2f},{r['D_std']:.2f},"
                    f"{r['D_pressure']:.2f},{r['D_viscous']:.2f},{r['Cd']:.4f},{r['P']:.1f},{r['niter']},{y[0]:.1f},{y[1]:.1f},{y[2]:.1f}\n")

    # production table: coarse-mesh speed sweep, Cd(Re) fitted as linear in log10(Re), then corrected by medium/coarse ratio at 6 kn
    table = {}
    for ax in AXES:
        pr = sorted([r for r in rows if r['axis'] == ax and r['mesh'] == 'coarse'], key=lambda r: r['kn'])
        if len(pr) < 2:
            continue
        lr = np.log10([r['Re'] for r in pr]); cd = np.array([r['Cd'] for r in pr])
        if ax in CONST_CD:   # Cd scatter is solver noise (unsteady wake), not a Re trend: use the mean
            c1, c0 = 0.0, float(cd.mean())
        else:
            c1, c0 = np.polyfit(lr, cd, 1)
        # finest available mesh at 6 kn sets the grid correction
        med = [r for r in rows if r['axis'] == ax and r['mesh'] == 'fine' and r['kn'] == 6] or \
              [r for r in rows if r['axis'] == ax and r['mesh'] == 'medium' and r['kn'] == 6]
        # grid correction = medium-mesh Cd / coarse-mesh fitted Cd, both at 6 kn
        corr = med[0]['Cd'] / (c0 + c1 * np.log10(med[0]['Re'])) if med else 1.0
        table[ax] = dict(fit=(c0, c1), corr=corr, cd_min=float(cd.min()), cd_max=float(cd.max()), A=pr[0]['A'], L=pr[0]['Re'] * 1.1883e-6 / pr[0]['V'],
                         pts=[(r['kn'], r['D'] * corr, r['P'] * corr) for r in pr])
    kns = np.round(np.arange(0.5, 6.01, 0.5), 2)
    rho, nu = 1025.9, 1.1883e-6
    with open(os.path.join(OUT, 'drag_power_table.csv'), 'w') as f:
        f.write('speed_kn,speed_mps,' + ','.join(f'{a}_drag_N,{a}_power_W,{a}_Cd' for a in table) + '\n')
        f.write('0,0,' + ','.join('0,0,' for _ in table) + '\n')
        for k in kns:
            V = k * KN; cells = []
            for a, t in table.items():
                Cd = (t['fit'][0] + t['fit'][1] * np.log10(V * t['L'] / nu)) * t['corr']
                D = 0.5 * rho * V ** 2 * t['A'] * Cd
                cells.append(f'{D:.1f},{D * V:.1f},{Cd:.4f}')
            f.write(f'{k:g},{V:.4f},' + ','.join(cells) + '\n')
    json.dump({a: dict(Cd_fit_c0=t['fit'][0], Cd_fit_c1_per_log10Re=t['fit'][1], grid_correction=t['corr'],
                       Cd_runs_min=t['cd_min'], Cd_runs_max=t['cd_max'], fit='constant (mean)' if a in CONST_CD else 'c0 + c1*log10(Re)',
                       A_proj_m2=t['A'], L_ref_m=t['L']) for a, t in table.items()},
              open(os.path.join(OUT, 'fit_coefficients.json'), 'w'), indent=1)

    # plots
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    vv = np.linspace(0, 6, 121)
    for a, t in table.items():
        V = np.maximum(vv * KN, 1e-6)
        Cd = (t['fit'][0] + t['fit'][1] * np.log10(V * t['L'] / nu)) * t['corr']
        D = 0.5 * rho * V ** 2 * t['A'] * Cd
        k, Dp, Pp = zip(*t['pts'])
        axs[0].plot(vv, D, color=COL[a], label=a); axs[0].plot(k, Dp, 'o', color=COL[a])
        axs[1].plot(vv, D * V / 1000, color=COL[a], label=a); axs[1].plot(k, np.array(Pp) / 1000, 'o', color=COL[a])
        axs[2].plot(vv[1:], Cd[1:], color=COL[a], label=a)
    axs[0].set(xlabel='speed [kn]', ylabel='drag [N]', title='Drag force')
    axs[1].set(xlabel='speed [kn]', ylabel='hydraulic power D·V [kW]', title='Hydraulic (tow) power')
    axs[2].set(xlabel='speed [kn]', ylabel='Cd (on projected area)', title='Drag coefficient')
    for ax in axs: ax.grid(alpha=.3); ax.legend(); ax.set_xlim(0, 6)
    fig.suptitle('Markers = CFD runs, lines = Cd(log Re) fit; sea water 15 °C (ρ=1025.9 kg/m³)')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'drag_power_vs_speed.png'), dpi=100)
    print(open(os.path.join(OUT, 'runs_raw.csv')).read())
    print(open(os.path.join(OUT, 'drag_power_table.csv')).read())


if __name__ == '__main__':
    main()
