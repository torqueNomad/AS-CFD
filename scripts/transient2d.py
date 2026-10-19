"""2-D unsteady-RANS impulsive-start cases of the mid-planes, with flow-development videos (illustration only).
Each plane is the cut of the sealed envelope by x=0, y=0 or z=0 (body frame: X sway, Y heave, Z surge). The fluid starts
at rest and the 6 kn free stream is imposed at t = 0. Flow runs towards +x of the picture (the flow-axis coordinate is
mirrored, exactly as in the steady slice figures). Not used for the drag numbers: a 2-D section ignores the body ends.
Usage:
  transient2d.py all <plane> [<plane> ...]     mesh, run (4 MPI ranks), render video, clean up, one plane after another
  transient2d.py mesh <plane> [...]            mesh only (dry run, prints cell count and mesh quality)
  transient2d.py video <case_dir> <plane>      (re-)render the video from existing time directories
planes: surge_x0 surge_y0 sway_z0 sway_y0 heave_x0 heave_z0"""
import os, sys, json, re, shutil, subprocess, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'cases2d'); OUTV = os.path.join(ROOT, 'results', 'animations')
RHO, NU, KN = 1025.9, 1.1883e-6, 0.514444
OF = 'source /usr/lib/openfoam/openfoam2512/etc/bashrc'
PLANES = {  # n = plane normal axis, a = flow axis, j = cross-flow axis (0 x, 1 y, 2 z)
    'surge_x0': dict(axis='surge', n=0, a=2, j=1, t_end=20.0), 'surge_y0': dict(axis='surge', n=1, a=2, j=0, t_end=20.0),
    'sway_z0': dict(axis='sway', n=2, a=0, j=1, t_end=12.0), 'sway_y0': dict(axis='sway', n=1, a=0, j=2, t_end=12.0),
    'heave_x0': dict(axis='heave', n=0, a=1, j=2, t_end=12.0), 'heave_z0': dict(axis='heave', n=2, a=1, j=0, t_end=12.0)}
FH = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    %sobject %s;\n}\n"


def sh(case, cmd, log, mode='w'):
    r = subprocess.run(['bash', '-c', f'{OF}; cd {case}; {cmd}'], stdout=open(os.path.join(case, log), mode), stderr=subprocess.STDOUT)
    return r.returncode


def wf(path, obj, body, cls='dictionary', loc=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write(FH % (cls, loc, obj) + body)


def outline(pl):
    import trimesh
    e = trimesh.load(os.path.join(ROOT, 'geometry', 'shell_envelope.stl')); e.apply_translation(-e.bounds.mean(0))
    s = e.section(plane_origin=[0, 0, 0], plane_normal=np.eye(3)[pl['n']])
    p = max(s.discrete, key=len)[:, [pl['a'], pl['j']]]
    p = np.c_[-p[:, 0], p[:, 1]]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1))]
    per = d[-1]; npts = int(per / (0.012 if per < 6 else 0.02))
    t = np.linspace(0, per, npts, endpoint=False)
    return np.c_[np.interp(t, d, np.r_[p[:, 0], p[0, 0]]), np.interp(t, d, np.r_[p[:, 1], p[0, 1]])], per


def build_mesh(case, ol, per, use_bl):
    import gmsh
    bx0, by0 = ol.min(0); bx1, by1 = ol.max(0); Lf, Lc = bx1 - bx0, by1 - by0
    big = per > 6
    sz, bl1, wk, far = (0.02, 0.004, 0.08 if Lc > 2 else 0.05, 0.6) if big else (0.012, 0.003, 0.04, 0.45)
    up, down, lat = max(5, 1.2 * Lc + 3), max(14, 2.5 * Lc + 10), max(4.5, 2 * Lc)
    xl, xr, yb, yt = bx0 - up, bx1 + down, by0 - lat, by1 + lat
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0); gmsh.model.add('s2d'); g = gmsh.model.geo
    pts = [g.addPoint(x, y, 0, sz) for x, y in ol]; n = len(pts); sp = []; k = 30
    for i in range(0, n, k):
        sp.append(g.addSpline(pts[i:i + k + 1] if i + k < n else pts[i:] + [pts[0]]))
    body = g.addCurveLoop(sp)
    c = [g.addPoint(xl, yb, 0, far), g.addPoint(xr, yb, 0, far), g.addPoint(xr, yt, 0, far), g.addPoint(xl, yt, 0, far)]
    L = [g.addLine(c[0], c[1]), g.addLine(c[1], c[2]), g.addLine(c[2], c[3]), g.addLine(c[3], c[0])]
    surf = g.addPlaneSurface([g.addCurveLoop(L), body]); g.synchronize()
    f = gmsh.model.mesh.field
    f.add('Distance', 1); f.setNumbers(1, 'CurvesList', sp); f.setNumber(1, 'Sampling', 400)
    f.add('Threshold', 2); f.setNumber(2, 'InField', 1); f.setNumber(2, 'SizeMin', sz); f.setNumber(2, 'SizeMax', far)
    f.setNumber(2, 'DistMin', 0.08); f.setNumber(2, 'DistMax', max(4, 0.5 * Lc))
    f.add('Box', 3); f.setNumber(3, 'VIn', wk); f.setNumber(3, 'VOut', far)
    f.setNumber(3, 'XMin', (bx0 + bx1) / 2); f.setNumber(3, 'XMax', bx1 + 11)
    f.setNumber(3, 'YMin', by0 - 0.9); f.setNumber(3, 'YMax', by1 + 0.9); f.setNumber(3, 'Thickness', 2)
    f.add('Min', 4); f.setNumbers(4, 'FieldsList', [2, 3]); f.setAsBackgroundMesh(4)
    if use_bl:
        f.add('BoundaryLayer', 5); f.setNumbers(5, 'CurvesList', sp); f.setNumber(5, 'Size', bl1); f.setNumber(5, 'Ratio', 1.2)
        f.setNumber(5, 'Quads', 1); f.setNumber(5, 'Thickness', 0.05 if big else 0.03); f.setNumber(5, 'IntersectMetrics', 0)
        f.setAsBoundaryLayer(5)
    gmsh.option.setNumber('Mesh.Algorithm', 6); gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0); gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    ex = g.extrude([(2, surf)], 0, 0, 0.05, [1], recombine=True); g.synchronize()
    faces = [e[1] for e in ex if e[0] == 2]; top, lat_f = faces[0], faces[1:]
    gmsh.model.addPhysicalGroup(3, [[e for e in ex if e[0] == 3][0][1]], 1, 'fluid')
    gmsh.model.addPhysicalGroup(2, [surf, top], 2, 'frontAndBack')
    grp = {}
    for s in lat_f:
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, s); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        gname = 'inlet' if w < 1e-6 and abs(cx - xl) < 1e-6 else 'outlet' if w < 1e-6 and abs(cx - xr) < 1e-6 else \
            'sides' if h < 1e-6 and (abs(cy - yb) < 1e-6 or abs(cy - yt) < 1e-6) else 'body'
        grp.setdefault(gname, []).append(s)
    for i, gname in enumerate(['inlet', 'outlet', 'sides', 'body']): gmsh.model.addPhysicalGroup(2, grp[gname], 10 + i, gname)
    gmsh.model.mesh.generate(3); gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
    gmsh.write(os.path.join(case, 's2d.msh')); gmsh.finalize()


def setup(plane, mesh_only=False):
    pl = PLANES[plane]; case = os.path.join(BASE, plane); V = 6 * KN
    if not mesh_only and os.path.exists(os.path.join(case, 'case2d.json')) and any(d.startswith('processor') for d in os.listdir(case)):
        print(f'{plane}: resuming existing case', flush=True); return case
    shutil.rmtree(case, ignore_errors=True); os.makedirs(case)
    ol, per = outline(pl)
    wf(os.path.join(case, 'system/controlDict'), 'controlDict', 'application pimpleFoam;\nstartFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n')
    wf(os.path.join(case, 'system/fvSchemes'), 'fvSchemes', 'ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n')
    wf(os.path.join(case, 'system/fvSolution'), 'fvSolution', 'solvers {}\n')
    for use_bl in (True, False):
        try:
            build_mesh(case, ol, per, use_bl)
            sh(case, 'gmshToFoam s2d.msh', 'log.gmshToFoam')
            b = os.path.join(case, 'constant/polyMesh/boundary'); s = open(b).read()
            s = re.sub(r'(frontAndBack\s*\{[^}]*?)type\s+patch;', r'\1type empty;', s)
            s = re.sub(r'(body\s*\{[^}]*?)type\s+patch;', r'\1type wall;', s); open(b, 'w').write(s)
            sh(case, 'checkMesh', 'log.checkMesh'); lg = open(os.path.join(case, 'log.checkMesh')).read()
            if 'Mesh OK' in lg and (float(re.search(r'non-orthogonality Max: ([\d.]+)', lg)[1]) < 70): break
        except Exception as ex:
            print('  mesh attempt failed', use_bl, ex)
        shutil.rmtree(os.path.join(case, 'constant'), ignore_errors=True)
    nc = int(re.search(r'cells:\s+(\d+)', lg)[1]); no = float(re.search(r'non-orthogonality Max: ([\d.]+)', lg)[1])
    print(f'{plane}: {nc} cells, BL={use_bl}, maxNonOrtho {no:.0f}, Mesh OK={"Mesh OK" in lg}', flush=True)
    if mesh_only: return case
    t_end = pl['t_end']; wdt = t_end / 400
    json.dump(dict(plane=plane, V=V, t_end=t_end, outline=ol.tolist()), open(os.path.join(case, 'case2d.json'), 'w'))
    k = 1.5 * (0.01 * V) ** 2; om = k / (NU * 10); cons = '#includeEtc "caseDicts/setConstraintTypes"\n'
    wf(os.path.join(case, 'constant/transportProperties'), 'transportProperties', f'transportModel Newtonian;\nnu {NU};\n')
    wf(os.path.join(case, 'constant/turbulenceProperties'), 'turbulenceProperties',
       'simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs off; }\n')

    def fld(name, dim, cls, internal, inlet, outlet, wall):
        wf(os.path.join(case, '0', name), name, f'dimensions {dim};\ninternalField {internal};\nboundaryField\n{{\n    {cons}'
           f'    inlet {{ {inlet} }}\n    outlet {{ {outlet} }}\n    sides {{ type slip; }}\n    body {{ {wall} }}\n'
           f'    frontAndBack {{ type empty; }}\n}}\n', cls=cls, loc='location "0";\n    ')
    fld('U', '[0 1 -1 0 0 0 0]', 'volVectorField', 'uniform (0 0 0)', f'type fixedValue; value uniform ({V} 0 0);',
        'type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);', 'type noSlip;')
    fld('p', '[0 2 -2 0 0 0 0]', 'volScalarField', 'uniform 0', 'type zeroGradient;', 'type fixedValue; value uniform 0;', 'type zeroGradient;')
    fld('k', '[0 2 -2 0 0 0 0]', 'volScalarField', f'uniform {k}', f'type fixedValue; value uniform {k};',
        f'type inletOutlet; inletValue uniform {k}; value uniform {k};', f'type kqRWallFunction; value uniform {k};')
    fld('omega', '[0 0 -1 0 0 0 0]', 'volScalarField', f'uniform {om}', f'type fixedValue; value uniform {om};',
        f'type inletOutlet; inletValue uniform {om}; value uniform {om};', f'type omegaWallFunction; value uniform {om};')
    fld('nut', '[0 2 -1 0 0 0 0]', 'volScalarField', 'uniform 0', 'type calculated; value uniform 0;', 'type calculated; value uniform 0;',
        'type nutUSpaldingWallFunction; value uniform 0;')
    wf(os.path.join(case, 'system/controlDict'), 'controlDict', f"""application pimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {t_end};
deltaT 1e-4; adjustTimeStep yes; maxCo 3; maxDeltaT 0.01;
writeControl adjustableRunTime; writeInterval {wdt}; purgeWrite 0;
writeFormat binary; writePrecision 7; timeFormat general; timePrecision 6; runTimeModifiable true;
""")
    wf(os.path.join(case, 'system/fvSchemes'), 'fvSchemes', """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwindV grad(U); div(phi,k) bounded Gauss upwind; div(phi,omega) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""")
    wf(os.path.join(case, 'system/fvSolution'), 'fvSolution', """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01; }
    pFinal { $p; relTol 0; }
    "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-8; relTol 0.01; }
    "(U|k|omega)Final" { $U; relTol 0; }
}
PIMPLE { nOuterCorrectors 3; nCorrectors 2; nNonOrthogonalCorrectors 1; momentumPredictor yes; }
relaxationFactors { equations { ".*" 1; } }
""")
    wf(os.path.join(case, 'system/decomposeParDict'), 'decomposeParDict',
       'numberOfSubdomains 4;\nmethod hierarchical;\ncoeffs { n (4 1 1); order xyz; }\n')
    return case


def run(case):
    t0 = time.time()
    if not any(d.startswith('processor') for d in os.listdir(case)):
        sh(case, 'decomposePar -force', 'log.decompose')
    cd = os.path.join(case, 'system/controlDict'); t = open(cd).read().replace('startFrom startTime;', 'startFrom latestTime;'); open(cd, 'w').write(t)
    rc = sh(case, 'mpirun --allow-run-as-root --oversubscribe -np 4 pimpleFoam -parallel', 'log.pimpleFoam', 'a')
    sh(case, 'reconstructPar', 'log.reconstruct')
    shutil.rmtree(case, ignore_errors=True) if False else [shutil.rmtree(d, ignore_errors=True) for d in
        [os.path.join(case, x) for x in os.listdir(case) if x.startswith('processor')]]
    print(f'  solver exit {rc}, {(time.time() - t0) / 60:.0f} min', flush=True)


def video(case, plane, fps=30):
    import pyvista as pv
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import LineCollection
    from matplotlib.animation import FFMpegWriter
    import imageio_ffmpeg
    from scipy.interpolate import RegularGridInterpolator
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    pl = PLANES[plane]; meta = json.load(open(os.path.join(case, 'case2d.json'))); V = meta['V']; ol = np.array(meta['outline'])
    bx0, by0 = ol.min(0); bx1, by1 = ol.max(0); Lf, Lc = bx1 - bx0, by1 - by0
    X0, X1, Y0, Y1 = bx0 - 2.5, bx1 + 12, by0 - max(1.6, 0.3 * Lc), by1 + max(1.6, 0.3 * Lc)
    W, Hh = X1 - X0, Y1 - Y0; H = float(np.sqrt(W * Hh / 1.2e6)); nx, ny = int(W / H) + 1, int(Hh / H) + 1
    xs = np.linspace(X0, X1, nx); ys = np.linspace(Y0, Y1, ny)
    open(os.path.join(case, 'case.foam'), 'a').close()
    rd = pv.OpenFOAMReader(os.path.join(case, 'case.foam')); times = [t for t in rd.time_values if t > 0]
    grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(H, H, 1), origin=(X0, Y0, 0.025))
    figw, figh = 16.0, 16.0 * Hh / W + 1.6
    if figh > 9: figw, figh = figw * 9 / figh, 9
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=120, layout='constrained')
    im = ax.imshow(np.zeros((ny, nx)), extent=(X0, X1, Y0, Y1), origin='lower', cmap='turbo', vmin=0, vmax=1.3, interpolation='bilinear', aspect='equal')
    ax.add_patch(Polygon(ol, closed=True, fc='#9aa3ad', ec='black', lw=1.2, zorder=5))
    cb = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.01, aspect=16); cb.set_label('|U| / V', fontsize=12); cb.set_ticks(np.linspace(0, 1.3, 6))
    lab = ['x', 'y', 'z']
    ax.set_xlabel(f'−{lab[pl["a"]]} [m]  (flow left → right, {pl["axis"]} axis)', fontsize=11); ax.set_ylabel(f'{lab[pl["j"]]} [m]', fontsize=11)
    ax.set_title(f'{pl["axis"].capitalize()}, 6 kn (3.09 m/s): flow development from rest, mid-plane {lab[pl["n"]]} = 0, 2-D unsteady RANS (illustration only)', fontsize=11)
    txt = ax.text(0.012, 0.93, '', transform=ax.transAxes, fontsize=12, family='monospace', zorder=9, bbox=dict(fc='white', ec='none', alpha=0.85, pad=3))
    NP, NT = int(np.clip(W * Hh * 55, 1500, 4500)), 10
    rng = np.random.default_rng(1); alpha = np.linspace(0.05, 0.95, NT - 1)
    lc = LineCollection([], linewidths=1.0, zorder=6); ax.add_collection(lc)
    P = np.c_[rng.uniform(X0, X1, NP), rng.uniform(Y0, Y1, NP)]; age = rng.integers(0, 200, NP); life = rng.integers(160, 320, NP)
    hist = np.repeat(P[None], NT, 0); tprev = 0.0
    wr = FFMpegWriter(fps=fps, bitrate=7000, codec='libx264', extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p'])
    os.makedirs(OUTV, exist_ok=True); out = os.path.join(OUTV, f'anim_development_{plane}_2d.mp4')
    with wr.saving(fig, out, dpi=120):
        for i, t in enumerate(times):
            rd.set_active_time_value(t); m = rd.read()['internalMesh'].cell_data_to_point_data(); s = grid.sample(m)
            U = np.asarray(s.point_data['U']); ok = np.asarray(s.point_data['vtkValidPointMask']) > 0
            Ux = np.where(ok, U[:, 0], 0).reshape(ny, nx); Uy = np.where(ok, U[:, 1], 0).reshape(ny, nx); body = ~ok.reshape(ny, nx)
            im.set_data(np.ma.masked_array(np.linalg.norm(np.stack([Ux, Uy]), axis=0) / V, mask=body))
            iu = RegularGridInterpolator((ys, xs), Ux, bounds_error=False, fill_value=0); iv = RegularGridInterpolator((ys, xs), Uy, bounds_error=False, fill_value=0)
            ib = RegularGridInterpolator((ys, xs), body.astype(float), bounds_error=False, fill_value=1)
            dt = t - tprev; tprev = t; q = P[:, ::-1]
            k1 = np.c_[iu(q), iv(q)]; q2 = (P + 0.5 * dt * k1)[:, ::-1]; P = P + dt * np.c_[iu(q2), iv(q2)]; age += 1
            dead = (P[:, 0] < X0) | (P[:, 0] > X1) | (P[:, 1] < Y0) | (P[:, 1] > Y1) | (ib(P[:, ::-1]) > 0.5) | (age > life)
            if dead.any():
                nd = int(dead.sum()); np_ = np.c_[rng.uniform(X0, X1, nd), rng.uniform(Y0, Y1, nd)]
                up = rng.random(nd) < 0.5; np_[up, 0] = X0 + rng.uniform(0, 0.02 * W, up.sum()); P[dead] = np_; age[dead] = 0; hist[:, dead] = np_
            hist = np.roll(hist, -1, 0); hist[-1] = P
            segs = np.stack([hist[:-1], hist[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
            cols = np.ones((len(segs), 4)); cols[:, 3] = np.tile(alpha, NP); lc.set_segments(segs); lc.set_color(cols)
            txt.set_text(f't = {t:5.2f} s    distance travelled = {V * t / Lf:6.1f} body lengths')
            wr.grab_frame()
    plt.close(fig); print(f'  video {out}', flush=True)


def cleanup(case):
    for d in os.listdir(case):
        if re.fullmatch(r'\d+(\.\d+)?(e-?\d+)?', d) and d != '0': shutil.rmtree(os.path.join(case, d), ignore_errors=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'video': video(os.path.abspath(sys.argv[2]), sys.argv[3])
    else:
        for p in sys.argv[2:]:
            c = setup(p, mesh_only=(cmd == 'mesh'))
            if cmd == 'all': run(c); video(c, p); cleanup(c)
