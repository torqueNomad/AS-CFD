"""2-D unsteady-RANS impulsive-start case of the mid-span sway section (illustration of wake development only).
The body outline is the z = 0 (mid-length) cut of the sealed envelope, in the same frame as the 3-D runs
(centred, x = sway, y = heave). Flow runs towards +x, so the outline is mirrored in x, exactly as in the sway
slice figures. Fluid starts at rest; the free stream is imposed at t = 0.
Usage: transient2d_setup.py <case_dir> [speed_kn=6] [t_end=12] [write_dt=0.025]
Outputs a complete OpenFOAM case (gmsh mesh -> gmshToFoam, pimpleFoam, k-omega SST, Spalding wall function)."""
import os, sys, json, subprocess
import numpy as np, trimesh
import gmsh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO, NU, KN = 1025.9, 1.1883e-6, 0.514444
HDR = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    %sobject %s;\n}\n"


def w(path, obj, body, cls='dictionary', loc=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write(HDR % (cls, loc, obj) + body)


def section_outline():
    e = trimesh.load(os.path.join(ROOT, 'geometry', 'shell_envelope.stl'))
    e.apply_translation(-e.bounds.mean(0))
    s = e.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    loops = s.discrete
    p = max(loops, key=lambda q: len(q))[:, :2]
    p = np.c_[-p[:, 0], p[:, 1]]                       # mirror x: flow towards +x
    # resample to uniform arclength, 360 points, closed
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1))]
    t = np.linspace(0, d[-1], 361)[:-1]
    q = np.c_[np.interp(t, d, np.r_[p[:, 0], p[0, 0]]), np.interp(t, d, np.r_[p[:, 1], p[0, 1]])]
    return q


def build_mesh(case, outline):
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add('sway2d')
    occ = gmsh.model.geo
    pts = [occ.addPoint(x, y, 0, 0.012) for x, y in outline]
    n = len(pts); sp = []
    k = 36                                              # 10 splines of ~36 points keep the outline smooth
    for i in range(0, n, k):
        seg = pts[i:i + k + 1] if i + k < n else pts[i:] + [pts[0]]
        sp.append(occ.addSpline(seg))
    body = occ.addCurveLoop(sp)
    xl, xr, yb, yt = -6.0, 14.0, -4.5, 4.5
    c = [occ.addPoint(xl, yb, 0, 0.5), occ.addPoint(xr, yb, 0, 0.5), occ.addPoint(xr, yt, 0, 0.5), occ.addPoint(xl, yt, 0, 0.5)]
    L = [occ.addLine(c[0], c[1]), occ.addLine(c[1], c[2]), occ.addLine(c[2], c[3]), occ.addLine(c[3], c[0])]
    outer = occ.addCurveLoop(L)
    surf = occ.addPlaneSurface([outer, body])
    occ.synchronize()
    # size field: fine at the body, a refined wake box, coarse far away
    f = gmsh.model.mesh.field
    f.add('Distance', 1); f.setNumbers(1, 'CurvesList', sp); f.setNumber(1, 'Sampling', 400)
    f.add('Threshold', 2); f.setNumber(2, 'InField', 1); f.setNumber(2, 'SizeMin', 0.012); f.setNumber(2, 'SizeMax', 0.45)
    f.setNumber(2, 'DistMin', 0.08); f.setNumber(2, 'DistMax', 4.0)
    f.add('Box', 3); f.setNumber(3, 'VIn', 0.04); f.setNumber(3, 'VOut', 0.45)
    f.setNumber(3, 'XMin', 0.0); f.setNumber(3, 'XMax', 9.0); f.setNumber(3, 'YMin', -1.4); f.setNumber(3, 'YMax', 1.4)
    f.setNumber(3, 'Thickness', 1.5)
    f.add('Min', 4); f.setNumbers(4, 'FieldsList', [2, 3]); f.setAsBackgroundMesh(4)
    # boundary layer: 8 layers, first cell 3 mm, growth 1.2 (y+ ~ 100-250 at 6 kn; Spalding wall function)
    f.add('BoundaryLayer', 5)
    f.setNumbers(5, 'CurvesList', sp); f.setNumber(5, 'Size', 0.003); f.setNumber(5, 'Ratio', 1.2)
    f.setNumber(5, 'Quads', 1); f.setNumber(5, 'Thickness', 0.03)
    f.setNumber(5, 'IntersectMetrics', 0)
    f.setAsBoundaryLayer(5)
    gmsh.option.setNumber('Mesh.Algorithm', 6); gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0); gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    # one cell thick in z, then physical groups
    ex = gmsh.model.geo.extrude([(2, surf)], 0, 0, 0.05, [1], recombine=True)
    gmsh.model.geo.synchronize()
    sides = {e[1]: e for e in ex}
    vol = [e for e in ex if e[0] == 3][0][1]
    top = [e for e in ex if e[0] == 2][0][1]
    lat = [e[1] for e in ex if e[0] == 2][1:]
    gmsh.model.addPhysicalGroup(3, [vol], 1, 'fluid')
    gmsh.model.addPhysicalGroup(2, [surf, top], 2, 'frontAndBack')
    # lateral faces appear in curve order: outer rectangle (4) then body splines
    bb = {}
    for s in lat:
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if abs(x1 - x0) < 1e-6 and abs(cx - xl) < 1e-6: g = 'inlet'
        elif abs(x1 - x0) < 1e-6 and abs(cx - xr) < 1e-6: g = 'outlet'
        elif abs(y1 - y0) < 1e-6 and (abs(cy - yb) < 1e-6 or abs(cy - yt) < 1e-6): g = 'sides'
        else: g = 'body'
        bb.setdefault(g, []).append(s)
    for i, g in enumerate(['inlet', 'outlet', 'sides', 'body']):
        gmsh.model.addPhysicalGroup(2, bb[g], 10 + i, g)
    gmsh.model.mesh.generate(3)
    os.makedirs(case, exist_ok=True)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
    gmsh.write(os.path.join(case, 'sway2d.msh'))
    nel = len(gmsh.model.mesh.getElementsByType(6)[0]) + len(gmsh.model.mesh.getElementsByType(5)[0])
    gmsh.finalize()
    return nel


def main(case, kn=6.0, t_end=12.0, write_dt=0.025):
    V = kn * KN
    outline = section_outline()
    json.dump(dict(kn=kn, V=V, t_end=t_end, write_dt=write_dt, outline=outline.tolist(), nu=NU, rho=RHO),
              open(os.path.join(case if os.path.isdir(case) else (os.makedirs(case) or case), 'case2d.json'), 'w'))
    n = build_mesh(case, outline)
    print('mesh elements', n)
    I, vr = 0.01, 10.0
    k = 1.5 * (I * V) ** 2; om = k / (NU * vr)
    w(os.path.join(case, 'constant', 'transportProperties'), 'transportProperties', f'transportModel Newtonian;\nnu {NU};\n')
    w(os.path.join(case, 'constant', 'turbulenceProperties'), 'turbulenceProperties', 'simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs off; }\n')
    cons = '#includeEtc "caseDicts/setConstraintTypes"\n'
    def fld(name, dim, cls, internal, inlet, outlet, body, sides='slip'):
        w(os.path.join(case, '0', name), name,
          f'dimensions {dim};\ninternalField {internal};\nboundaryField\n{{\n    {cons}'
          f'    inlet {{ {inlet} }}\n    outlet {{ {outlet} }}\n    sides {{ {sides} }}\n    body {{ {body} }}\n'
          f'    frontAndBack {{ type empty; }}\n}}\n', cls=cls, loc='location "0";\n    ')
    fld('U', '[0 1 -1 0 0 0 0]', 'volVectorField', 'uniform (0 0 0)', f'type fixedValue; value uniform ({V} 0 0);',
        'type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);', 'type noSlip;', 'type slip;')
    fld('p', '[0 2 -2 0 0 0 0]', 'volScalarField', 'uniform 0', 'type zeroGradient;', 'type fixedValue; value uniform 0;', 'type zeroGradient;', 'type slip;')
    fld('k', '[0 2 -2 0 0 0 0]', 'volScalarField', f'uniform {k}', f'type fixedValue; value uniform {k};',
        f'type inletOutlet; inletValue uniform {k}; value uniform {k};', f'type kqRWallFunction; value uniform {k};', 'type slip;')
    fld('omega', '[0 0 -1 0 0 0 0]', 'volScalarField', f'uniform {om}', f'type fixedValue; value uniform {om};',
        f'type inletOutlet; inletValue uniform {om}; value uniform {om};', f'type omegaWallFunction; value uniform {om};', 'type slip;')
    fld('nut', '[0 2 -1 0 0 0 0]', 'volScalarField', 'uniform 0', 'type calculated; value uniform 0;', 'type calculated; value uniform 0;',
        'type nutUSpaldingWallFunction; value uniform 0;', 'type slip;')
    w(os.path.join(case, 'system', 'controlDict'), 'controlDict', f"""application pimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {t_end};
deltaT 1e-4; adjustTimeStep yes; maxCo 3; maxDeltaT 0.01;
writeControl adjustableRunTime; writeInterval {write_dt}; purgeWrite 0;
writeFormat binary; writePrecision 7; timeFormat general; timePrecision 6; runTimeModifiable true;
functions
{{
    forces {{ type forces; libs ("libforces.so"); writeControl timeStep; writeInterval 5; patches (body); rho rhoInf; rhoInf {RHO}; CofR (0 0 0); log false; }}
}}
""")
    w(os.path.join(case, 'system', 'fvSchemes'), 'fvSchemes', """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwindV grad(U); div(phi,k) bounded Gauss upwind; div(phi,omega) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""")
    w(os.path.join(case, 'system', 'fvSolution'), 'fvSolution', """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01; }
    pFinal { $p; relTol 0; }
    "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-8; relTol 0.01; }
    "(U|k|omega)Final" { $U; relTol 0; }
}
PIMPLE { nOuterCorrectors 3; nCorrectors 2; nNonOrthogonalCorrectors 1; momentumPredictor yes; }
relaxationFactors { equations { ".*" 1; } }
""")


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]), *[float(a) for a in sys.argv[2:]])
