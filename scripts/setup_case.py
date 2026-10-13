"""Generate an OpenFOAM (v1912, simpleFoam, k-omega SST) case for one motion axis.

Body frame (from CAD): Z = surge, X = sway, Y = heave.  Body is moved to put its
bounding-box centre at the origin. Motion is in the +axis direction, so the
free stream in the body frame is U = -V * e_axis.

Usage: setup_case.py <case_dir> <surge|sway|heave> <mesh: coarse|medium|fine> <speed_kn> [nIter]
"""
import os, sys, json, shutil, textwrap
import numpy as np, trimesh
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Standard sea water, 15 degC, 35 g/kg: ITTC 7.5-02-01-03 (2011), Fresh Water and Seawater Properties
RHO, NU = 1025.9, 1.1883e-6
KN = 0.514444  # m/s per knot
AX = {'sway': 0, 'heave': 1, 'surge': 2}
# background cell size [m] per mesh level (refinement levels are fixed => surface cell = base/2^5)
BASE = {'coarse': 0.64, 'medium': 0.48, 'fine': 0.36}

HDR = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    {loc}object      {obj};
}}
"""

def w(path, obj, body, cls='dictionary', loc=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(HDR.format(cls=cls, obj=obj, loc=loc) + textwrap.dedent(body))


def projected_area(mesh, a, px=0.004):
    """Silhouette area normal to axis a, by rasterising densely sampled surface points."""
    p, _ = trimesh.sample.sample_surface_even(mesh, int(mesh.area / (px * 0.4) ** 2))
    q = p[:, [i for i in range(3) if i != a]]
    ij = np.floor((q - q.min(0)) / px).astype(int) + 2
    g = np.zeros(ij.max(0) + 3, bool); g[tuple(ij.T)] = True
    g = ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=2))
    return float(g.sum() * px * px)


def main(case, axis, mesh, kn, niter=1500):
    a = AX[axis]
    V = kn * KN
    e = trimesh.load(os.path.join(ROOT, 'geometry', 'shell_envelope.stl'))
    c = e.bounds.mean(0)
    e.apply_translation(-c)
    ext = e.extents
    os.makedirs(os.path.join(case, 'constant', 'triSurface'), exist_ok=True)
    e.export(os.path.join(case, 'constant', 'triSurface', 'body.stl'))

    # projected (frontal) area normal to the motion axis
    others = [i for i in range(3) if i != a]
    Aproj = projected_area(e, a)

    # ---- domain: upstream 1.5 L_flow+5, downstream ~3x, lateral >= 5 x body size
    lo, hi = np.zeros(3), np.zeros(3)
    for i in range(3):
        h = ext[i] / 2
        if i == a:  # flow comes from +a side, goes to -a
            lo[i], hi[i] = -(h + max(15.0, 3 * ext[i])), h + max(8.0, 1.5 * ext[i])
        else:
            lo[i], hi[i] = -(h + 6.0), h + 6.0
    b = BASE[mesh]
    n = np.ceil((hi - lo) / b).astype(int)
    hi = lo + n * b

    names = ['x', 'y', 'z']
    face = {  # blockMesh hex faces
        ('x', 0): '(0 4 7 3)', ('x', 1): '(1 2 6 5)', ('y', 0): '(0 1 5 4)', ('y', 1): '(3 7 6 2)',
        ('z', 0): '(0 3 2 1)', ('z', 1): '(4 5 6 7)'}
    inlet = face[(names[a], 1)]; outlet = face[(names[a], 0)]
    sides = [face[(names[i], s)] for i in others for s in (0, 1)]
    w(os.path.join(case, 'system', 'blockMeshDict'), 'blockMeshDict', f"""
    scale 1;
    vertices
    (
        ({lo[0]} {lo[1]} {lo[2]}) ({hi[0]} {lo[1]} {lo[2]}) ({hi[0]} {hi[1]} {lo[2]}) ({lo[0]} {hi[1]} {lo[2]})
        ({lo[0]} {lo[1]} {hi[2]}) ({hi[0]} {lo[1]} {hi[2]}) ({hi[0]} {hi[1]} {hi[2]}) ({lo[0]} {hi[1]} {hi[2]})
    );
    blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
    boundary
    (
        inlet  {{ type patch; faces ( {inlet} ); }}
        outlet {{ type patch; faces ( {outlet} ); }}
        sides  {{ type patch; faces ( {' '.join(sides)} ); }}
    );
    """)

    # refinement boxes: near box around body (level 2) and wake box (level 3)
    m1 = ext / 2 + 0.6; m2 = ext / 2 + 0.25
    nb_lo, nb_hi = -m1.copy(), m1.copy(); nb_lo[a] -= 1.5 * ext[a] + 3.0
    wk_lo, wk_hi = -m2.copy(), m2.copy(); wk_lo[a] -= 0.6 * ext[a] + 1.2
    v3 = lambda v: f"({v[0]:.4f} {v[1]:.4f} {v[2]:.4f})"
    loc = np.zeros(3); loc[others[0]] = lo[others[0]] + 0.5 * b + 0.0123; loc[others[1]] = lo[others[1]] + 0.5 * b + 0.0117; loc[a] = 0.0131
    w(os.path.join(case, 'system', 'snappyHexMeshDict'), 'snappyHexMeshDict', f"""
    castellatedMesh true;
    snap            true;
    addLayers       true;
    geometry
    {{
        body.stl {{ type triSurfaceMesh; name body; }}
        near {{ type searchableBox; min {v3(nb_lo)}; max {v3(nb_hi)}; }}
        wake {{ type searchableBox; min {v3(wk_lo)}; max {v3(wk_hi)}; }}
    }}
    castellatedMeshControls
    {{
        maxLocalCells 4000000;
        maxGlobalCells 12000000;
        minRefinementCells 10;
        maxLoadUnbalance 0.10;
        nCellsBetweenLevels 3;
        features ( {{ file "body.eMesh"; level 6; }} );
        refinementSurfaces {{ body {{ level (5 5); patchInfo {{ type wall; }} }} }}
        resolveFeatureAngle 30;
        refinementRegions
        {{
            near {{ mode inside; levels ((1E15 2)); }}
            wake {{ mode inside; levels ((1E15 3)); }}
        }}
        locationInMesh {v3(loc)};
        allowFreeStandingZoneFaces true;
    }}
    snapControls
    {{
        nSmoothPatch 3;
        tolerance 2.0;
        nSolveIter 50;
        nRelaxIter 5;
        nFeatureSnapIter 10;
        implicitFeatureSnap false;
        explicitFeatureSnap true;
        multiRegionFeatureSnap false;
    }}
    addLayersControls
    {{
        relativeSizes true;
        layers {{ body {{ nSurfaceLayers 3; }} }}
        expansionRatio 1.2;
        finalLayerThickness 0.35;
        minThickness 0.08;
        nGrow 0;
        featureAngle 130;
        slipFeatureAngle 30;
        nRelaxIter 5;
        nSmoothSurfaceNormals 1;
        nSmoothNormals 3;
        nSmoothThickness 10;
        maxFaceThicknessRatio 0.5;
        maxThicknessToMedialRatio 0.3;
        minMedialAxisAngle 90;
        nBufferCellsNoExtrude 0;
        nLayerIter 50;
    }}
    meshQualityControls
    {{
        #include "meshQualityDict"
        nSmoothScale 4;
        errorReduction 0.75;
    }}
    mergeTolerance 1e-6;
    """)
    w(os.path.join(case, 'system', 'meshQualityDict'), 'meshQualityDict', """
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80;
    minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001;
    minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1;
    """)
    w(os.path.join(case, 'system', 'surfaceFeatureExtractDict'), 'surfaceFeatureExtractDict', """
    body.stl
    {
        extractionMethod extractFromSurface;
        extractFromSurfaceCoeffs { includedAngle 150; }
        writeObj no;
    }
    """)
    w(os.path.join(case, 'system', 'decomposeParDict'), 'decomposeParDict', """
    numberOfSubdomains 4;
    method hierarchical;
    coeffs { n (1 1 4); order xyz; }
    """)

    # ---- physics
    Uvec = np.zeros(3); Uvec[a] = -V
    I, visc_ratio = 0.01, 10.0
    k = 1.5 * (I * V) ** 2
    om = k / (NU * visc_ratio)
    dd = np.zeros(3); dd[a] = -1.0  # drag acts along the flow (-axis)
    w(os.path.join(case, 'constant', 'transportProperties'), 'transportProperties', f"""
    transportModel Newtonian;
    nu {NU};
    """)
    w(os.path.join(case, 'constant', 'turbulenceProperties'), 'turbulenceProperties', """
    simulationType RAS;
    RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }
    """)
    U3 = v3(Uvec)
    w(os.path.join(case, '0.orig', 'U'), 'U', f"""
    dimensions [0 1 -1 0 0 0 0];
    internalField uniform {U3};
    boundaryField
    {{
        #includeEtc "caseDicts/setConstraintTypes"
        inlet  {{ type fixedValue; value uniform {U3}; }}
        outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform {U3}; }}
        sides  {{ type slip; }}
        body   {{ type noSlip; }}
    }}
    """, cls='volVectorField', loc='location "0";\n    ')
    w(os.path.join(case, '0.orig', 'p'), 'p', """
    dimensions [0 2 -2 0 0 0 0];
    internalField uniform 0;
    boundaryField
    {
        #includeEtc "caseDicts/setConstraintTypes"
        inlet  { type zeroGradient; }
        outlet { type fixedValue; value uniform 0; }
        sides  { type slip; }
        body   { type zeroGradient; }
    }
    """, cls='volScalarField', loc='location "0";\n    ')
    for fld, val, dim, wall in [('k', k, '[0 2 -2 0 0 0 0]', f'kqRWallFunction; value uniform {k}'),
                                ('omega', om, '[0 0 -1 0 0 0 0]', f'omegaWallFunction; value uniform {om}'),
                                ('nut', 0, '[0 2 -1 0 0 0 0]', 'nutUSpaldingWallFunction; value uniform 0')]:
        inl = f'fixedValue; value uniform {val}' if fld != 'nut' else 'calculated; value uniform 0'
        out = f'inletOutlet; inletValue uniform {val}; value uniform {val}' if fld != 'nut' else 'calculated; value uniform 0'
        w(os.path.join(case, '0.orig', fld), fld, f"""
        dimensions {dim};
        internalField uniform {val};
        boundaryField
        {{
            #includeEtc "caseDicts/setConstraintTypes"
            inlet  {{ type {inl}; }}
            outlet {{ type {out}; }}
            sides  {{ type slip; }}
            body   {{ type {wall}; }}
        }}
        """, cls='volScalarField', loc='location "0";\n    ')

    w(os.path.join(case, 'system', 'controlDict'), 'controlDict', f"""
    application simpleFoam;
    startFrom latestTime;
    startTime 0;
    stopAt endTime;
    endTime {niter};
    deltaT 1;
    writeControl timeStep;
    writeInterval {niter};
    purgeWrite 1;
    writeFormat binary;
    writePrecision 8;
    writeCompression off;
    timeFormat general;
    timePrecision 6;
    runTimeModifiable true;
    functions
    {{
        forces
        {{
            type forces;
            libs ("libforces.so");
            writeControl timeStep;
            writeInterval 1;
            patches (body);
            rho rhoInf;
            rhoInf {RHO};
            CofR (0 0 0);
            log false;
        }}
        yPlus
        {{
            type yPlus;
            libs ("libfieldFunctionObjects.so");
            writeControl writeTime;
        }}
        residuals
        {{
            type solverInfo;
            libs ("libutilityFunctionObjects.so");
            fields (U p k omega);
        }}
    }}
    """)
    w(os.path.join(case, 'system', 'fvSchemes'), 'fvSchemes', """
    ddtSchemes { default steadyState; }
    gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; grad(k) cellLimited Gauss linear 1; grad(omega) cellLimited Gauss linear 1; }
    divSchemes
    {
        default none;
        div(phi,U) bounded Gauss linearUpwindV grad(U);
        div(phi,k) bounded Gauss upwind;
        div(phi,omega) bounded Gauss upwind;
        div((nuEff*dev2(T(grad(U))))) Gauss linear;
    }
    laplacianSchemes { default Gauss linear limited corrected 0.5; }
    interpolationSchemes { default linear; }
    snGradSchemes { default limited corrected 0.5; }
    wallDist { method meshWave; }
    """)
    w(os.path.join(case, 'system', 'fvSolution'), 'fvSolution', """
    solvers
    {
        p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; nCellsInCoarsestLevel 50; }
        "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-8; relTol 0.1; }
    }
    SIMPLE
    {
        nNonOrthogonalCorrectors 0;
        consistent yes;
        residualControl { p 1e-5; U 1e-6; "(k|omega)" 1e-6; }
    }
    relaxationFactors
    {
        equations { U 0.7; k 0.5; omega 0.5; }
        fields { p 0.5; }
    }
    """)
    meta = dict(axis=axis, mesh=mesh, kn=kn, V=V, U=Uvec.tolist(), dragDir=dd.tolist(), rho=RHO, nu=NU,
                Aproj=Aproj, extents=ext.tolist(), L_axis=float(ext[a]), centre_shift_m=(-c).tolist(),
                domain=[lo.tolist(), hi.tolist()], base=b, k=k, omega=om)
    json.dump(meta, open(os.path.join(case, 'case.json'), 'w'), indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1500)
