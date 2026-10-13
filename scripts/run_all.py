"""Batch driver: one mesh per axis, then one simpleFoam run per speed (each run re-uses the mesh).
Usage: run_all.py <mesh: coarse|medium|fine> <axes comma list> <speeds_kn comma list> [nIter]
Cases are written to cases/<axis>_<mesh>/ (mesh) and cases/<axis>_<mesh>/V<kn>kn/ (runs)."""
import os, sys, shutil, subprocess, glob, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, 'scripts')


def sh(cmd):
    print('+', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main(mesh, axes, speeds, niter):
    for axis in axes:
        mcase = os.path.join(ROOT, 'cases', f'{axis}_{mesh}')
        if not os.path.exists(os.path.join(mcase, 'processor0', 'constant', 'polyMesh', 'faces')):
            sh(f'python3 {S}/setup_case.py {mcase} {axis} {mesh} {speeds[0]} {niter}')
            sh(f'bash {S}/mesh_case.sh {mcase}')
        for kn in speeds:
            run = os.path.join(mcase, f'V{kn:g}kn')
            if os.path.exists(os.path.join(run, 'DONE')):
                continue
            if os.path.exists(run):
                shutil.rmtree(run)
            # fresh case files for this speed, mesh copied from the meshed case
            sh(f'python3 {S}/setup_case.py {run} {axis} {mesh} {kn} {niter} > /dev/null')
            for p in sorted(glob.glob(os.path.join(mcase, 'processor*'))):
                q = os.path.join(run, os.path.basename(p))
                shutil.copytree(os.path.join(p, 'constant'), os.path.join(q, 'constant'))
                shutil.copytree(os.path.join(run, '0.orig'), os.path.join(q, '0'))
            sh(f'bash {S}/run_case.sh {run}')
            # keep only what post-processing needs: logs, forces, yPlus, case.json and the final fields
            open(os.path.join(run, 'DONE'), 'w').write('ok\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(','), [float(v) for v in sys.argv[3].split(',')],
         int(sys.argv[4]) if len(sys.argv) > 4 else 1500)
