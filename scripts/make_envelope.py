"""Build a closed, watertight outer envelope of the shell.
All openings (through-holes, side cut-outs, open ends) are treated as blind/closed, as instructed.
Method (body is a pod elongated along Z with star-shaped cross sections):
  1. slice original mesh every dz along Z
  2. per slice, outer radius r(theta) about the section centroid axis = max radius of section points per angular bin
  3. bins with no skin (openings) are filled by linear interpolation along Z, then around theta
  4. loft the r(z,theta) grid into a surface and cap both ends flat
Usage: make_envelope.py in.stl out.stl   (input mm, output m)"""
import sys, trimesh, numpy as np
from scipy import ndimage

src, dst = sys.argv[1], sys.argv[2]
NT, DZ = 240, 10.0
m = trimesh.load(src)
z0, z1 = m.bounds[:, 2]
zs = np.arange(z0 + 0.5, z1 - 0.5, DZ); zs = np.append(zs, z1 - 0.5)
cx, cy = m.bounds[:, 0].mean(), m.bounds[:, 1].mean()
segs, _, _ = trimesh.intersections.mesh_multiplane(m, [0, 0, 0], [0, 0, 1], zs)
th_e = np.linspace(-np.pi, np.pi, NT + 1)
R = np.full((len(zs), NT), np.nan)
for k, s in enumerate(segs):
    if len(s) == 0: continue
    # densify segments to 2 mm so narrow arcs fill their angular bins
    a, b = s[:, 0, :], s[:, 1, :]
    n = np.maximum(1, (np.linalg.norm(b - a, axis=1) / 2).astype(int))
    t = np.concatenate([np.linspace(0, 1, i + 1) for i in n]); ia = np.repeat(np.arange(len(a)), n + 1)
    p = a[ia] + (b[ia] - a[ia]) * t[:, None]
    th = np.arctan2(p[:, 1] - cy, p[:, 0] - cx); r = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    bi = np.clip(np.digitize(th, th_e) - 1, 0, NT - 1)
    out = np.full(NT, np.nan)
    for j in range(NT):
        rr = r[bi == j]
        if len(rr): out[j] = rr.max()
    R[k] = out
miss0 = np.isnan(R).copy()
# outer radius should be continuous: discard bins whose max came from interior members only (much smaller than neighbours)
med = ndimage.median_filter(np.nan_to_num(R, nan=0), size=(1, 9), mode='wrap')
R[(R < 0.85 * med)] = np.nan
miss = np.isnan(R)
# interpolate along z per theta
for j in range(NT):
    c = R[:, j]; ok = ~np.isnan(c)
    if ok.sum() >= 2: R[:, j] = np.interp(zs, zs[ok], c[ok])
# then around theta (periodic)
for k in range(len(zs)):
    c = R[k]; ok = ~np.isnan(c)
    if 0 < ok.sum() < NT:
        tt = np.arange(NT)
        R[k] = np.interp(tt, np.concatenate([tt[ok] - NT, tt[ok], tt[ok] + NT]), np.tile(c[ok], 3))
R = ndimage.gaussian_filter(R, sigma=(0.8, 0.8), mode=('nearest', 'wrap'))
print(f'slices {len(zs)}, theta bins {NT}, filled (opening) fraction {miss.mean()*100:.1f} %')
np.savez(dst.replace('.stl', '_rgrid.npz'), zs=zs, R=R, miss=miss)
thc = 0.5 * (th_e[:-1] + th_e[1:])
X = cx + R * np.cos(thc); Y = cy + R * np.sin(thc); Z = np.repeat(zs[:, None], NT, 1)
V = np.stack([X, Y, Z], -1).reshape(-1, 3)
F = []
for k in range(len(zs) - 1):
    for j in range(NT):
        a = k * NT + j; b = k * NT + (j + 1) % NT; c = (k + 1) * NT + (j + 1) % NT; d = (k + 1) * NT + j
        F += [[a, b, c], [a, c, d]]
V = np.vstack([V, [cx, cy, zs[0]], [cx, cy, zs[-1]]]); i0, i1 = len(V) - 2, len(V) - 1
L = (len(zs) - 1) * NT
for j in range(NT):
    F += [[i0, (j + 1) % NT, j], [i1, L + j, L + (j + 1) % NT]]
out = trimesh.Trimesh(V * 1e-3, np.array(F), process=True)
out.fix_normals()
print('watertight', out.is_watertight, 'faces', len(out.faces), 'bounds m', out.bounds.round(4).tolist(),
      'vol m3', round(out.volume, 4), 'wetted area m2', round(out.area, 4))
out.export(dst)
