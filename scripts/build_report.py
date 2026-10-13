"""Build the HTML report (report/index.html + report/assets/*) from results/ and the figures/animations.
Images are converted to WebP (max 1800 px wide), videos re-encoded to H.264 1280 px for the web page.
Usage: python3 build_report.py"""
import os, json, csv, glob, html, subprocess, math, datetime
import numpy as np
from PIL import Image
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'report'); AS = os.path.join(OUT, 'assets'); os.makedirs(AS, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
AXES = ['surge', 'sway', 'heave']
AXN = {'surge': ('Surge', 'Z', '+Z (bow first)'), 'sway': ('Sway', 'X', '+X (beam on)'), 'heave': ('Heave', 'Y', '+Y (top face first)')}
KN = 0.514444


def img(src, name, w=1800):
    dst = os.path.join(AS, name)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        im = Image.open(src).convert('RGB')
        if im.width > w: im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        im.save(dst, 'WEBP', quality=86, method=6)
    return 'assets/' + name


def vid(src, name, w=1280):
    dst = os.path.join(AS, name)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        subprocess.run([FF, '-loglevel', 'error', '-y', '-i', src, '-vf', f'scale={w}:-2', '-c:v', 'libx264', '-crf', '27',
                        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', dst], check=True)
    poster = name.replace('.mp4', '.webp')
    pdst = os.path.join(AS, poster)
    if not os.path.exists(pdst) or os.path.getmtime(pdst) < os.path.getmtime(dst):
        tmp = os.path.join(AS, '_p.png')
        subprocess.run([FF, '-loglevel', 'error', '-y', '-ss', '1', '-i', dst, '-frames:v', '1', tmp], check=True)
        Image.open(tmp).convert('RGB').save(pdst, 'WEBP', quality=80); os.remove(tmp)
    return 'assets/' + name, 'assets/' + poster


def esc(s): return html.escape(str(s))


def fmt_n(v, dp=0):
    return f'{v:,.{dp}f}'


def main():
    runs = json.load(open(os.path.join(RES, 'runs.json')))
    fit = json.load(open(os.path.join(RES, 'fit_coefficients.json')))
    table = list(csv.DictReader(open(os.path.join(RES, 'drag_power_table.csv'))))
    yp = list(csv.DictReader(open(os.path.join(RES, 'yplus_summary.csv'))))
    R = {(r['axis'], r['mesh'], r['kn']): r for r in runs}
    cells = {}
    for ax in AXES:
        for m in ('coarse', 'medium', 'fine'):
            f = os.path.join(ROOT, 'cases', f'{ax}_{m}', 'log.checkMesh')
            if os.path.exists(f):
                for line in open(f):
                    if 'cells:' in line: cells[(ax, m)] = int(line.split()[-1]); break
    rho, nu = 1025.9, 1.1883e-6

    # ---- data for the chart: fitted curves on 0..6 kn (0.1 kn), CFD points (grid-corrected)
    curves = {}
    for ax in AXES:
        t = fit[ax]; kk = np.round(np.arange(0, 6.0001, 0.1), 2); D = []
        for k in kk:
            V = k * KN
            if V == 0: D.append(0.0); continue
            Cd = (t['Cd_fit_c0'] + t['Cd_fit_c1_per_log10Re'] * math.log10(V * t['L_ref_m'] / nu)) * t['grid_correction']
            D.append(0.5 * rho * V * V * t['A_proj_m2'] * Cd)
        pts = [[r['kn'], r['D'] * t['grid_correction']] for r in runs if r['axis'] == ax and r['mesh'] == 'coarse']
        curves[ax] = dict(kn=kk.tolist(), D=[round(d, 2) for d in D], pts=sorted(pts))
    six = {}
    for ax in AXES:
        row = [r for r in table if r['speed_kn'] == '6'][0]
        six[ax] = dict(D=float(row[f'{ax}_drag_N']), P=float(row[f'{ax}_power_W']), Cd=float(row[f'{ax}_Cd']),
                       A=fit[ax]['A_proj_m2'], cdmin=fit[ax]['Cd_runs_min'], cdmax=fit[ax]['Cd_runs_max'])
        rm = R.get((ax, 'fine', 6.0)) or R.get((ax, 'medium', 6.0)) or R.get((ax, 'coarse', 6.0))
        six[ax]['pf'] = f"{rm['D_pressure'] / rm['D'] * 100:.0f} % / {rm['D_viscous'] / rm['D'] * 100:.0f} %"

    # ---- grid study
    grid_rows = []
    gci = None
    for ax in AXES:
        for m in ('coarse', 'medium', 'fine'):
            r = R.get((ax, m, 6.0))
            if r: grid_rows.append((ax, m, cells.get((ax, m)), r['D'], r['D_std'], r['Cd']))
    sf = [R.get(('surge', m, 6.0)) for m in ('fine', 'medium', 'coarse')]
    if all(sf):
        f1, f2, f3 = [x['D'] for x in sf]; rr = 0.48 / 0.36
        e32, e21 = f3 - f2, f2 - f1
        if e21 != 0 and e32 / e21 > 0:
            p = abs(math.log(abs(e32 / e21))) / math.log(rr)
            fext = f1 + (f1 - f2) / (rr ** p - 1)
            gci = dict(p=p, fext=fext, gci=1.25 * abs((f1 - f2) / f1) / (rr ** p - 1) * 100, f=(f1, f2, f3), mono=True)
        else:
            gci = dict(f=(f1, f2, f3), mono=False)

    # ---- assets
    A = {}
    for ax in AXES:
        mesh = 'medium' if os.path.exists(os.path.join(RES, 'figures', 'flow', f'flow3d_{ax}_medium_6kn.png')) else 'coarse'
        tag = f'{ax}_{mesh}_6kn'; A[ax] = dict(mesh=mesh)
        fl = os.path.join(RES, 'figures', 'flow')
        A[ax]['flow3d'] = img(os.path.join(fl, f'flow3d_{tag}.png'), f'flow3d_{ax}.webp')
        A[ax]['cp'] = img(os.path.join(fl, f'cp_{tag}.png'), f'cp_{ax}.webp')
        A[ax]['slices'] = [img(p, f'{ax}_' + os.path.basename(p).split('_')[1] + '.webp')
                           for p in sorted(glob.glob(os.path.join(fl, f'slice_*_{tag}.png')))]
        an = os.path.join(RES, 'animations')
        A[ax]['v3d'] = vid(os.path.join(an, f'anim3d_{tag}.mp4'), f'v3d_{ax}.mp4') if os.path.exists(os.path.join(an, f'anim3d_{tag}.mp4')) else None
        A[ax]['vcp'] = vid(os.path.join(an, f'anim_cp_{tag}.mp4'), f'vcp_{ax}.mp4') if os.path.exists(os.path.join(an, f'anim_cp_{tag}.mp4')) else None
        A[ax]['vsl'] = [vid(p, f'vsl_{ax}_' + os.path.basename(p).split('_')[2] + '.mp4')
                        for p in sorted(glob.glob(os.path.join(an, f'anim_slice_*_{tag}.mp4')))]
        A[ax]['yph'] = img(os.path.join(RES, 'figures', 'yplus', f'hist_{ax}_coarse.png'), f'yph_{ax}.webp', 1100)
        A[ax]['ypc'] = img(os.path.join(RES, 'figures', 'yplus', f'yplus_{ax}_coarse_6kn.png'), f'ypc_{ax}.webp')
    A['geo_orig'] = img(os.path.join(RES, 'figures', 'geometry_original.png'), 'geo_orig.webp')
    A['geo_z'] = img(os.path.join(RES, 'figures', 'envelope_sections_z.png'), 'geo_z.webp')
    A['geo_l'] = img(os.path.join(RES, 'figures', 'envelope_sections_long.png'), 'geo_l.webp')

    def video(v, cap, cls=''):
        if not v: return ''
        return (f'<figure class="fig {cls}"><video src="{v[0]}" poster="{v[1]}" autoplay muted loop playsinline controls preload="metadata"></video>'
                f'<figcaption>{cap}</figcaption></figure>')

    def image(src, cap, cls=''):
        return f'<figure class="fig {cls}"><img src="{src}" alt="{esc(cap)}" loading="lazy"><figcaption>{cap}</figcaption></figure>'

    # ---- tables HTML
    trows = []
    for r in table:
        k = float(r['speed_kn'])
        cells_ = ''.join(f'<td class="num">{fmt_n(float(r[f"{a}_drag_N"]), 1 if float(r[f"{a}_drag_N"]) < 100 else 0)}</td>'
                         f'<td class="num">{float(r[f"{a}_power_W"]) / 1000:,.{3 if float(r[f"{a}_power_W"]) < 1000 else 2 if float(r[f"{a}_power_W"]) < 1e4 else 1}f}</td>' for a in AXES)
        trows.append(f'<tr{" class=hl" if k in (1, 2, 4, 6) else ""}><td class="num">{k:g}</td><td class="num">{float(r["speed_mps"]):.3f}</td>{cells_}</tr>')
    grid_html = ''.join(
        f'<tr><td>{AXN[ax][0]}</td><td>{m}</td><td class="num">{fmt_n(c) if c else "–"}</td><td class="num">{fmt_n(D, 1)}</td>'
        f'<td class="num">±{fmt_n(sd, 1)}</td><td class="num">{Cd:.4f}</td>'
        f'<td class="num">{"–" if m == "coarse" else f"{(D / [g for g in grid_rows if g[0] == ax and g[1] == "coarse"][0][3] - 1) * 100:+.1f} %"}</td></tr>'
        for ax, m, c, D, sd, Cd in grid_rows)
    yp_html = ''.join(
        f'<tr><td>{AXN[r["axis"]][0]}</td><td>{r["mesh"]}</td><td class="num">{float(r["speed_kn"]):g}</td><td class="num">{r["yplus_area_avg"]}</td>'
        f'<td class="num">{r["yplus_p05"]}–{r["yplus_p95"]}</td><td class="num">{r["area_pct_5_30"]}</td><td class="num">{r["area_pct_30_300"]}</td>'
        f'<td class="num">{r["area_pct_gt300"]}</td></tr>' for r in sorted(yp, key=lambda r: (AXES.index(r['axis']), ['coarse', 'medium', 'fine'].index(r['mesh']), float(r['speed_kn']))))
    runs_html = ''.join(
        f'<tr><td>{AXN[r["axis"]][0]}</td><td>{r["mesh"]}</td><td class="num">{r["kn"]:g}</td><td class="num">{r["Re"]:.2e}</td>'
        f'<td class="num">{fmt_n(r["D"], 1)}</td><td class="num">{fmt_n(r["D_pressure"], 1)}</td><td class="num">{fmt_n(r["D_viscous"], 1)}</td>'
        f'<td class="num">{r["Cd"]:.4f}</td><td class="num">{r["niter"]}</td></tr>'
        for r in sorted(runs, key=lambda r: (AXES.index(r['axis']), ['coarse', 'medium', 'fine'].index(r['mesh']), r['kn'])))
    # ITTC friction check (surge, coarse)
    S = 27.5553
    ittc = []
    for r in sorted([r for r in runs if r['axis'] == 'surge' and r['mesh'] == 'coarse'], key=lambda r: r['kn']):
        Cf = 0.075 / (math.log10(r['Re']) - 2) ** 2; Df = 0.5 * rho * r['V'] ** 2 * S * Cf
        ittc.append(f'<tr><td class="num">{r["kn"]:g}</td><td class="num">{Cf:.5f}</td><td class="num">{Df:,.1f}</td>'
                    f'<td class="num">{r["D_viscous"]:,.1f}</td><td class="num">{(r["D_viscous"] / Df - 1) * 100:+.1f} %</td></tr>')
    for mm in ('medium', 'fine'):
        sm = R.get(('surge', mm, 6.0))
        if not sm: continue
        Cf = 0.075 / (math.log10(sm['Re']) - 2) ** 2; Df = 0.5 * rho * sm['V'] ** 2 * S * Cf
        ittc.append(f'<tr><td class="num">6 ({mm})</td><td class="num">{Cf:.5f}</td><td class="num">{Df:,.1f}</td>'
                    f'<td class="num">{sm["D_viscous"]:,.1f}</td><td class="num">{(sm["D_viscous"] / Df - 1) * 100:+.1f} %</td></tr>')

    gci_html = ''
    if gci and gci.get('mono'):
        gci_html = (f'<p>Surge drag at 6 kn on three systematically refined meshes (refinement ratio r = 1.33): '
                    f'{gci["f"][2]:.1f} N (coarse), {gci["f"][1]:.1f} N (medium), {gci["f"][0]:.1f} N (fine). '
                    f'Observed order of convergence p = {gci["p"]:.2f}; Richardson-extrapolated drag {gci["fext"]:.1f} N; '
                    f'fine-grid convergence index GCI = {gci["gci"]:.1f} % (Celik et al. 2008, ASME J. Fluids Eng. 130, 078001).</p>')
    elif gci:
        gci_html = (f'<p>Surge drag at 6 kn: {gci["f"][2]:.1f} N (coarse), {gci["f"][1]:.1f} N (medium), {gci["f"][0]:.1f} N (fine). '
                    f'The three values do not converge monotonically, so a Richardson extrapolation is not valid; the spread '
                    f'between meshes is used as the discretisation uncertainty.</p>')

    data_js = json.dumps(dict(curves=curves, rho=rho))
    tabs = []
    for ax in AXES:
        a = A[ax]
        sl_imgs = ''.join(image(s, f'{AXN[ax][0]} 6 kn: mid-plane slice. Top: speed |U|/V with streamlines. Bottom: pressure coefficient Cp.', 'wide') for s in a['slices'])
        sl_vids = ''.join(video(v, f'{AXN[ax][0]} 6 kn: particles carried by the in-plane velocity on a mid-plane.', 'wide') for v in a['vsl'])
        tabs.append(f'''
      <div class="tabpanel" id="tab-{ax}" role="tabpanel" aria-labelledby="tb-{ax}" {'hidden' if ax != 'surge' else ''}>
        <div class="grid2">
          {video(a['v3d'], f'{AXN[ax][0]} 6 kn: water particles along 3-D streamlines, slowed ×10, coloured by local speed U/V; body coloured by Cp.', 'wide')}
          {image(a['flow3d'], f'{AXN[ax][0]} 6 kn: 3-D streamlines coloured by U/V over the speed mid-plane; body coloured by pressure coefficient Cp.', 'wide')}
          {sl_vids}
          {sl_imgs}
          {video(a['vcp'], f'{AXN[ax][0]} 6 kn: surface pressure coefficient, rotating view.')}
          {image(a['cp'], f'{AXN[ax][0]} 6 kn: surface pressure coefficient Cp, four views.')}
        </div>
        <p class="note">Rendered from the {a['mesh']}-mesh solution. Particles follow the converged steady (time-averaged) velocity field; they show where water goes on average, not unsteady vortex shedding.</p>
      </div>''')

    tiles = ''.join(f'''
      <article class="tile" style="--c: var(--s-{ax})">
        <header><span class="eyebrow">{AXN[ax][0]} · motion {AXN[ax][2]}</span></header>
        <div class="big">{six[ax]['D'] / 1000:,.{2 if six[ax]['D'] < 1e4 else 1}f}<span class="u">kN</span></div>
        <div class="lbl">drag at 6 kn</div>
        <dl>
          <div><dt>Hydraulic power</dt><dd>{six[ax]['P'] / 1000:,.1f} kW</dd></div>
          <div><dt>Cd (A = {six[ax]['A']:.2f} m²)</dt><dd>{six[ax]['Cd']:.3f}</dd></div>
          <div><dt>Pressure / friction</dt><dd>{six[ax]['pf']}</dd></div>
        </dl>
      </article>''' for ax in AXES)

    today = datetime.date.today().strftime('%d %b %Y')
    tpl = open(os.path.join(ROOT, 'scripts', 'report_template.html')).read()
    page = (tpl.replace('{{TILES}}', tiles).replace('{{TROWS}}', ''.join(trows)).replace('{{TABS}}', ''.join(tabs))
            .replace('{{GRID}}', grid_html).replace('{{GCI}}', gci_html).replace('{{YP}}', yp_html).replace('{{RUNS}}', runs_html)
            .replace('{{ITTC}}', ''.join(ittc)).replace('{{DATA}}', data_js).replace('{{DATE}}', today)
            .replace('{{GEO_ORIG}}', A['geo_orig']).replace('{{GEO_Z}}', A['geo_z']).replace('{{GEO_L}}', A['geo_l'])
            .replace('{{CELLS_C}}', fmt_n(cells.get(('surge', 'coarse'), 0))).replace('{{CELLS_M}}', fmt_n(cells.get(('surge', 'medium'), 0)))
            .replace('{{CELLS_F}}', fmt_n(cells.get(('surge', 'fine'), 0)) if cells.get(('surge', 'fine')) else '–')
            .replace('{{YPH}}', ''.join(image(A[a]['yph'], f'{AXN[a][0]}: area-weighted y+ distribution, coarse mesh, 1–6 kn.') for a in AXES))
            .replace('{{YPC}}', ''.join(image(A[a]['ypc'], f'{AXN[a][0]} 6 kn: y+ on the body, coarse mesh.') for a in AXES))
            .replace('{{CORR}}', ', '.join(f'{AXN[a][0].lower()} {(fit[a]["grid_correction"] - 1) * 100:+.1f} %' for a in AXES)))
    for ax in AXES:
        page = page.replace('{{D6_%s}}' % ax, f'{six[ax]["D"]:,.0f}').replace('{{P6_%s}}' % ax, f'{six[ax]["P"] / 1000:,.1f}')
    open(os.path.join(OUT, 'index.html'), 'w').write(page)
    tot = sum(os.path.getsize(os.path.join(AS, f)) for f in os.listdir(AS))
    print('report written; assets', len(os.listdir(AS)), 'files', round(tot / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
