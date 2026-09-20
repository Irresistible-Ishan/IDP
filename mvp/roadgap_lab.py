"""
roadgap_lab.py - GPU-free post-processing lab for SAM-Road-style road-graph extraction.

What it does
------------
1. Builds a synthetic "road world": ground-truth roads, an occluded road-probability map
   (what SAM-Road's road mask looks like under trees/shadows) and faint non-road distractors.
2. Implements the post-processing families discussed in the research-gap report:
     A  baseline threshold
     B  global gamma on the mask                (Paper 1)
     C  plain threshold sweep                   (the control Paper 1 never ran)
     D  P-only "adaptive" gamma                 (the other AI's Gap 1 formula)
     E  distance-only gap bridging              (Paper 1's modified A*, emulated)
     F  global gamma + distance-only bridging   (Paper 1's hybrid, emulated)
     G  direction-aware + evidence-verified bridging   (proposed MVP)
3. Scores them with APLS-lite / buffer-F1 proxies. These are NOT the official TOPO/APLS
   (use the SAM-Road repo's Go APLS + Sat2Graph TOPO for real numbers).

Run:  python roadgap_lab.py --n_val 10 --n_test 30
"""
from __future__ import annotations

import argparse
import heapq
import math
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splev, splprep
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

H = W = 192
N8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
N8_LEN = np.array([math.hypot(*d) for d in N8])
N8_UNIT = np.array(N8, float) / N8_LEN[:, None]
COS_TURN = (N8_UNIT @ N8_UNIT.T).tolist()


# --------------------------------------------------------------------------- synthetic world
def _spline(rng, pts, n=300):
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts) - 1))
    xs, ys = splev(np.linspace(0, 1, n), tck)
    return np.column_stack([xs, ys]).astype(np.float32)


def _draw(canvas, xy, thickness, value=1):
    cv2.polylines(canvas, [np.round(xy).astype(np.int32).reshape(-1, 1, 2)], False, value, thickness)


def make_world(seed):
    """Returns dict(P=prob map, gt_sk=GT skeleton, occ=occlusion mask)."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((H, W), np.uint8)
    for horizontal in (True, False):
        for base in (40, 96, 150):
            along = np.linspace(0, W - 1, 5)
            across = base + rng.uniform(-8, 8) + rng.normal(0, 10, 5)
            pts = np.column_stack([along, across]) if horizontal else np.column_stack([across, along])
            _draw(gt, _spline(rng, pts), 3)
    gt_sk = skeletonize(gt > 0)

    # occlusions centred on roads (trees / shadows): gap length ~ 2r = 12..26 px
    ys, xs = np.nonzero(gt_sk)
    occ = np.zeros((H, W), np.uint8)
    for _ in range(7):
        i = rng.integers(len(ys))
        cv2.circle(occ, (int(xs[i]), int(ys[i])), int(rng.integers(6, 14)), 1, -1)

    road = cv2.dilate(gt, np.ones((3, 3), np.uint8))
    P = 0.03 + 0.90 * ndi.gaussian_filter(road.astype(float), 0.8)

    # faint non-road linear structures (trails, field edges): the "spurious shortcut" bait
    for _ in range(4):
        c = rng.uniform(20, H - 20, 2)
        ang = rng.uniform(0, 2 * math.pi)
        ln = rng.uniform(40, 80)
        pts = np.array([c + ln * 0.5 * np.array([math.cos(ang), math.sin(ang)]) * s + rng.normal(0, 4, 2)
                        for s in (-1, 0, 1)])
        cv = np.zeros((H, W), np.uint8)
        _draw(cv, _spline(rng, np.clip(pts, 2, H - 3), 120), 3)
        P = np.maximum(P, rng.uniform(0.28, 0.45) * ndi.gaussian_filter(cv.astype(float), 0.8))

    soft = ndi.gaussian_filter(occ.astype(float), 2.0)
    P = P * (1 - soft * rng.uniform(0.55, 0.85))
    noise = ndi.gaussian_filter(rng.normal(size=(H, W)), 2.5)
    P = np.clip(P + 0.04 * noise / noise.std(), 0, 1)
    return dict(P=P, gt_sk=gt_sk, occ=occ.astype(bool))


# --------------------------------------------------------------------------- mask -> skeleton
def extract_skeleton(mask_bool, min_comp=12):
    sk = skeletonize(mask_bool)
    lab, n = ndi.label(sk, structure=np.ones((3, 3)))
    if n == 0:
        return sk
    sizes = ndi.sum(sk, lab, range(1, n + 1))
    return sk & np.isin(lab, 1 + np.nonzero(sizes >= min_comp)[0])


# --------------------------------------------------------------------------- tone-curve variants
def gamma_mask(P, g):                     # Paper 1: x^(1/g) on the predicted mask
    return P ** (1.0 / g)


def p_only_adaptive(P, alpha):            # other AI's Gap 1: gamma(x,y)=1+alpha*(1-P)
    return P ** (1.0 / (1.0 + alpha * (1.0 - P)))


# --------------------------------------------------------------------------- metrics (proxies)
def pixel_graph(sk):
    ys, xs = np.nonzero(sk)
    idx = -np.ones(sk.shape, int)
    idx[ys, xs] = np.arange(len(ys))
    r, c, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        j = np.full(len(ys), -1)
        j[ok] = idx[y2[ok], x2[ok]]
        sel = j >= 0
        r.append(np.nonzero(sel)[0]); c.append(j[sel]); w.append(np.full(sel.sum(), math.hypot(dy, dx)))
    r, c, w = np.concatenate(r), np.concatenate(c), np.concatenate(w)
    n = len(ys)
    A = coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    return np.column_stack([ys, xs]).astype(float), (A + A.T).tocsr()


def _apls_dir(src, dst, seed, n_src=18, n_tgt=4, buf=4.0, min_sep=25.0):
    (sp, sA), (dp, dA) = src, dst
    if len(sp) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    _, lab = connected_components(sA, directed=False)
    srcs = rng.choice(len(sp), size=min(n_src, len(sp)), replace=False)
    Dsrc = dijkstra(sA, indices=srcs)
    kd = cKDTree(dp) if len(dp) else None
    map_d, map_i = kd.query(sp) if kd is not None else (np.full(len(sp), np.inf), np.zeros(len(sp), int))
    uniq = np.unique(map_i[srcs])
    Ddst = dijkstra(dA, indices=uniq) if len(dp) else None
    row = {u: k for k, u in enumerate(uniq)}
    scores = []
    for k, a in enumerate(srcs):
        cand = np.nonzero((lab == lab[a]) & (np.linalg.norm(sp - sp[a], axis=1) >= min_sep))[0]
        if len(cand) == 0:
            continue
        for b in rng.choice(cand, size=min(n_tgt, len(cand)), replace=False):
            L = Dsrc[k, b]
            if not np.isfinite(L) or L <= 0:
                continue
            if map_d[a] > buf or map_d[b] > buf or Ddst is None:
                scores.append(0.0); continue
            Ld = Ddst[row[map_i[a]], map_i[b]]
            scores.append(0.0 if not np.isfinite(Ld) else 1.0 - min(1.0, abs(L - Ld) / L))
    return float(np.mean(scores)) if scores else 0.0


def evaluate(pred_sk, w, seed):
    gt_sk, occ = w["gt_sk"], w["occ"]
    if not pred_sk.any():
        return dict(apls=0.0, prec=0.0, rec=0.0, f1=0.0, gap_rec=0.0)
    d_gt = ndi.distance_transform_edt(~gt_sk)
    d_pr = ndi.distance_transform_edt(~pred_sk)
    prec = float((d_gt[pred_sk] <= 3).mean())
    rec = float((d_pr[gt_sk] <= 3).mean())
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    g, p = pixel_graph(gt_sk), pixel_graph(pred_sk)
    a1, a2 = _apls_dir(g, p, seed), _apls_dir(p, g, seed + 7)
    apls = 2 * a1 * a2 / (a1 + a2 + 1e-9)
    m = gt_sk & occ
    gap_rec = float((d_pr[m] <= 3).mean()) if m.any() else 1.0
    return dict(apls=apls, prec=prec, rec=rec, f1=f1, gap_rec=gap_rec)


# --------------------------------------------------------------------------- gap bridging
def _tangent(sk, p, steps=10):
    cur, path = tuple(p), [tuple(p)]
    for _ in range(steps):
        nxt = None
        for dy, dx in N8:
            q = (cur[0] + dy, cur[1] + dx)
            if 0 <= q[0] < H and 0 <= q[1] < W and sk[q] and q not in path:
                nxt = q; break
        if nxt is None:
            break
        cur = nxt; path.append(cur)
    v = np.array(p, float) - np.array(cur, float)
    n = np.linalg.norm(v)
    return None if n < 4 else v / n


def find_bridges(sk, Pm, R=30, lam=0.0, w_ev=3.0):
    """One search per dead-end. lam=0 -> distance-only (Paper-1-like); lam>0 -> direction-aware.
    Cost per step = len * (1 + w_ev*(1-P)) * (1 + lam*(1-cos(turn))).  Search state = (y, x, heading)."""
    lab, ncomp = ndi.label(sk, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=ncomp + 1)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    skl = sk.tolist()                                     # plain lists: ~4x faster scalar access
    pcost = (1 + w_ev * (1 - Pm)).tolist()
    pts, A = pixel_graph(sk)
    pidx = -np.ones(sk.shape, int)
    pidx[pts[:, 0].astype(int), pts[:, 1].astype(int)] = np.arange(len(pts))
    min_gd = 2 * R                                        # Paper 1: MIN_GRAPH_DISTANCE = 2 x MAX_STRAIGHT_DISTANCE
    bridges = []
    for ey, ex in np.argwhere(sk & (nb == 1)):
        if sizes[lab[ey, ex]] < 30:                       # ignore stubs / spurs
            continue
        t = _tangent(sk, (ey, ex))
        if t is None:
            continue
        near = np.isfinite(dijkstra(A, indices=int(pidx[ey, ex]), limit=min_gd))
        closem = np.zeros(sk.shape, bool)
        closem[pts[near, 0].astype(int), pts[near, 1].astype(int)] = True
        closel = closem.tolist()                          # skeleton pixels graph-close to this dead end
        oy, ox = ey - 8 * t[0], ex - 8 * t[1]              # "opp" reference point behind the dead end
        d0 = int(np.argmax(N8_UNIT @ t))
        y0, y1, x0, x1 = max(0, ey - R), min(H, ey + R + 1), max(0, ex - R), min(W, ex + R + 1)
        max_cost = 2.5 * R * (1 + w_ev)                       # give up on hopeless searches early
        start = (int(ey), int(ex), d0)
        dist, parent, heap = {start: 0.0}, {}, [(0.0, start)]
        hit = None
        while heap:
            c, s = heapq.heappop(heap)
            if c > dist.get(s, 1e18):
                continue
            y, x, d = s
            if (y, x) != (ey, ex) and skl[y][x] and not closel[y][x]:
                hit = s; break
            if c > max_cost:
                break
            for nd in range(8):
                cos_turn = COS_TURN[d][nd]
                if lam > 0 and cos_turn < -1e-9:            # no >90 degree turns when direction prior is on
                    continue
                ny, nx = y + N8[nd][0], x + N8[nd][1]
                if not (y0 <= ny < y1 and x0 <= nx < x1):
                    continue
                if (ny - ey) ** 2 + (nx - ex) ** 2 >= (ny - oy) ** 2 + (nx - ox) ** 2:   # half-plane (Paper 1)
                    continue
                if skl[ny][nx] and closel[ny][nx]:
                    continue
                nc = c + N8_LEN[nd] * pcost[ny][nx] * (1 + lam * (1 - cos_turn))
                ns = (ny, nx, nd)
                if nc < dist.get(ns, 1e18):
                    dist[ns] = nc; parent[ns] = s
                    heapq.heappush(heap, (nc, ns))
        if hit is None:
            continue
        path, s = [], hit
        while s in parent:
            path.append((s[0], s[1])); s = parent[s]
        path.append((ey, ex)); path.reverse()
        inner = path[1:-1] or path
        vals = np.array([Pm[y, x] for y, x in inner])
        bridges.append(dict(path=path, mean_p=float(vals.mean()), min_p=float(vals.min()), length=len(path)))
    return bridges


def apply_bridges(sk, bridges, p_min=0.0):
    out = sk.copy()
    for b in bridges:
        if b["mean_p"] >= p_min:
            for y, x in b["path"]:
                out[y, x] = True
    return out


# --------------------------------------------------------------------------- experiment driver
def _mean(rs):
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


def _score(m):
    return 0.5 * (m["apls"] + m["f1"])


def _boot(diffs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(n_val, n_test, R):
    t0 = time.time()
    val = [make_world(s) for s in range(n_val)]
    test = [make_world(1000 + s) for s in range(n_test)]
    for i, w in enumerate(val + test):
        w["seed"] = i
        w["Pm"] = ndi.gaussian_filter(w["P"], 1.0)

    def evaluate_set(ws, fn):
        return [evaluate(fn(w), w, w["seed"]) for w in ws]

    # --- sanity checks on the two mathematical claims ------------------------------------------
    P = val[0]["P"]
    for g in (1.25, 1.5, 2.0):
        mism = np.mean((gamma_mask(P, g) > 0.5) != (P > 0.5 ** g))
        print(f"[check] gamma={g}: mask(P^(1/g)>0.5) vs mask(P>0.5^g) mismatch = {mism:.6f}")
    xs = np.linspace(0, 1, 100001)
    f = p_only_adaptive(xs, 1.0)
    print(f"[check] P-only 'adaptive' gamma (alpha=1) is monotone in P: {bool(np.all(np.diff(f) >= -1e-12))}; "
          f"=> equivalent to threshold P>{xs[np.searchsorted(f, 0.5)]:.4f}")

    # --- pixel-level families (cheap) ----------------------------------------------------------
    fam = {}
    fam["A baseline tau=0.5"] = lambda w: extract_skeleton(w["P"] > 0.5)
    for g in (1.25, 1.5, 1.75, 2.0, 2.2):
        fam[f"B gamma={g}"] = (lambda g: lambda w: extract_skeleton(gamma_mask(w["P"], g) > 0.5))(g)
    for tau in (0.45, 0.4, 0.35, 0.3, 0.25):
        fam[f"C tau={tau}"] = (lambda t: lambda w: extract_skeleton(w["P"] > t))(tau)
    for a in (0.5, 1.0, 2.0):
        fam[f"D P-only-gamma alpha={a}"] = (lambda a: lambda w: extract_skeleton(p_only_adaptive(w["P"], a) > 0.5))(a)

    val_res = {k: _mean(evaluate_set(val, fn)) for k, fn in fam.items()}

    def best(prefix):
        ks = [k for k in fam if k.startswith(prefix)]
        return max(ks, key=lambda k: _score(val_res[k]))

    chosen = {"A": "A baseline tau=0.5", "B": best("B"), "C": best("C"), "D": best("D")}

    # --- bridging families (expensive): search once per (mask, lam), filter p_min for free -----
    gam = float(chosen["B"].split("=")[1])
    print(f"[info] val-selected: B {chosen['B']} | C {chosen['C']} | D {chosen['D']}")

    def searches(ws, mask_fn, lam):
        cache = []
        for w in ws:
            sk = mask_fn(w)
            cache.append((sk, find_bridges(sk, w["Pm"], R=R, lam=lam)))
        return cache

    base_mask = lambda w: extract_skeleton(w["P"] > 0.5)
    gam_mask_fn = lambda w: extract_skeleton(gamma_mask(w["P"], gam) > 0.5)
    print("[info] running bridge searches (this is the slow part)...")
    S = {}
    for name, ws in (("val", val), ("test", test)):
        S[(name, "base", 0.0)] = searches(ws, base_mask, 0.0)
        S[(name, "gam", 0.0)] = searches(ws, gam_mask_fn, 0.0)
        for lam in (2.0, 4.0):
            S[(name, "base", lam)] = searches(ws, base_mask, lam)
        print(f"   {name} done ({time.time() - t0:.0f}s)")

    def eval_bridge(split, ws, mask_key, lam, p_min):
        return [evaluate(apply_bridges(sk, br, p_min), w, w["seed"])
                for w, (sk, br) in zip(ws, S[(split, mask_key, lam)])]

    p_grid = (0.0, 0.15, 0.2, 0.25, 0.3, 0.35)
    cfgs = {}
    cfgs["E distance-only bridging (Paper-1 style)"] = ("base", 0.0, 0.0)
    cfgs[f"F gamma={gam} + distance-only bridging (Paper-1 hybrid)"] = ("gam", 0.0, 0.0)
    # 2x2 ablation on baseline mask: direction {off,on} x evidence {off,on}; on-values chosen on VAL
    def best_cfg(lams, use_p):
        opts = [(lam, p) for lam in lams for p in (p_grid[1:] if use_p else (0.0,))]
        return max(opts, key=lambda o: _score(_mean(eval_bridge("val", val, "base", o[0], o[1]))))
    lam_e, p_e = best_cfg((0.0,), True)
    lam_d, _ = best_cfg((2.0, 4.0), False)
    lam_g, p_g = best_cfg((2.0, 4.0), True)
    cfgs[f"G0 evidence-only (p_min={p_e})"] = ("base", lam_e, p_e)
    cfgs[f"G1 direction-only (lam={lam_d})"] = ("base", lam_d, 0.0)
    cfgs[f"G2 direction+evidence (lam={lam_g}, p_min={p_g})  <-- proposed"] = ("base", lam_g, p_g)

    # --- test-set report ------------------------------------------------------------------------
    rows = {}
    for tag in ("A", "B", "C", "D"):
        rows[f"{tag} {chosen[tag][2:]}"] = evaluate_set(test, fam[chosen[tag]])
    for name, (mk, lam, p) in cfgs.items():
        rows[name] = eval_bridge("test", test, mk, lam, p)

    base_key = "A " + chosen["A"][2:]
    base = rows[base_key]
    print("\nTEST SET (synthetic, %d worlds; params chosen on %d VAL worlds)" % (n_test, n_val))
    hdr = f"{'method':66s} APLS*  Buf-P  Buf-R  Buf-F1 GapRec | dAPLS [95% CI]        dF1 [95% CI]"
    print(hdr); print("-" * len(hdr))
    for name, rs in rows.items():
        m = _mean(rs)
        da = _boot([r["apls"] - b["apls"] for r, b in zip(rs, base)])
        df = _boot([r["f1"] - b["f1"] for r, b in zip(rs, base)])
        print(f"{name[:66]:66s} {m['apls']:.3f}  {m['prec']:.3f}  {m['rec']:.3f}  {m['f1']:.3f}  {m['gap_rec']:.3f} |"
              f" {da[0]:+.3f} [{da[1]:+.3f},{da[2]:+.3f}]  {df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}]")
    print("\n* APLS-lite / Buf-F1 are proxies on synthetic data, NOT the official TOPO/APLS.")
    print(f"total time {time.time() - t0:.0f}s")
    return dict(test=test, S=S, cfgs=cfgs, chosen=chosen, gam=gam)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_val", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--R", type=int, default=30, help="max straight distance / search window radius (px = m)")
    a = ap.parse_args()
    run(a.n_val, a.n_test, a.R)
