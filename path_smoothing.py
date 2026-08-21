"""
Path smoothing / simplification for VTracer's SVG output.

VTracer's raw output (especially in "polygon" mode, or wherever the tracer
left many nearly-collinear points along a curved edge) tends to have far more
anchor points than the shape actually needs, and any straight-line runs are
literally straight lines rather than smooth curves.

This module re-processes each <path> element's `d` string in three steps:
  1. Collect each subpath's on-curve anchor points (the M point, plus the
     endpoint of every L/C segment -- NOT a dense resample of existing
     curves, so a path VTracer already fit efficiently in "spline" mode
     isn't blown back up into more points than it started with).
  2. Simplify that anchor list with Ramer-Douglas-Peucker (closed subpaths
     are split at their two most distant points first, so the loop isn't
     collapsed against a degenerate zero-length "chord").
  3. Re-fit smooth cubic Bezier curves (Catmull-Rom -> Bezier) through the
     surviving anchor points -- but detect sharp corners first and break the
     curve there, so real corners stay crisp straight joins instead of being
     rounded off, while genuinely curved runs become one smooth C1 curve
     using as few anchors as the shape allows.
"""

import re
import numpy as np

_CMD_TOKEN_RE = re.compile(r'[MLCZmlcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?')


def parse_path_to_subpaths(d):
    """Parse an SVG path `d` string (absolute M/L/C/Z commands, as emitted by
    VTracer) into a list of {"points": Nx2 ndarray, "closed": bool} subpaths,
    keeping only the on-curve anchor points (curve control points are
    dropped -- they get re-derived from scratch during re-fitting)."""
    tokens = _CMD_TOKEN_RE.findall(d)
    subpaths = []
    pts = []
    closed = False
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    cmd = None
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t in "MLCZmlcz":
            cmd = t
            i += 1
            if cmd in "Zz":
                closed = True
                if pts and pts[-1] != start:
                    pts.append(start)
                continue
        if cmd is None:
            i += 1
            continue
        if cmd in "Mm":
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if len(pts) > 1:
                subpaths.append({"points": np.array(pts, dtype=float), "closed": closed})
            pts = [(x, y)]
            closed = False
            cur = (x, y)
            start = (x, y)
            cmd = "L"  # any further bare coordinate pairs are implicit linetos
        elif cmd in "Ll":
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            pts.append((x, y))
            cur = (x, y)
        elif cmd in "Cc":
            x1, y1, x2, y2, x, y = (float(tokens[i + k]) for k in range(6))
            i += 6
            pts.append((x, y))  # keep the on-curve endpoint only, drop control points
            cur = (x, y)
        else:
            i += 1
    if len(pts) > 1:
        subpaths.append({"points": np.array(pts, dtype=float), "closed": closed})
    return subpaths


def _rdp(points, epsilon):
    """Ramer-Douglas-Peucker simplification of an open polyline."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = float(np.hypot(*line_vec))
    if line_len < 1e-9:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        normal = np.array([-line_vec[1], line_vec[0]]) / line_len
        dists = np.abs((points - start) @ normal)
    idx = int(np.argmax(dists))
    dmax = dists[idx]
    if dmax > epsilon:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def _simplify_polyline(points, closed, epsilon):
    if len(points) < 4:
        return points
    if not closed:
        return _rdp(points, epsilon)

    # A closed loop can't be RDP'd directly against its own zero-length
    # start==end chord, so split it at its two most distant points into two
    # open chains, simplify each, then re-close.
    pts = points[:-1] if np.allclose(points[0], points[-1]) else points
    m = len(pts)
    if m < 4:
        return points
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    if i > j:
        i, j = j, i
    chain1 = pts[i: j + 1]
    chain2 = np.vstack([pts[j:], pts[: i + 1]])
    s1 = _rdp(chain1, epsilon)
    s2 = _rdp(chain2, epsilon)
    merged = np.vstack([s1, s2[1:-1]])
    return np.vstack([merged, merged[0]])


def _format_num(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt(p):
    return f"{_format_num(p[0])},{_format_num(p[1])}"


def _turn_angle(v1, v2):
    """Degrees of deviation from straight-ahead (0 = collinear, 180 = full
    reversal) between two direction vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _open_chain_commands(chain):
    """Smooth Catmull-Rom -> Bezier commands (no leading M) that travel from
    chain[0] to chain[-1] through every interior point, using phantom
    endpoints mirrored from inside this chain only -- so tangents never
    bleed across a corner into a neighboring chain."""
    m = len(chain)
    if m == 2:
        return [f"L {_fmt(chain[1])}"]
    p_start = 2 * chain[0] - chain[1]
    p_end = 2 * chain[-1] - chain[-2]
    ext = np.vstack([p_start, chain, p_end])
    cmds = []
    for k in range(m - 1):
        p0, p1, p2, p3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        cmds.append(f"C {_fmt(c1)} {_fmt(c2)} {_fmt(p2)}")
    return cmds


def _catmull_rom_cyclic_d(pts):
    """Fully smooth closed curve through every point, wrapping around (used
    only when no corners were detected anywhere on the loop)."""
    n = len(pts)
    parts = [f"M {_fmt(pts[0])}"]
    for k in range(n):
        p0, p1, p2, p3 = pts[(k - 1) % n], pts[k % n], pts[(k + 1) % n], pts[(k + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        parts.append(f"C {_fmt(c1)} {_fmt(c2)} {_fmt(p2)}")
    parts.append("Z")
    return " ".join(parts)


def _chains_to_d(chains, closed):
    d = f"M {_fmt(chains[0][0])}"
    for chain in chains:
        d += " " + " ".join(_open_chain_commands(chain))
    if closed:
        d += " Z"
    return d


def _fit_smooth_d(points, closed, corner_angle_deg=30.0):
    pts = np.array(points, dtype=float)
    if closed and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)

    if n < 3:
        d = f"M {_fmt(pts[0])}"
        for p in pts[1:]:
            d += f" L {_fmt(p)}"
        if closed:
            d += " Z"
        return d

    if closed:
        angles = np.array([
            _turn_angle(pts[i] - pts[(i - 1) % n], pts[(i + 1) % n] - pts[i])
            for i in range(n)
        ])
        corner_idx = [i for i in range(n) if angles[i] > corner_angle_deg]
        if not corner_idx:
            return _catmull_rom_cyclic_d(pts)
        corner_idx = sorted(corner_idx)
        chains = []
        for a, b in zip(corner_idx, corner_idx[1:] + [corner_idx[0] + n]):
            idxs = [k % n for k in range(a, b + 1)]
            chains.append(pts[idxs])
        return _chains_to_d(chains, closed=True)

    # open subpath: interior corners split it into independently-smoothed chains
    interior = [0]
    for i in range(1, n - 1):
        angle = _turn_angle(pts[i] - pts[i - 1], pts[i + 1] - pts[i])
        if angle > corner_angle_deg:
            interior.append(i)
    interior.append(n - 1)
    corner_idx = sorted(set(interior))
    chains = [pts[a: b + 1] for a, b in zip(corner_idx, corner_idx[1:])]
    return _chains_to_d(chains, closed=False)


_SUBPATH_RE = re.compile(r'[Mm][^Mm]*')
_HAS_CURVE_RE = re.compile(r'[Cc]')


def smooth_path_d(d, tolerance=0.8, corner_angle_deg=30.0):
    """Re-fit a path `d` string with fewer anchor points and smooth cubic
    Bezier curves through genuinely curved, line-only runs, while keeping
    real corners sharp.

    Subpaths VTracer already drew with cubic Beziers (its "spline" mode) are
    left untouched -- its own least-squares curve fit already places control
    points optimally, and re-deriving them from a resampled polyline can only
    lose accuracy or add anchors back. The real win is on line-only subpaths
    (e.g. "polygon" mode, or any traced region made of many tiny straight
    segments): those get simplified and re-fit as smooth curves here.
    """
    chunks = _SUBPATH_RE.findall(d)
    if not chunks:
        return d
    out = []
    for chunk in chunks:
        if _HAS_CURVE_RE.search(chunk):
            out.append(chunk.strip())
            continue
        subpaths = parse_path_to_subpaths(chunk)
        if not subpaths:
            out.append(chunk.strip())
            continue
        for sp in subpaths:
            simplified = _simplify_polyline(sp["points"], sp["closed"], tolerance)
            out.append(_fit_smooth_d(simplified, sp["closed"], corner_angle_deg=corner_angle_deg))
    return " ".join(out)


_PATH_D_RE = re.compile(r'(<path\b[^>]*?\bd=")([^"]+)(")', re.IGNORECASE)


def smooth_svg_paths(svg_str, tolerance=0.8, corner_angle_deg=30.0):
    """Replace every <path d="..."> in an SVG document with a version that
    uses fewer anchor points and smooth cubic Bezier curves through curved
    runs, while keeping real corners sharp."""

    def _replace(match):
        prefix, d, suffix = match.group(1), match.group(2), match.group(3)
        try:
            new_d = smooth_path_d(d, tolerance=tolerance, corner_angle_deg=corner_angle_deg)
        except Exception:
            return match.group(0)  # leave this path untouched if anything goes wrong
        return prefix + new_d + suffix

    return _PATH_D_RE.sub(_replace, svg_str)
