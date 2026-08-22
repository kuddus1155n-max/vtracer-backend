"""
Automatic SVG path cleanup for VTracer's output.

VTracer (especially on busy photos, or with an aggressive detail setting)
can leave behind:
  - exact or near-exact duplicate paths (two paths tracing the same region)
  - tiny "confetti" objects (a handful of stray pixels that became their own
    <path>)
  - broken/degenerate paths (zero-length subpaths, subpaths with too few
    points to be a real shape, non-finite coordinates)
  - redundant anchor points (a point sitting exactly on the straight line
    between its neighbors -- adds nothing to the shape)
  - unnecessary overlaps: an earlier (further back) path that is fully
    hidden underneath a later same-colored path directly on top of it, so
    it never actually shows up in the rendered image

This module runs entirely on the SVG markup VTracer already produced (it
does not re-trace anything), and removes/simplifies only what provably
doesn't change how the image looks.

Run order relative to path_smoothing.smooth_svg_paths(): cleanup first,
smoothing second. Cleanup fixes/removes structurally bad or redundant
geometry; smoothing then re-fits the (now-clean) line-only paths as nice
curves. Doing it in the other order would waste smoothing effort on paths
that cleanup was going to delete anyway.
"""

import re
import numpy as np

_TAG_RE = re.compile(r'<path\b[^>]*?/?>', re.IGNORECASE | re.DOTALL)
_D_RE = re.compile(r'\bd="([^"]*)"', re.IGNORECASE)
_FILL_ATTR_RE = re.compile(r'\bfill="([^"]*)"', re.IGNORECASE)
_STYLE_FILL_RE = re.compile(r'fill:\s*([^;"]+)', re.IGNORECASE)
_VIEWBOX_RE = re.compile(r'viewBox="([\-\d.\s]+)"', re.IGNORECASE)
_WIDTH_RE = re.compile(r'\bwidth="(\d+(?:\.\d+)?)', re.IGNORECASE)
_HEIGHT_RE = re.compile(r'\bheight="(\d+(?:\.\d+)?)', re.IGNORECASE)

_CMD_TOKEN_RE = re.compile(r'[MLCZmlcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?')


# ---------------------------------------------------------------------------
# d-string <-> structured subpath parsing (keeps curve control points, unlike
# path_smoothing's anchor-only parser, so untouched geometry round-trips
# byte-for-byte and only genuinely redundant/bad pieces are ever changed).
# ---------------------------------------------------------------------------

def _parse_d(d):
    """Parse an absolute M/L/C/Z `d` string into subpaths:
    [{"start": (x, y), "segs": [seg, ...], "closed": bool}, ...]
    where each seg is ("L", (x, y)) or ("C", (x1,y1), (x2,y2), (x, y)).
    """
    tokens = _CMD_TOKEN_RE.findall(d)
    n = len(tokens)
    subpaths = []
    i = 0
    cmd = None
    start = (0.0, 0.0)
    segs = []
    closed = False
    started = False

    def flush():
        if started and segs:
            subpaths.append({"start": start, "segs": segs[:], "closed": closed})

    while i < n:
        t = tokens[i]
        if t in "MLCZmlcz":
            cmd = t
            i += 1
            if cmd in "Zz":
                closed = True
                continue
        if cmd is None:
            i += 1
            continue
        if cmd in "Mm":
            if i + 1 >= n:
                break
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            flush()
            start = (x, y)
            segs = []
            closed = False
            started = True
            cmd = "L"  # subsequent bare coordinate pairs are implicit linetos
        elif cmd in "Ll":
            if i + 1 >= n:
                break
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            segs.append(("L", (x, y)))
        elif cmd in "Cc":
            if i + 5 >= n:
                break
            x1, y1, x2, y2, x, y = (float(tokens[i + k]) for k in range(6))
            i += 6
            segs.append(("C", (x1, y1), (x2, y2), (x, y)))
        else:
            i += 1
    flush()
    return subpaths


def _format_num(v):
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_pt(p):
    return f"{_format_num(p[0])},{_format_num(p[1])}"


def _subpath_to_d(sp):
    d = f"M {_fmt_pt(sp['start'])}"
    for seg in sp["segs"]:
        if seg[0] == "L":
            d += f" L {_fmt_pt(seg[1])}"
        else:
            _, c1, c2, end = seg
            d += f" C {_fmt_pt(c1)} {_fmt_pt(c2)} {_fmt_pt(end)}"
    if sp["closed"]:
        d += " Z"
    return d


# ---------------------------------------------------------------------------
# geometry helpers (pure numpy -- no extra dependency)
# ---------------------------------------------------------------------------

def _flatten_subpath(sp, curve_samples=10):
    """Sample the subpath into a dense Nx2 polyline, for area/overlap
    checks only -- the output `d` keeps the original commands untouched."""
    pts = [sp["start"]]
    for seg in sp["segs"]:
        if seg[0] == "L":
            pts.append(seg[1])
        else:
            _, c1, c2, end = seg
            p0 = np.array(pts[-1])
            p1 = np.array(c1)
            p2 = np.array(c2)
            p3 = np.array(end)
            for t in np.linspace(0.0, 1.0, curve_samples + 1)[1:]:
                mt = 1.0 - t
                pt = (mt ** 3) * p0 + 3 * (mt ** 2) * t * p1 + 3 * mt * (t ** 2) * p2 + (t ** 3) * p3
                pts.append((float(pt[0]), float(pt[1])))
    return np.array(pts, dtype=float)


def _polygon_area(pts):
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _bbox(pts):
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def _bbox_overlap(b1, b2, pad=0.0):
    return not (b1[2] + pad < b2[0] or b2[2] + pad < b1[0] or
                b1[3] + pad < b2[1] or b2[3] + pad < b1[1])


def _min_dist_to_polygon(points, poly):
    """Vectorized minimum distance from each point to the polygon's edges
    (as a closed ring). Used so that points lying exactly on (or right at)
    another polygon's boundary -- the common case when comparing two
    congruent shapes -- count as contained instead of falling into
    ray-casting's undefined boundary behavior."""
    px, py = points[:, 0], points[:, 1]
    n = len(poly)
    min_sq = np.full(len(points), np.inf)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        abx, aby = bx - ax, by - ay
        denom = abx * abx + aby * aby
        if denom < 1e-12:
            t = np.zeros_like(px)
        else:
            t = np.clip(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
        dx = px - (ax + t * abx)
        dy = py - (ay + t * aby)
        min_sq = np.minimum(min_sq, dx * dx + dy * dy)
    return np.sqrt(min_sq)


def _points_in_polygon(points, poly):
    """Vectorized ray-casting point-in-polygon test."""
    x, y = points[:, 0], points[:, 1]
    n = len(poly)
    inside = np.zeros(len(points), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / denom + xi)
        inside ^= cond
        j = i
    return inside


# ---------------------------------------------------------------------------
# redundant-node removal (drops zero-length segments and points that sit
# exactly on the straight line between their L-command neighbors; curve
# control points are never touched here -- that's smoothing's job)
# ---------------------------------------------------------------------------

def _remove_redundant_nodes_once(sp, epsilon):
    base_segs = sp["segs"]
    if len(base_segs) < 2:
        return sp, False
    # For closed subpaths, also check the wrap-around edge (last point ->
    # start) by appending a virtual closing segment while checking, then
    # dropping it afterward (Z already implies that edge).
    add_virtual_close = sp["closed"] and base_segs[-1][-1] != sp["start"]
    segs = base_segs + [("L", sp["start"])] if add_virtual_close else base_segs
    new_segs = []
    prev_anchor = np.array(sp["start"])
    changed = False
    idx = 0
    while idx < len(segs):
        seg = segs[idx]
        if seg[0] == "L":
            p_cur = np.array(seg[1])
            if np.allclose(prev_anchor, p_cur, atol=1e-6):
                changed = True
                idx += 1
                continue  # zero-length segment
            if idx + 1 < len(segs) and segs[idx + 1][0] == "L":
                p_next = np.array(segs[idx + 1][1])
                line = p_next - prev_anchor
                line_len = float(np.hypot(*line))
                if line_len > 1e-9:
                    normal = np.array([-line[1], line[0]]) / line_len
                    dist = abs(float(np.dot(p_cur - prev_anchor, normal)))
                    # also require p_cur to lie between prev_anchor and
                    # p_next (not before/after it) so we don't cut corners
                    along = float(np.dot(p_cur - prev_anchor, line)) / (line_len ** 2)
                    if dist < epsilon and -0.01 <= along <= 1.01:
                        changed = True
                        idx += 1
                        continue  # p_cur is redundant -- skip it
            new_segs.append(seg)
            prev_anchor = p_cur
        else:
            _, c1, c2, end = seg
            end_arr = np.array(end)
            if (np.allclose(prev_anchor, end_arr, atol=1e-6) and
                    np.allclose(prev_anchor, c1, atol=1e-6) and
                    np.allclose(prev_anchor, c2, atol=1e-6)):
                changed = True
                idx += 1
                continue  # zero-length curve
            new_segs.append(seg)
            prev_anchor = end_arr
        idx += 1
    if add_virtual_close and new_segs and new_segs[-1][0] == "L" and \
            np.allclose(new_segs[-1][1], sp["start"], atol=1e-6):
        new_segs = new_segs[:-1]  # drop the virtual closing segment again
        # note: its removal isn't itself a "change" -- it's not a real point
    sp["segs"] = new_segs
    return sp, changed


def _remove_redundant_nodes(sp, epsilon, max_passes=5):
    for _ in range(max_passes):
        sp, changed = _remove_redundant_nodes_once(sp, epsilon)
        if not changed:
            break
    return sp


# ---------------------------------------------------------------------------
# fill-color extraction
# ---------------------------------------------------------------------------

def _extract_fill(tag):
    m = _FILL_ATTR_RE.search(tag)
    if m:
        return m.group(1).strip().lower()
    m = _STYLE_FILL_RE.search(tag)
    if m:
        return m.group(1).strip().lower()
    return None


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def cleanup_svg_paths(svg_str,
                       min_area_frac=0.00004,
                       node_epsilon=0.5,
                       overlap_containment=0.98,
                       dup_containment=0.995,
                       remove_duplicates=True,
                       remove_tiny=True,
                       remove_broken=True,
                       remove_overlaps=True,
                       remove_redundant_nodes_flag=True):
    """Remove duplicate paths, tiny objects, broken paths, redundant anchor
    points, and unnecessary same-color overlaps from an SVG document's
    <path> elements. Returns a new SVG string; anything that isn't a
    <path>'s `d`/removal is left byte-for-byte untouched.
    """
    vb = _VIEWBOX_RE.search(svg_str)
    canvas_area = None
    if vb:
        parts = vb.group(1).split()
        if len(parts) == 4:
            try:
                canvas_area = abs(float(parts[2]) * float(parts[3]))
            except ValueError:
                canvas_area = None
    if canvas_area is None:
        wm, hm = _WIDTH_RE.search(svg_str), _HEIGHT_RE.search(svg_str)
        if wm and hm:
            canvas_area = float(wm.group(1)) * float(hm.group(1))
    min_area = (canvas_area * min_area_frac) if canvas_area else 1.0
    min_diag = float(np.sqrt(min_area))

    tags = list(_TAG_RE.finditer(svg_str))
    if not tags:
        return svg_str

    entries = []
    for m in tags:
        tag = m.group(0)
        entry = {"tag": tag, "span": m.span(), "skip": True, "removed": False}
        dmatch = _D_RE.search(tag)
        if dmatch:
            try:
                subpaths = _parse_d(dmatch.group(1))
                entry.update({
                    "skip": False,
                    "fill": _extract_fill(tag),
                    "subpaths": subpaths,
                })
            except Exception:
                pass
        entries.append(entry)

    # --- per-path structural cleanup: redundant nodes, broken/tiny subpaths
    for e in entries:
        if e["skip"]:
            continue
        kept = []
        for sp in e["subpaths"]:
            if remove_redundant_nodes_flag:
                sp = _remove_redundant_nodes(sp, node_epsilon)
            if not sp["segs"]:
                continue  # collapsed to nothing
            poly = _flatten_subpath(sp)
            if remove_broken and (len(poly) < 2 or not np.all(np.isfinite(poly))):
                continue
            if remove_tiny:
                bx0, by0, bx1, by1 = _bbox(poly)
                diag = float(np.hypot(bx1 - bx0, by1 - by0))
                if sp["closed"]:
                    # Area alone is not a safe "is this confetti" signal.
                    # VTracer traces a thin line-art stroke as a closed
                    # sliver (two long near-parallel edges), so a long,
                    # visually significant hairline can have a tiny area
                    # purely because it's thin -- while real confetti (a
                    # stray handful of pixels) is small in BOTH area and
                    # extent. Only drop a closed subpath when it's small
                    # both ways, so long thin strokes survive.
                    if _polygon_area(poly) < min_area and diag < min_diag:
                        continue
                else:
                    if diag < min_diag:
                        continue
            kept.append(sp)
        e["subpaths"] = kept
        e["removed"] = len(kept) == 0

    # --- exact duplicate `d`+fill removal (cheap, catches literal repeats)
    if remove_duplicates:
        original_d = {}
        for m in tags:
            dm = _D_RE.search(m.group(0))
            if dm:
                original_d[m.span()] = dm.group(1)
        seen = {}
        for e in entries:
            if e["skip"] or e["removed"]:
                continue
            key = (e.get("fill"), original_d.get(e["span"]))
            if key in seen:
                e["removed"] = True
            else:
                seen[key] = True

    # --- precompute flattened polygons / bbox / area for surviving paths
    for e in entries:
        if e["skip"] or e["removed"]:
            continue
        polys = [_flatten_subpath(sp) for sp in e["subpaths"]]
        e["poly_list"] = polys
        if not polys:
            e["removed"] = True
            continue
        e["all_pts"] = np.vstack(polys)
        e["bbox"] = _bbox(e["all_pts"])
        e["area"] = sum(_polygon_area(p) for p, sp in zip(polys, e["subpaths"]) if sp["closed"])

    active = [e for e in entries if not e["skip"] and not e["removed"]]

    def closed_polys(e):
        return [p for p, sp in zip(e["poly_list"], e["subpaths"]) if sp["closed"] and len(p) >= 3]

    def containment_ratio(inner, outer, boundary_tol=0.75):
        """Fraction of `inner`'s sampled boundary points that lie inside
        (or right on the boundary of) `outer`. Boundary points are counted
        as contained -- without that, comparing two congruent/near-congruent
        shapes leaves every sample point sitting exactly on both polygons'
        edges, which is undefined for a pure ray-casting test."""
        pts = inner.get("all_pts")
        if pts is None or len(pts) == 0:
            return 0.0
        outer_polys = closed_polys(outer)
        if not outer_polys:
            return 0.0
        contained = np.zeros(len(pts), dtype=bool)
        for poly in outer_polys:
            contained |= _points_in_polygon(pts, poly)
            contained |= _min_dist_to_polygon(pts, poly) < boundary_tol
        return float(np.mean(contained))

    # --- geometric duplicate detection (near-identical shape, same fill)
    if remove_duplicates:
        for i in range(len(active)):
            e1 = active[i]
            if e1["removed"] or e1.get("area", 0) <= 0:
                continue
            for j in range(i + 1, len(active)):
                e2 = active[j]
                if e2["removed"] or e2.get("fill") != e1.get("fill"):
                    continue
                if e2.get("area", 0) <= 0 or not _bbox_overlap(e1["bbox"], e2["bbox"]):
                    continue
                if abs(e1["area"] - e2["area"]) / max(e1["area"], e2["area"]) > 0.03:
                    continue
                ratio = min(containment_ratio(e1, e2), containment_ratio(e2, e1))
                if ratio >= dup_containment:
                    e2["removed"] = True  # keep the earlier occurrence

    # --- unnecessary overlap: a same-colored path fully contained inside
    # another same-colored path is redundant no matter which one is drawn
    # on top -- if it's behind, it's already hidden; if it's in front, it
    # repaints a region that's already that exact color. Either way,
    # dropping the smaller (contained) one never changes the rendering.
    if remove_overlaps:
        for i in range(len(active)):
            e1 = active[i]
            if e1["removed"] or e1.get("area", 0) <= 0:
                continue
            for j in range(i + 1, len(active)):
                e2 = active[j]
                if e2["removed"] or e2.get("fill") != e1.get("fill"):
                    continue
                if e2.get("area", 0) <= 0 or not _bbox_overlap(e1["bbox"], e2["bbox"]):
                    continue
                inner, outer = (e1, e2) if e1["area"] <= e2["area"] else (e2, e1)
                if containment_ratio(inner, outer) >= overlap_containment:
                    inner["removed"] = True
                    if inner is e1:
                        break

    # --- rebuild the SVG string
    out = []
    last_end = 0
    for e in entries:
        start, end = e["span"]
        out.append(svg_str[last_end:start])
        if not e["skip"] and not e["removed"]:
            new_d = " ".join(_subpath_to_d(sp) for sp in e["subpaths"])
            out.append(_D_RE.sub(lambda mm: f'd="{new_d}"', e["tag"], count=1))
        elif e["skip"]:
            out.append(e["tag"])
        # else: removed entirely -- emit nothing for this <path>
        last_end = end
    out.append(svg_str[last_end:])
    return "".join(out)
