"""
Automatic color segmentation for VTracer's SVG output.

Detects the main colors used in a traced SVG and turns VTracer's flat list
of <path> elements into color-addressable shapes:

  - detect_svg_color_palette() -- aggregates every <path>'s fill color and
    its filled area, and reports the dominant colors as a percentage of the
    total drawn area. Colors that individually cover less than
    `min_percent` of the drawn area are dropped -- they're pixel-level
    speckle, not a "main color" of the image.

  - group_svg_by_color() -- re-emits the SVG with same-colored paths
    gathered under one <g data-color="#rrggbb" data-percent=".."
    data-paths="n"> per color, so a color's shapes can be selected, edited,
    or hidden as a single unit in any SVG editor. Paths of the SAME color
    are never reordered relative to each other -- painting a flat color
    over itself in a different order never changes how it looks -- so the
    only thing that can move is where a color's *first* occurrence sits
    relative to *other* colors. That's a no-op for VTracer's default
    "stacked" hierarchical mode, which already emits one contiguous run of
    paths per color.

  - split_svg_by_color() -- builds one standalone SVG document per detected
    color: same canvas size/viewBox as the source, containing only that
    color's paths. This is the most literal reading of "separate vector
    shapes for each color" -- e.g. for multi-layer laser cutting, screen-
    print color separations, multi-material 3D printing, or per-color
    editing in another tool.

  - build_color_layers_zip() -- packages split_svg_by_color()'s output as
    a downloadable .zip, one .svg file per color.

This module only rearranges/regroups the <path> elements VTracer (and the
cleanup/smoothing passes) already produced -- it never re-traces anything,
and never invents or blends a color that wasn't already an exact fill in
the input SVG.
"""

import io
import re
import zipfile

import numpy as np

_SVG_OPEN_RE = re.compile(r'<svg\b[^>]*>', re.IGNORECASE | re.DOTALL)
_SVG_CLOSE_RE = re.compile(r'</svg\s*>', re.IGNORECASE)
_TAG_RE = re.compile(r'<path\b[^>]*?/?>', re.IGNORECASE | re.DOTALL)
_D_RE = re.compile(r'\bd="([^"]*)"', re.IGNORECASE)
_FILL_ATTR_RE = re.compile(r'\bfill="([^"]*)"', re.IGNORECASE)
_STYLE_FILL_RE = re.compile(r'fill:\s*([^;"]+)', re.IGNORECASE)
_HEX6_RE = re.compile(r'#[0-9a-f]{6}')
_CMD_TOKEN_RE = re.compile(r'[MLCZmlcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?')


# ---------------------------------------------------------------------------
# minimal, self-contained `d`-string parsing (mirrors path_cleanup._parse_d,
# duplicated rather than imported so this module has no dependency on
# path_cleanup's internals -- only what's needed for area estimation)
# ---------------------------------------------------------------------------

def _parse_d(d):
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
            cmd = "L"
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


def _flatten_subpath(sp, curve_samples=8):
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


def _extract_fill(tag):
    m = _FILL_ATTR_RE.search(tag)
    if m:
        return m.group(1).strip()
    m = _STYLE_FILL_RE.search(tag)
    if m:
        return m.group(1).strip()
    return None


def _hex_norm(fill):
    """Normalize a fill value to '#rrggbb' lowercase where possible. Anything
    that isn't a plain hex color (e.g. 'none', 'url(#grad)', a named color)
    is passed through unchanged so it never gets silently misclassified as
    a "main color" it isn't."""
    if not fill:
        return None
    f = fill.strip().lower()
    if re.fullmatch(r'#[0-9a-f]{6}', f):
        return f
    if re.fullmatch(r'#[0-9a-f]{3}', f):
        return "#" + "".join(c * 2 for c in f[1:])
    return f


# ---------------------------------------------------------------------------
# shared parsing: every <path>'s tag, fill, and filled area
# ---------------------------------------------------------------------------

def _path_entries(svg_str):
    entries = []
    for m in _TAG_RE.finditer(svg_str):
        tag = m.group(0)
        fill = _hex_norm(_extract_fill(tag))
        area = 0.0
        dmatch = _D_RE.search(tag)
        if dmatch:
            try:
                for sp in _parse_d(dmatch.group(1)):
                    if not sp["closed"]:
                        continue
                    poly = _flatten_subpath(sp)
                    if len(poly) >= 3:
                        area += _polygon_area(poly)
            except Exception:
                pass
        entries.append({"tag": tag, "span": m.span(), "fill": fill, "area": area})
    return entries


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def detect_svg_color_palette(svg_str, min_percent=0.5, max_colors=20):
    """Return the dominant colors used by an SVG's <path> fills, as a list
    of {"hex", "area", "percent", "path_count"} sorted by drawn area,
    descending. Only plain hex fills count toward the palette --
    `fill="none"`, gradients, and named colors are ignored.
    """
    entries = _path_entries(svg_str)
    totals, counts = {}, {}
    for e in entries:
        fill = e["fill"]
        if not fill or not _HEX6_RE.fullmatch(fill):
            continue
        totals[fill] = totals.get(fill, 0.0) + e["area"]
        counts[fill] = counts.get(fill, 0) + 1

    total_area = sum(totals.values())
    if total_area <= 0:
        return []

    palette = []
    for hexcolor, area in totals.items():
        percent = 100.0 * area / total_area
        if percent < min_percent:
            continue
        # Cast away from numpy scalar types (float64 etc.) -- area/percent
        # are computed with numpy, but this dict is returned straight
        # through jsonify(), whose default encoder can't serialize them.
        palette.append({
            "hex": hexcolor,
            "area": round(float(area), 2),
            "percent": round(float(percent), 2),
            "path_count": int(counts[hexcolor]),
        })
    palette.sort(key=lambda c: c["area"], reverse=True)
    return palette[:max_colors]


def group_svg_by_color(svg_str):
    """Re-emit the SVG with same-fill <path> elements gathered under a
    <g data-color="#rrggbb" data-percent=".." data-paths="n">, one group per
    distinct hex fill, in first-seen order. Paths without a plain hex fill
    (none/gradient/named) are left in place, ungrouped."""
    open_m = _SVG_OPEN_RE.search(svg_str)
    close_m = _SVG_CLOSE_RE.search(svg_str)
    if not open_m or not close_m:
        return svg_str

    entries = _path_entries(svg_str)
    if not entries:
        return svg_str

    palette = {c["hex"]: c for c in detect_svg_color_palette(svg_str, min_percent=0.0)}

    body_start = open_m.end()
    prefix_gap = svg_str[body_start:entries[0]["span"][0]]

    order = []          # first-seen sequence: hex color, or ("__raw__", idx)
    buckets = {}         # hex -> [tag, ...]
    passthrough = []     # tags with no plain hex fill, kept inline in place

    for e in entries:
        hexcolor = e["fill"]
        if hexcolor and _HEX6_RE.fullmatch(hexcolor):
            if hexcolor not in buckets:
                buckets[hexcolor] = []
                order.append(hexcolor)
            buckets[hexcolor].append(e["tag"])
        else:
            order.append(("__raw__", len(passthrough)))
            passthrough.append(e["tag"])

    out = [svg_str[:body_start], prefix_gap]
    seen = set()
    for item in order:
        if isinstance(item, tuple):
            out.append(passthrough[item[1]])
            continue
        if item in seen:
            continue
        seen.add(item)
        info = palette.get(item, {})
        out.append(
            f'<g data-color="{item}" data-percent="{info.get("percent", 0.0)}" '
            f'data-paths="{info.get("path_count", len(buckets[item]))}">'
        )
        out.append("".join(buckets[item]))
        out.append("</g>")
    out.append(svg_str[close_m.start():])
    return "".join(out)


def split_svg_by_color(svg_str):
    """Build one standalone SVG document per distinct hex fill color found
    among the <path> elements -- same opening <svg ...> tag (so canvas
    size/viewBox match the source) but containing only that color's paths.
    Returns {hex: svg_string}. Paths without a plain hex fill are skipped --
    they can't be attributed to a single "main color"."""
    open_m = _SVG_OPEN_RE.search(svg_str)
    if not open_m:
        return {}
    svg_open = open_m.group(0)

    buckets = {}
    for e in _path_entries(svg_str):
        hexcolor = e["fill"]
        if hexcolor and _HEX6_RE.fullmatch(hexcolor):
            buckets.setdefault(hexcolor, []).append(e["tag"])

    return {hexcolor: svg_open + "".join(tags) + "</svg>" for hexcolor, tags in buckets.items()}


def build_color_layers_zip(svg_str, min_percent=0.5):
    """Package split_svg_by_color()'s output as a .zip, one .svg per
    detected main color (colors below `min_percent` are excluded, matching
    detect_svg_color_palette()). Returns the zip file's raw bytes."""
    palette = detect_svg_color_palette(svg_str, min_percent=min_percent)
    layers = split_svg_by_color(svg_str)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, color in enumerate(palette, start=1):
            svg_doc = layers.get(color["hex"])
            if not svg_doc:
                continue
            name = f"layer_{i:02d}_{color['hex'].lstrip('#')}.svg"
            zf.writestr(name, svg_doc)
    return buf.getvalue()
