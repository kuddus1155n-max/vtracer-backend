"""
SVG -> EPS (Encapsulated PostScript) export for VTracer's cleaned/smoothed
output.

Only handles what this pipeline ever produces: a flat list of
<path d="M ... L ... C ... Z" fill="#rrggbb"> elements inside an SVG with
a viewBox (or width/height). That's exactly cleanup_svg_paths()'s and
smooth_svg_paths()'s output format (see path_cleanup.py / path_smoothing.py),
so no general-purpose SVG feature support (gradients, clipping, transforms,
strokes, text, etc.) is needed or attempted here.

Each <path> becomes its own independent `newpath ... fill` block in the EPS
content stream, in the same front-to-back paint order the SVG uses -- never
merged or grouped -- which is what lets the exported EPS still open as
fully separate, editable path objects in Adobe Illustrator (or any other
EPS-reading vector editor), the same way any flat-color vector EPS does.

SVG's coordinate system has y increasing downward from the top-left;
PostScript/EPS has y increasing upward from the bottom-left. Every point
is flipped (y' = canvas_height - y) when it's written out.
"""

import math
import re

_TAG_RE = re.compile(r'<path\b[^>]*?/?>', re.IGNORECASE | re.DOTALL)
_D_RE = re.compile(r'\bd="([^"]*)"', re.IGNORECASE)
_FILL_ATTR_RE = re.compile(r'\bfill="([^"]*)"', re.IGNORECASE)
_STYLE_FILL_RE = re.compile(r'fill:\s*([^;"]+)', re.IGNORECASE)
_OPACITY_ATTR_RE = re.compile(r'\bfill-opacity="([^"]*)"', re.IGNORECASE)
_VIEWBOX_RE = re.compile(r'viewBox="([\-\d.\s]+)"', re.IGNORECASE)
_WIDTH_RE = re.compile(r'\bwidth="(\d+(?:\.\d+)?)', re.IGNORECASE)
_HEIGHT_RE = re.compile(r'\bheight="(\d+(?:\.\d+)?)', re.IGNORECASE)

_CMD_TOKEN_RE = re.compile(r'[MLCZmlcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?')

_NAMED_COLORS = {
    "black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0), "green": (0.0, 0.5, 0.0), "blue": (0.0, 0.0, 1.0),
}


# ---------------------------------------------------------------------------
# d-string parsing -- kept as its own copy (rather than importing
# path_cleanup._parse_d) so this module has no dependency on another
# module's internals; each parsing module in this pipeline already keeps
# its own copy tuned for its own needs (see path_smoothing.py's docstring).
# ---------------------------------------------------------------------------

def _parse_d(d):
    """Parse an absolute M/L/C/Z `d` string into subpaths:
    [{"start": (x, y), "segs": [seg, ...], "closed": bool}, ...]
    where each seg is ("L", (x, y)) or ("C", (x1,y1), (x2,y2), (x, y))."""
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


# ---------------------------------------------------------------------------
# color / geometry / formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _extract_fill(tag):
    m = _FILL_ATTR_RE.search(tag)
    if m:
        return m.group(1).strip()
    m = _STYLE_FILL_RE.search(tag)
    if m:
        return m.group(1).strip()
    return None


def _extract_opacity(tag):
    m = _OPACITY_ATTR_RE.search(tag)
    if not m:
        return 1.0
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 1.0


def _color_to_rgb01(color):
    """Parses '#rgb', '#rrggbb', 'rgb(r,g,b)', or a small set of CSS color
    keywords into (r, g, b) floats 0-1. Returns None for 'none'/
    'transparent' (nothing to paint). Falls back to black for anything else
    unrecognized, rather than raising -- one odd fill shouldn't sink the
    whole export."""
    if not color:
        return (0.0, 0.0, 0.0)
    c = color.strip().lower()
    if c in ("none", "transparent"):
        return None
    if c.startswith("#"):
        hex_part = c[1:]
        if len(hex_part) == 3:
            hex_part = "".join(ch * 2 for ch in hex_part)
        if len(hex_part) >= 6:
            try:
                r = int(hex_part[0:2], 16) / 255.0
                g = int(hex_part[2:4], 16) / 255.0
                b = int(hex_part[4:6], 16) / 255.0
                return (r, g, b)
            except ValueError:
                return (0.0, 0.0, 0.0)
        return (0.0, 0.0, 0.0)
    m = re.match(r"rgb\(\s*([\d.]+)%?\s*,\s*([\d.]+)%?\s*,\s*([\d.]+)%?\s*\)", c)
    if m:
        is_pct = "%" in c
        return tuple((float(g_) / 100.0) if is_pct else (float(g_) / 255.0) for g_ in m.groups())
    return _NAMED_COLORS.get(c, (0.0, 0.0, 0.0))


def _get_canvas_size(svg_str):
    vb = _VIEWBOX_RE.search(svg_str)
    if vb:
        parts = vb.group(1).split()
        if len(parts) == 4:
            try:
                _x0, _y0, w, h = (float(p) for p in parts)
                if w > 0 and h > 0:
                    return w, h
            except ValueError:
                pass
    wm, hm = _WIDTH_RE.search(svg_str), _HEIGHT_RE.search(svg_str)
    if wm and hm:
        try:
            w, h = float(wm.group(1)), float(hm.group(1))
            if w > 0 and h > 0:
                return w, h
        except ValueError:
            pass
    return 1000.0, 1000.0  # last-resort fallback so export never crashes on a stray/edited SVG


def _subpath_to_ps(sp, height):
    """Emits moveto/lineto/curveto operators for one subpath, flipping y
    (SVG is y-down from the top; PostScript is y-up from the bottom)."""

    def flip(p):
        return p[0], height - p[1]

    x0, y0 = flip(sp["start"])
    lines = [f"{_fmt(x0)} {_fmt(y0)} moveto"]
    for seg in sp["segs"]:
        if seg[0] == "L":
            x, y = flip(seg[1])
            lines.append(f"{_fmt(x)} {_fmt(y)} lineto")
        else:
            _, c1, c2, end = seg
            x1, y1 = flip(c1)
            x2, y2 = flip(c2)
            x3, y3 = flip(end)
            lines.append(f"{_fmt(x1)} {_fmt(y1)} {_fmt(x2)} {_fmt(y2)} {_fmt(x3)} {_fmt(y3)} curveto")
    if sp["closed"]:
        lines.append("closepath")
    return lines


def _eps_comment_safe(text):
    """PostScript comment lines can't contain newlines/CR; strip them so a
    stray title never breaks the header."""
    return re.sub(r"[\r\n]+", " ", text or "").strip() or "Vectorized Image"


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def svg_to_eps(svg_str, title="Vectorized Image"):
    """Converts a flat-<path>-only SVG document (the shape this pipeline's
    cleanup/smoothing stages always produce) into an EPS document string.

    Each <path> becomes its own newpath/moveto/.../fill block, in the same
    order the SVG paints them, so the exported EPS keeps every shape as a
    separate, independently editable object when opened in Adobe
    Illustrator (or any other EPS-reading vector editor) -- nothing here
    rasterizes or flattens the artwork.
    """
    width, height = _get_canvas_size(svg_str)
    tags = [m.group(0) for m in _TAG_RE.finditer(svg_str)]

    body = []
    for tag in tags:
        dmatch = _D_RE.search(tag)
        if not dmatch:
            continue
        fill = _extract_fill(tag)
        rgb = _color_to_rgb01(fill)
        if rgb is None:
            continue  # fill="none" -- nothing to paint
        opacity = _extract_opacity(tag)
        try:
            subpaths = _parse_d(dmatch.group(1))
        except Exception:
            continue  # skip anything malformed rather than failing the whole export
        subpaths = [sp for sp in subpaths if sp["segs"]]
        if not subpaths:
            continue

        body.append("newpath")
        for sp in subpaths:
            body.extend(_subpath_to_ps(sp, height))

        r, g, b = rgb
        if opacity < 1.0:
            # Plain PostScript/EPS Level 2 has no native alpha compositing.
            # Approximate partial opacity by blending the fill toward white
            # so flattened viewers/printers still show something reasonable,
            # and leave a comment noting it's an approximation.
            r = 1 - (1 - r) * opacity
            g = 1 - (1 - g) * opacity
            b = 1 - (1 - b) * opacity
            body.append(f"{_fmt(r)} {_fmt(g)} {_fmt(b)} setrgbcolor % approximated {opacity:.2f} opacity, no native alpha in EPS")
        else:
            body.append(f"{_fmt(r)} {_fmt(g)} {_fmt(b)} setrgbcolor")
        body.append("fill")

    bbox_w, bbox_h = math.ceil(width), math.ceil(height)
    header = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        "%%Creator: Nur Meta AI - VTracer EPS export",
        f"%%Title: {_eps_comment_safe(title)}",
        f"%%BoundingBox: 0 0 {bbox_w} {bbox_h}",
        f"%%HiResBoundingBox: 0.000000 0.000000 {width:.6f} {height:.6f}",
        "%%DocumentData: Clean7Bit",
        "%%LanguageLevel: 2",
        "%%Pages: 1",
        "%%EndComments",
        "%%Page: 1 1",
    ]
    footer = ["", "%%EOF"]

    return "\n".join(header + body + footer)
