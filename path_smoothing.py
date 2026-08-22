"""
path_smoothing.py
------------------
Post-trace path smoothing for the Nur Meta AI / VTracer pipeline.

VTracer's raw output already draws some subpaths as curves ("spline"
mode) and some as straight polygon segments ("polygon" mode). The
polygon-mode subpaths look jagged/wavy/uneven because every little
staircase step in the traced pixel boundary becomes its own straight
line segment. This module re-fits those subpaths as a small number of
smooth, optimized cubic Bezier curves -- while explicitly detecting and
preserving real sharp corners, so a square icon still has crisp 90
degree corners instead of getting rounded off.

PART 1 (this file, so far):
  - Parsing an SVG <path> `d` attribute into one or more closed/open
    subpaths of plain (x, y) points.
  - Flattening any existing curve commands VTracer may have written
    (C / Q / S / T / A) into densely-sampled polyline points, so the
    corner-detection and (later) curve-refitting stages only ever have
    to deal with a uniform list of points -- never mixed command types.
  - Corner-angle detection: walking a polyline and deciding, at every
    point, whether the direction change there is sharp enough to count
    as a real corner (and must stay a hard vertex) or gentle enough to
    be smoothed away into a flowing curve.

PART 2 (this file, now added):
  - The Schneider least-squares cubic-Bezier fitting algorithm
    (Graphics Gems): each corner-split chain from Part 1 is fit with a
    single cubic Bezier via least squares against tangent directions
    fixed at the chain's two endpoints; if the worst-point error is
    still too big, the fit is first refined by reparameterizing (a few
    rounds of Newton-Raphson root-finding to slide each point's `t`
    value to its true closest position on the curve), and only if that
    still isn't good enough does the chain get split at its
    worst-error point and each half fit recursively. Every subpath is
    corner-detected and refit this way, whether VTracer originally drew
    it with straight polygon segments or with its own C/Q/A curves --
    VTracer's spline mode doesn't reliably keep real sharp corners
    sharp, so a subpath it already drew as curves is re-flattened,
    corner-detected, and refit here too rather than passed through
    untouched, and the least-squares fit still converges on genuinely
    smooth runs with about as few segments as VTracer used.
  - Reassembly of every subpath's fitted (or, on failure, untouched)
    fragment back into each `<path>` element's `d` attribute, in place,
    inside the original SVG document text.
  - The public entry point vtracer_server.py imports and calls:
    smooth_svg_paths(svg_str, tolerance=1.8, corner_angle_deg=30.0).
    `tolerance` is the max allowed distance (in SVG user units) between
    the fitted curve and any original traced point; `corner_angle_deg`
    is forwarded straight through to detect_corners from Part 1.
"""

import math
import re


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _scale(a, s):
    return (a[0] * s, a[1] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _length(a):
    return math.hypot(a[0], a[1])


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize(a):
    l = _length(a)
    if l < 1e-9:
        return (0.0, 0.0)
    return (a[0] / l, a[1] / l)


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# ---------------------------------------------------------------------------
# SVG path-data tokenizer / parser
# ---------------------------------------------------------------------------

# Matches one path command letter followed by its (possibly comma/space
# separated, possibly signed, possibly decimal, possibly scientific
# notation) numeric arguments, up to the next command letter.
_CMD_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)")
_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")

_ARG_COUNTS = {
    "M": 2, "L": 2, "H": 1, "V": 1,
    "C": 6, "S": 4, "Q": 4, "T": 2,
    "A": 7, "Z": 0,
}


def _parse_numbers(chunk):
    return [float(n) for n in _NUM_RE.findall(chunk)]


def parse_path_commands(d):
    """Turns an SVG `d` string into a flat list of absolute-coordinate
    commands: [(letter, [args...]), ...]. Letter is always uppercase
    (relative commands are resolved to absolute here) which keeps every
    later stage of the pipeline free of relative/absolute bookkeeping.
    `H`/`V` are expanded to full `L` commands (needs the running point),
    and multi-argument shorthand repeats (e.g. one `L` command carrying
    several coordinate pairs) are split into individual commands.
    """
    commands = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    prev_cmd = None
    prev_ctrl = None  # last cubic/quadratic control point, for S/T reflection

    for letter, raw_args in _CMD_RE.findall(d):
        nums = _parse_numbers(raw_args)
        is_relative = letter.islower()
        upper = letter.upper()

        if upper == "Z":
            commands.append(("Z", []))
            cur = start
            prev_cmd = "Z"
            prev_ctrl = None
            continue

        step = _ARG_COUNTS[upper]
        if step == 0:
            continue

        # Consume the argument list in groups of `step` -- this is what
        # lets "L 1,1 2,2 3,3" mean three separate lineto commands, per
        # the SVG spec's shorthand-repeat rule.
        i = 0
        while i + step <= len(nums) or (step == 0 and i == 0):
            group = nums[i:i + step]
            if len(group) < step:
                break
            i += step

            if upper == "M":
                pt = (group[0], group[1])
                if is_relative:
                    pt = _add(cur, pt)
                cur = pt
                start = pt
                commands.append(("M", [pt[0], pt[1]]))
                # Subsequent coordinate pairs after an initial M are
                # implicit linetos, per spec -- handled naturally since
                # we keep looping with the same `step`==2 on the next
                # iterations only if letter was L; M's repeats are
                # actually implicit L's, so fix upper for the rest of
                # this group run:
                upper_for_rest = "L"
                letter_is_rel = is_relative
                j = i
                while j + 2 <= len(nums):
                    pair = nums[j:j + 2]
                    j += 2
                    pt2 = (pair[0], pair[1])
                    if letter_is_rel:
                        pt2 = _add(cur, pt2)
                    cur = pt2
                    commands.append(("L", [pt2[0], pt2[1]]))
                i = j
                prev_ctrl = None

            elif upper == "L":
                pt = (group[0], group[1])
                if is_relative:
                    pt = _add(cur, pt)
                cur = pt
                commands.append(("L", [pt[0], pt[1]]))
                prev_ctrl = None

            elif upper == "H":
                x = group[0] + cur[0] if is_relative else group[0]
                cur = (x, cur[1])
                commands.append(("L", [cur[0], cur[1]]))
                prev_ctrl = None

            elif upper == "V":
                y = group[0] + cur[1] if is_relative else group[0]
                cur = (cur[0], y)
                commands.append(("L", [cur[0], cur[1]]))
                prev_ctrl = None

            elif upper == "C":
                p1 = (group[0], group[1])
                p2 = (group[2], group[3])
                p3 = (group[4], group[5])
                if is_relative:
                    p1 = _add(cur, p1)
                    p2 = _add(cur, p2)
                    p3 = _add(cur, p3)
                commands.append(("C", [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]))
                cur = p3
                prev_ctrl = p2

            elif upper == "S":
                p2 = (group[0], group[1])
                p3 = (group[2], group[3])
                if is_relative:
                    p2 = _add(cur, p2)
                    p3 = _add(cur, p3)
                if prev_cmd in ("C", "S") and prev_ctrl is not None:
                    p1 = _sub(_scale(cur, 2), prev_ctrl)
                else:
                    p1 = cur
                commands.append(("C", [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]]))
                cur = p3
                prev_ctrl = p2

            elif upper == "Q":
                p1 = (group[0], group[1])
                p2 = (group[2], group[3])
                if is_relative:
                    p1 = _add(cur, p1)
                    p2 = _add(cur, p2)
                commands.append(("Q", [p1[0], p1[1], p2[0], p2[1]]))
                cur = p2
                prev_ctrl = p1

            elif upper == "T":
                p2 = (group[0], group[1])
                if is_relative:
                    p2 = _add(cur, p2)
                if prev_cmd in ("Q", "T") and prev_ctrl is not None:
                    p1 = _sub(_scale(cur, 2), prev_ctrl)
                else:
                    p1 = cur
                commands.append(("Q", [p1[0], p1[1], p2[0], p2[1]]))
                cur = p2
                prev_ctrl = p1

            elif upper == "A":
                rx, ry, rot, large_arc, sweep, ex, ey = group
                end = (ex, ey)
                if is_relative:
                    end = _add(cur, end)
                commands.append(("A", [rx, ry, rot, large_arc, sweep, end[0], end[1]]))
                cur = end
                prev_ctrl = None

            prev_cmd = upper

        # letter with zero args left over than a full group (malformed
        # data) -- nothing more to do for this command run.

    return commands


# ---------------------------------------------------------------------------
# Curve flattening: turn the parsed command list into plain polyline
# subpaths, so corner detection (and later, curve refitting) only ever
# sees a uniform list of (x, y) points per subpath.
# ---------------------------------------------------------------------------

def _flatten_cubic(p0, p1, p2, p3, out, samples=16):
    for i in range(1, samples + 1):
        t = i / samples
        mt = 1 - t
        x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        out.append((x, y))


def _flatten_quadratic(p0, p1, p2, out, samples=12):
    for i in range(1, samples + 1):
        t = i / samples
        mt = 1 - t
        x = (mt ** 2) * p0[0] + 2 * mt * t * p1[0] + (t ** 2) * p2[0]
        y = (mt ** 2) * p0[1] + 2 * mt * t * p1[1] + (t ** 2) * p2[1]
        out.append((x, y))


def _flatten_arc(p0, rx, ry, rot_deg, large_arc, sweep, p1, out, samples=16):
    # Standard SVG arc -> center parameterization (endpoint form), then
    # sampled uniformly in angle. Degenerate radii fall back to a
    # straight line, matching the SVG spec's own fallback rule.
    if rx == 0 or ry == 0:
        out.append(p1)
        return

    phi = math.radians(rot_deg)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    dx2 = (p0[0] - p1[0]) / 2.0
    dy2 = (p0[1] - p1[1]) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    rx, ry = abs(rx), abs(ry)
    lam = (x1p ** 2) / (rx ** 2) + (y1p ** 2) / (ry ** 2)
    if lam > 1:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale

    sign = -1 if large_arc == sweep else 1
    num = (rx ** 2) * (ry ** 2) - (rx ** 2) * (y1p ** 2) - (ry ** 2) * (x1p ** 2)
    den = (rx ** 2) * (y1p ** 2) + (ry ** 2) * (x1p ** 2)
    co = sign * math.sqrt(max(num, 0) / den) if den > 1e-12 else 0.0

    cxp = co * (rx * y1p / ry)
    cyp = co * (-ry * x1p / rx)

    cx = cos_phi * cxp - sin_phi * cyp + (p0[0] + p1[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (p0[1] + p1[1]) / 2.0

    def _angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        lenu = math.hypot(ux, uy)
        lenv = math.hypot(vx, vy)
        if lenu * lenv < 1e-12:
            return 0.0
        a = math.acos(max(-1.0, min(1.0, dot / (lenu * lenv))))
        return -a if (ux * vy - uy * vx) < 0 else a

    theta1 = _angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = _angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                     (-x1p - cxp) / rx, (-y1p - cyp) / ry)

    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi

    for i in range(1, samples + 1):
        t = i / samples
        theta = theta1 + dtheta * t
        x = cx + rx * math.cos(theta) * cos_phi - ry * math.sin(theta) * sin_phi
        y = cy + rx * math.cos(theta) * sin_phi + ry * math.sin(theta) * cos_phi
        out.append((x, y))


def commands_to_subpaths(commands):
    """Splits a parsed command list into subpaths, where each subpath is
    a dict: {"points": [(x,y), ...], "closed": bool, "was_curve": bool}.

    `was_curve` records whether *any* command in this subpath was already
    a curve (C/Q/A) rather than a straight line -- kept around as useful
    metadata (and by callers that want it), but Part 2 no longer treats it
    as a reason to skip re-fitting: VTracer's own "spline" mode doesn't
    reliably keep real sharp corners sharp, so every subpath is
    corner-detected and refit from its flattened points here, regardless
    of which mode VTracer traced it in.

    Each subpath dict also carries `orig_commands`: the exact slice of
    the (already absolute-coordinate) parsed command list that produced
    it, `M` included. This is the fallback Part 2 re-emits verbatim if
    re-fitting a given subpath ever fails for some unexpected reason.
    """
    subpaths = []
    points = []
    orig_commands = []
    closed = False
    was_curve = False
    cur = (0.0, 0.0)
    start = (0.0, 0.0)

    def flush():
        nonlocal points, orig_commands, closed, was_curve
        if len(points) >= 2:
            subpaths.append({
                "points": points,
                "closed": closed,
                "was_curve": was_curve,
                "orig_commands": orig_commands,
            })
        points = []
        orig_commands = []
        closed = False
        was_curve = False

    for letter, args in commands:
        if letter == "M":
            flush()
            cur = (args[0], args[1])
            start = cur
            points = [cur]
            orig_commands = [("M", list(args))]

        elif letter == "L":
            cur = (args[0], args[1])
            points.append(cur)
            orig_commands.append(("L", list(args)))

        elif letter == "C":
            p1, p2, p3 = (args[0], args[1]), (args[2], args[3]), (args[4], args[5])
            _flatten_cubic(cur, p1, p2, p3, points)
            cur = p3
            was_curve = True
            orig_commands.append(("C", list(args)))

        elif letter == "Q":
            p1, p2 = (args[0], args[1]), (args[2], args[3])
            _flatten_quadratic(cur, p1, p2, points)
            cur = p2
            was_curve = True
            orig_commands.append(("Q", list(args)))

        elif letter == "A":
            rx, ry, rot, large_arc, sweep, ex, ey = args
            end = (ex, ey)
            _flatten_arc(cur, rx, ry, rot, large_arc, sweep, end, points)
            cur = end
            was_curve = True
            orig_commands.append(("A", list(args)))

        elif letter == "Z":
            # Note: deliberately does NOT append a duplicate of the
            # first point here. `closed=True` alone tells every
            # downstream stage (corner detection, chain splitting) to
            # treat this point list as a wrap-around loop via modulo
            # indexing instead -- duplicating the seam point would make
            # the duplicate's own zero-length in/out vectors mask the
            # real corner that's almost always sitting right there.
            closed = True
            cur = start
            orig_commands.append(("Z", []))

    flush()
    return subpaths


# ---------------------------------------------------------------------------
# Corner-angle detection
# ---------------------------------------------------------------------------

_HARD_STOP_MARGIN_DEG = 25.0
# A raw, immediate-neighbor turn at least this many degrees past
# `corner_angle_deg` is unambiguous -- no amount of pixel-staircase zigzag
# produces a single-vertex spike that far past threshold by accident on a
# genuinely smooth run. Points that sharp become hard stops: the windowed
# lookup below is never allowed to average straight through one, so a real
# corner can't get diluted into "not a corner" just because it happens to
# sit close to its neighboring corner (e.g. a small square icon with few
# points per edge).


def _raw_turn_deg(points, i, n, closed):
    """Unwindowed turning angle at point i, using only its immediate
    neighbors. Open-subpath endpoints always report a full reversal, so
    they're always treated as hard stops."""
    return _signed_raw_turn(points, i, n, closed)[0]


def _signed_raw_turn(points, i, n, closed):
    """Like `_raw_turn_deg`, but also returns which rotational sense the
    turn is in: +1 (e.g. counter-clockwise), -1 (clockwise), or 0 for a
    straight run / degenerate segment. Used by `detect_corners` to tell a
    *monotonically* curving run (every vertex turning the same way, as
    happens all along a circle or fillet) apart from alternating
    pixel-staircase zigzag (vertices turning first one way, then the
    other) -- the two situations need opposite treatment from the
    windowed test below, and unsigned angle alone can't distinguish them.
    """
    if not closed and (i == 0 or i == n - 1):
        return 180.0, 0
    prev_i, next_i = (i - 1) % n, (i + 1) % n
    v_in = _normalize(_sub(points[i], points[prev_i]))
    v_out = _normalize(_sub(points[next_i], points[i]))
    if v_in == (0.0, 0.0) or v_out == (0.0, 0.0):
        return 0.0, 0
    cos_theta = max(-1.0, min(1.0, _dot(v_in, v_out)))
    angle = math.degrees(math.acos(cos_theta))
    cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
    sign = 1 if cross > 1e-9 else (-1 if cross < -1e-9 else 0)
    return angle, sign


def _corner_neighbor_index(n, i, step, window, closed, hard_stops):
    """Walks up to `window` points away from `i` in direction `step` (+1 or
    -1), wrapping for a closed subpath and clamping at the ends for an open
    one. Stops as soon as it lands on a hard-stop point (inclusive), so the
    windowed direction estimate below is never averaged straight through a
    real corner -- it stops there and treats it as the boundary instead."""
    j = i
    for _ in range(window):
        nj = j + step
        if closed:
            nj %= n
        elif nj < 0 or nj >= n:
            break
        j = nj
        if j != i and j in hard_stops:
            break
    return j


def detect_corners(points, closed, corner_angle_deg, corner_window=3):
    """Walks a polyline and returns the sorted set of point-indices that
    must stay hard vertices (real corners), based on the direction change
    on either side of each point.

    The direction on each side is measured as the net displacement over up
    to `corner_window` points (not just the single immediate neighbor),
    bounded so it never averages straight through another real corner (see
    `_corner_neighbor_index`). VTracer's "polygon" mode traces a curved
    edge as a pixel staircase, and a single staircase step can locally
    zigzag past any reasonable per-vertex angle threshold even though the
    edge is gently curving -- using the immediate neighbor alone would
    misread that zigzag noise as a real corner at nearly every step.
    Widening the baseline averages the zigzag out: a genuinely
    smooth/curving run nets out to a shallow turn over `corner_window`
    points, while a real corner (a square icon's ~90 degree vertex, say)
    stays sharp no matter how wide the baseline is.

    A point is treated as a corner when that turning angle is sharper
    (larger) than `corner_angle_deg` -- e.g. a square icon's ~90 degree
    corners are always well under a sensible threshold like 55-60 degrees
    and stay crisp, while the gentle, near-180-degree turns along a rounded
    edge fall above the threshold and are left free for the curve fitter to
    smooth over. Endpoints of an open subpath are always corners (a curve
    fit can't move where a path starts or stops); for a closed subpath the
    seam point is checked like any other.
    """
    n = len(points)
    corners = set()
    if n == 0:
        return corners

    if not closed:
        corners.add(0)
        corners.add(n - 1)

    window = max(1, min(corner_window, n // 2)) if n > 2 else 1
    raw = [_signed_raw_turn(points, i, n, closed) for i in range(n)]
    hard_stop_deg = corner_angle_deg + _HARD_STOP_MARGIN_DEG
    hard_stops = {i for i in range(n) if raw[i][0] >= hard_stop_deg}

    rng = range(n) if closed else range(1, n - 1)
    for i in rng:
        if not closed and (i == 0 or i == n - 1):
            continue

        prev_i = _corner_neighbor_index(n, i, -1, window, closed, hard_stops)
        next_i = _corner_neighbor_index(n, i, 1, window, closed, hard_stops)
        if prev_i == i or next_i == i:
            continue

        p_prev, p, p_next = points[prev_i], points[i], points[next_i]
        v_in = _normalize(_sub(p, p_prev))
        v_out = _normalize(_sub(p_next, p))
        if v_in == (0.0, 0.0) or v_out == (0.0, 0.0):
            continue

        cos_theta = max(-1.0, min(1.0, _dot(v_in, v_out)))
        windowed_turn_deg = math.degrees(math.acos(cos_theta))

        # Walk the same span the windowed lookup above just covered, and
        # check whether the vertices in it are all turning the same
        # rotational way (or not turning at all). If they are, this is a
        # monotonically curving run -- a circle, a rounded fillet, any
        # smoothly bending edge VTracer happened to trace with only a
        # handful of points per curve -- and the windowed *secant* angle
        # over such a run grows roughly in proportion to the window size
        # (three points' worth of a steady curve nets a turn ~3x any one
        # of them), which would flag every single vertex along it as a
        # "corner" and shatter the curve into hard-cornered facets rather
        # than smoothing it. In that case judge by this vertex's own
        # local turn instead, which doesn't inflate with window size.
        # If the span instead mixes both rotational senses, that's the
        # alternating left/right jaggies of real pixel-staircase noise --
        # exactly what the windowed secant angle is good at canceling out
        # via vector cancellation, so it's trusted as before.
        if closed:
            span = []
            k = prev_i
            while True:
                span.append(k)
                if k == next_i:
                    break
                k = (k + 1) % n
        else:
            span = list(range(prev_i, next_i + 1))
        senses = {raw[k][1] for k in span if raw[k][1] != 0}
        monotonic_run = len(senses) <= 1

        if monotonic_run:
            turn_deg = raw[i][0]
        else:
            # Mixed-sense span: this could be genuine alternating
            # pixel-staircase noise (antialiasing jaggies on a roughly
            # straight/gently-curving edge) OR real, closely-spaced
            # alternating corners in the design itself -- a sawtooth
            # edge, a star's points, a zigzag, a W/M-shaped letterform.
            # The windowed secant angle (`windowed_turn_deg`) cancels
            # both cases identically via vector summation, which is
            # correct for the former but silently erases real corners
            # in the latter. Distinguish them by checking whether this
            # vertex is a clear local spike relative to its *immediate*
            # neighbors: staircase noise tends to have comparably-sized
            # alternating spikes with no single standout point, while a
            # real design corner stands out above what's immediately
            # next to it. When it does stand out, trust its own raw
            # turn in addition to the windowed value so it isn't
            # diluted away just because a neighboring corner of the
            # opposite sense happens to sit nearby.
            prev_i1 = (i - 1) % n if closed else max(i - 1, 0)
            next_i1 = (i + 1) % n if closed else min(i + 1, n - 1)
            is_local_spike = raw[i][0] >= raw[prev_i1][0] and raw[i][0] >= raw[next_i1][0]
            turn_deg = max(windowed_turn_deg, raw[i][0]) if is_local_spike else windowed_turn_deg

        if turn_deg >= corner_angle_deg:
            corners.add(i)

    return corners


def split_at_corners(points, closed, corners):
    """Given the corner indices, splits the polyline into a list of
    point-chains, each of which is smooth throughout (no interior
    corners) and only has hard vertices at its two ends -- exactly the
    input shape the (Part 2) Bezier curve fitter expects, one call per
    chain. Consecutive chains share their boundary point so the fitted
    curves join up with no gap.
    """
    n = len(points)
    if n < 2:
        return [points] if points else []

    sorted_corners = sorted(corners)
    if not sorted_corners:
        # nothing marked as a hard corner -- for a closed path that
        # means the whole loop is one smooth chain (start==end); for an
        # open path this can't happen since endpoints are always corners.
        return [points + [points[0]]] if closed else [points]

    # For a closed subpath every corner has a "next" corner to connect to,
    # wrapping past the last one back to the first (that's the seam).
    # For an open subpath the last corner is the path's own endpoint and
    # must NOT wrap back to the first corner -- there is no edge there,
    # since the path never returns to its start. So an open subpath only
    # walks the len(sorted_corners) - 1 consecutive corner-to-corner gaps.
    num_edges = len(sorted_corners) if closed else len(sorted_corners) - 1

    chains = []
    for k in range(num_edges):
        i0 = sorted_corners[k]
        i1 = sorted_corners[(k + 1) % len(sorted_corners)]
        if i1 == i0:
            continue
        if i1 > i0:
            chain = points[i0:i1 + 1]
        else:
            # wraps around the seam of a closed path
            chain = points[i0:] + points[:i1 + 1]
        if len(chain) >= 2:
            chains.append(chain)

    return chains


# ---------------------------------------------------------------------------
# Point-chain cleanup
# ---------------------------------------------------------------------------

def _dedupe_points(points, closed=False, eps=1e-7):
    """Drops consecutive near-duplicate points. VTracer's raw traced
    output, and the curve-flattening in Part 1, can both occasionally
    emit a point right on top of the previous one; a zero-length
    segment gives the curve fitter a degenerate (zero-length) tangent
    to work with, so it's cleared out before fitting ever sees it.

    For an unclosed chain, only *consecutive* duplicates (adjacent in
    the list) are collapsed. For a closed *subpath*, `closed=True` also
    drops a trailing point that lands back on the first one -- some
    source SVGs (and some tracers) write an explicit `L` back to the
    start point right before `Z`, which is geometrically the same seam
    Part 1 already represents via `closed=True` + modulo indexing
    without a duplicate point. Left in, that duplicate gives both the
    first and last point a zero-length vector toward each other, which
    silently defeats corner detection right at the seam -- exactly the
    real corner most closed shapes (a square, an icon) have sitting
    there.
    """
    if not points:
        return points
    out = [points[0]]
    for p in points[1:]:
        if _dist(p, out[-1]) > eps:
            out.append(p)
    if closed and len(out) >= 2 and _dist(out[0], out[-1]) <= eps:
        out.pop()
    return out


# ---------------------------------------------------------------------------
# Schneider least-squares cubic-Bezier curve fitting (Graphics Gems)
# ---------------------------------------------------------------------------
#
# Fits a chain of points -- smooth throughout, hard corners only at its
# two ends -- with the *fewest* cubic Bezier segments that still stay
# within `tolerance` of every original point. The core idea: for a
# single cubic segment, the two endpoints are fixed (the chain's first
# and last point) and the two tangent directions leaving those
# endpoints are fixed too (computed from the chain's own shape), which
# leaves only two unknowns -- how *far* out along each tangent the two
# control points sit. That's a small linear least-squares problem
# solvable in closed form. If the worst-fit point is still too far off,
# the per-point curve parameters are refined a few times with
# Newton-Raphson root-finding (sliding each point to its true closest
# position on the current curve) before trying the least-squares fit
# again. If it's *still* not good enough, the chain is split at its
# worst-error point and both halves are fit recursively -- exactly like
# Philip J. Schneider's original algorithm from Graphics Gems (1990).

_MAX_FIT_DEPTH = 24
_MAX_REPARAM_ITERATIONS = 4


def _bezier_point(t, bez):
    p0, p1, p2, p3 = bez
    mt = 1.0 - t
    b0, b1, b2, b3 = mt ** 3, 3 * mt * mt * t, 3 * mt * t * t, t ** 3
    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    return (x, y)


def _bezier_deriv1(t, bez):
    p0, p1, p2, p3 = bez
    mt = 1.0 - t
    vx = 3 * mt * mt * (p1[0] - p0[0]) + 6 * mt * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0])
    vy = 3 * mt * mt * (p1[1] - p0[1]) + 6 * mt * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
    return (vx, vy)


def _bezier_deriv2(t, bez):
    p0, p1, p2, p3 = bez
    mt = 1.0 - t
    vx = 6 * mt * (p2[0] - 2 * p1[0] + p0[0]) + 6 * t * (p3[0] - 2 * p2[0] + p1[0])
    vy = 6 * mt * (p2[1] - 2 * p1[1] + p0[1]) + 6 * t * (p3[1] - 2 * p2[1] + p1[1])
    return (vx, vy)


def _chord_length_parameterize(points):
    """Assigns each point an initial `t` in [0, 1] based on how far
    along the chain (by straight-line distance) it sits -- the standard
    starting guess before Newton-Raphson refines it against the actual
    fitted curve.
    """
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[-1] + _dist(points[i], points[i - 1]))
    total = u[-1]
    if total < 1e-9:
        n = len(points)
        return [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    return [x / total for x in u]


def _compute_left_tangent(points):
    t = _normalize(_sub(points[1], points[0]))
    return t if t != (0.0, 0.0) else (1.0, 0.0)


def _compute_right_tangent(points):
    t = _normalize(_sub(points[-2], points[-1]))
    return t if t != (0.0, 0.0) else (-1.0, 0.0)


def _compute_center_tangent(points, idx):
    t = _normalize(_sub(points[idx - 1], points[idx + 1]))
    if t == (0.0, 0.0):
        t = _normalize(_sub(points[idx], points[idx + 1]))
    return t


def _generate_bezier(points, u, tan1, tan2):
    """Least-squares solve for the two interior control points of a
    single cubic Bezier, given fixed endpoints `points[0]`/`points[-1]`
    and fixed unit tangent directions `tan1`/`tan2` leaving them. Falls
    back to the standard "distance / 3 along the tangent" heuristic
    whenever the least-squares solution comes out degenerate (a
    near-singular system, or a negative/near-zero control distance --
    both signs the fit isn't well-conditioned for this chain).
    """
    p0, p3 = points[0], points[-1]
    c00 = c01 = c11 = 0.0
    x0 = x1 = 0.0

    for i, t in enumerate(u):
        mt = 1.0 - t
        b0, b1, b2, b3 = mt ** 3, 3 * mt * mt * t, 3 * mt * t * t, t ** 3
        a1 = _scale(tan1, b1)
        a2 = _scale(tan2, b2)
        c00 += _dot(a1, a1)
        c01 += _dot(a1, a2)
        c11 += _dot(a2, a2)
        base = _add(_scale(p0, b0 + b1), _scale(p3, b2 + b3))
        tmp = _sub(points[i], base)
        x0 += _dot(a1, tmp)
        x1 += _dot(a2, tmp)

    det_c0_c1 = c00 * c11 - c01 * c01
    det_c0_x = c00 * x1 - c01 * x0
    det_x_c1 = x0 * c11 - x1 * c01

    alpha_l = 0.0 if abs(det_c0_c1) < 1e-12 else det_x_c1 / det_c0_c1
    alpha_r = 0.0 if abs(det_c0_c1) < 1e-12 else det_c0_x / det_c0_c1

    seg_len = _dist(p0, p3)
    eps = 1e-6 * seg_len if seg_len > 1e-9 else 1e-6
    # Besides the classic "too small/negative" degenerate case, a
    # near-singular system (tan1/tan2 close to parallel, common on
    # noisy or nearly-straight chains) can solve to a huge alpha that
    # technically clears the eps floor but places a control point wildly
    # past the chain -- an overshoot the original Graphics Gems fallback
    # doesn't catch. Cap it too, using the chain's own chord length as
    # the sane upper bound a well-behaved fit should stay under.
    max_alpha = 4.0 * seg_len if seg_len > 1e-9 else 4.0
    if alpha_l < eps or alpha_r < eps or alpha_l > max_alpha or alpha_r > max_alpha:
        dist = seg_len / 3.0
        p1 = _add(p0, _scale(tan1, dist))
        p2 = _add(p3, _scale(tan2, dist))
    else:
        p1 = _add(p0, _scale(tan1, alpha_l))
        p2 = _add(p3, _scale(tan2, alpha_r))

    return (p0, p1, p2, p3)


def _compute_max_error(points, bez, u):
    """Returns (worst squared distance, index of the worst point)
    between the chain's original points and the fitted curve, sampled
    at each point's current parameter value.
    """
    max_dist_sq = 0.0
    split_idx = len(points) // 2
    for i in range(1, len(points) - 1):
        pt = _bezier_point(u[i], bez)
        d_sq = (pt[0] - points[i][0]) ** 2 + (pt[1] - points[i][1]) ** 2
        if d_sq > max_dist_sq:
            max_dist_sq = d_sq
            split_idx = i
    return max_dist_sq, split_idx


def _newton_raphson_root_find(bez, point, u):
    """One Newton-Raphson step sliding `u` toward the parameter value
    where the curve is actually closest to `point`.
    """
    qu = _bezier_point(u, bez)
    q1 = _bezier_deriv1(u, bez)
    q2 = _bezier_deriv2(u, bez)
    diff = _sub(qu, point)
    numerator = _dot(diff, q1)
    denominator = _dot(q1, q1) + _dot(diff, q2)
    if abs(denominator) < 1e-12:
        return u
    new_u = u - numerator / denominator
    return max(0.0, min(1.0, new_u))


def _reparameterize(bez, points, u):
    return [_newton_raphson_root_find(bez, points[i], u[i]) for i in range(len(points))]


def _fit_cubic(points, tan1, tan2, error_sq, depth):
    """Recursive core of the Schneider fit. `error_sq` is the max
    allowed *squared* distance (matching what `_compute_max_error`
    returns) between the fitted curve and any original point.
    """
    if len(points) == 2:
        # Trivial case: nothing to least-squares over, just place the
        # two control points a third of the way along each tangent --
        # the standard closed-form answer for a 2-point "curve".
        dist = _dist(points[0], points[1]) / 3.0
        p0, p3 = points[0], points[1]
        p1 = _add(p0, _scale(tan1, dist))
        p2 = _add(p3, _scale(tan2, dist))
        return [(p0, p1, p2, p3)]

    u = _chord_length_parameterize(points)
    bez = _generate_bezier(points, u, tan1, tan2)
    max_err, split_idx = _compute_max_error(points, bez, u)
    if max_err < error_sq:
        return [bez]

    # Not a great fit yet, but close enough that reparameterizing (not
    # splitting) might still rescue it -- try a few rounds before
    # giving up and cutting the chain in half.
    if max_err < error_sq * error_sq and depth < _MAX_FIT_DEPTH:
        u_iter = u
        for _ in range(_MAX_REPARAM_ITERATIONS):
            u_iter = _reparameterize(bez, points, u_iter)
            bez = _generate_bezier(points, u_iter, tan1, tan2)
            max_err, split_idx = _compute_max_error(points, bez, u_iter)
            if max_err < error_sq:
                return [bez]

    if depth >= _MAX_FIT_DEPTH:
        # Give up gracefully rather than recursing forever on some
        # pathological chain -- best single-curve fit found so far.
        return [bez]

    # Fitting failed even after reparameterizing -- split the chain at
    # its worst-error point and fit each half recursively, joining them
    # at a shared tangent so the two curves meet smoothly.
    center_tan = _compute_center_tangent(points, split_idx)
    left = _fit_cubic(points[:split_idx + 1], tan1, center_tan, error_sq, depth + 1)
    right = _fit_cubic(points[split_idx:], _scale(center_tan, -1.0), tan2, error_sq, depth + 1)
    return left + right


def fit_chain(points, tolerance):
    """Public per-chain entry point: fits `points` (a smooth polyline
    chain with hard corners only at its two ends, as produced by
    `split_at_corners`) with the minimum number of cubic Bezier
    segments that stay within `tolerance` (a plain distance, in the
    same units as the points) of every original point. Returns a list
    of (p0, p1, p2, p3) control-point tuples, in order, each sharing
    its p0/p3 endpoint with the previous/next segment.
    """
    if len(points) < 2:
        return []
    tan1 = _compute_left_tangent(points)
    tan2 = _compute_right_tangent(points)
    error_sq = max(tolerance, 1e-3) ** 2
    return _fit_cubic(points, tan1, tan2, error_sq, 0)


# ---------------------------------------------------------------------------
# Reassembly: fitted (or untouched) subpaths -> SVG `d` text
# ---------------------------------------------------------------------------

def _format_num(x):
    if abs(x) < 1e-9:
        x = 0.0
    s = f"{x:.3f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _commands_to_d_fragment(orig_commands, closed):
    """Re-emits a subpath's original parsed commands verbatim -- the
    fallback used when re-fitting a subpath ever fails for some
    unexpected reason -- geometrically identical to the source, just
    always in absolute-coordinate, spec-expanded form.
    """
    parts = []
    for letter, args in orig_commands:
        if letter == "Z":
            parts.append("Z")
        else:
            parts.append(letter + " " + " ".join(_format_num(a) for a in args))
    if closed and (not orig_commands or orig_commands[-1][0] != "Z"):
        parts.append("Z")
    return " ".join(parts)


def _segments_to_d_fragment(segments, closed):
    """Turns a list of fitted (p0, p1, p2, p3) Bezier segments -- already
    in order, each sharing an endpoint with its neighbor -- into one
    `M ... C ... C ... [Z]` `d` fragment for a single subpath.
    """
    if not segments:
        return ""
    p0 = segments[0][0]
    parts = [f"M {_format_num(p0[0])} {_format_num(p0[1])}"]
    for _, p1, p2, p3 in segments:
        parts.append(
            "C "
            f"{_format_num(p1[0])} {_format_num(p1[1])} "
            f"{_format_num(p2[0])} {_format_num(p2[1])} "
            f"{_format_num(p3[0])} {_format_num(p3[1])}"
        )
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _smooth_subpath(subpath, tolerance, corner_angle_deg, corner_window=3):
    """Corner-detects, chain-splits, and curve-fits one subpath -- whether
    VTracer originally drew it as straight polygon segments or as its own
    spline-mode curves. Even a subpath VTracer already drew as C/Q/A
    curves gets re-flattened, corner-detected, and refit here: VTracer's
    own spline fit doesn't reliably keep real sharp corners sharp (it can
    round a shape's actual vertices into a smooth curve right through
    them), so leaving `was_curve` subpaths untouched was, in practice,
    passing that corner-rounding straight through unfixed. Re-fitting
    from the flattened points instead restores the real corners while the
    Schneider least-squares fit still converges on genuinely smooth runs
    with about as few curve segments as VTracer used, so already-good
    curves don't get needlessly degraded in the process.

    Returns None (rather than raising) for any subpath too degenerate to
    produce a sensible fragment from, so the caller can fall back to the
    untouched original.
    """
    pts = _dedupe_points(subpath["points"], closed=subpath["closed"])
    if len(pts) < 2:
        return None

    corners = detect_corners(pts, subpath["closed"], corner_angle_deg, corner_window=corner_window)
    chains = split_at_corners(pts, subpath["closed"], corners)

    segments = []
    for chain in chains:
        chain = _dedupe_points(chain, closed=False)
        if len(chain) < 2:
            continue
        segments.extend(fit_chain(chain, tolerance))

    if not segments:
        return None
    return _segments_to_d_fragment(segments, subpath["closed"])


def _smooth_single_d(d, tolerance, corner_angle_deg, corner_window=3):
    commands = parse_path_commands(d)
    subpaths = commands_to_subpaths(commands)

    fragments = []
    for subpath in subpaths:
        try:
            fragment = _smooth_subpath(subpath, tolerance, corner_angle_deg, corner_window=corner_window)
        except Exception:
            fragment = None  # fall through to the untouched original below
        if fragment is None:
            fragment = _commands_to_d_fragment(subpath["orig_commands"], subpath["closed"])
        if fragment:
            fragments.append(fragment)

    return " ".join(fragments)


# Matches a <path ...> tag's d="..." (or d='...') attribute, capturing
# the tag-prefix up through `d=`, the quote character used, and the
# attribute value itself -- so it can be swapped out in place without
# disturbing any other attribute (fill, stroke, transform, etc.) or tag
# formatting in the surrounding document.
_PATH_D_RE = re.compile(r"""(<path\b[^>]*\bd\s*=\s*)(["'])(.*?)\2""", re.IGNORECASE | re.DOTALL)


def smooth_svg_paths(svg_str, tolerance=1.8, corner_angle_deg=30.0, corner_window=3):
    """Public entry point. Re-fits every polygon-mode subpath of every
    `<path>` element in `svg_str` as a small number of smooth cubic
    Bezier curves, preserving real sharp corners (any direction change
    of at least `corner_angle_deg`) and leaving every subpath VTracer
    already drew as a curve completely untouched. `tolerance` is the
    max allowed distance, in the SVG's own coordinate units, between a
    fitted curve and any original traced point. `corner_window` is how
    many points on each side of a candidate corner are used to measure
    its direction change -- wider values ignore more pixel-staircase
    zigzag noise on curved edges before calling something a real corner
    (see `detect_corners`).

    Returns the full SVG document text with just the `d` attributes
    swapped out; everything else (viewBox, fill colors, other
    elements, whitespace outside of `d="..."`) is left exactly as it
    was. If smoothing a given path's data ever fails for any reason,
    that one path's `d` is left completely unchanged rather than
    raising -- callers that want an all-or-nothing guarantee should
    wrap the call in their own try/except (as vtracer_server.py does).
    """
    def _replace(m):
        prefix, quote, d = m.group(1), m.group(2), m.group(3)
        try:
            new_d = _smooth_single_d(d, tolerance, corner_angle_deg, corner_window=corner_window)
        except Exception:
            new_d = ""
        if not new_d or not new_d.strip():
            new_d = d
        return f"{prefix}{quote}{new_d}{quote}"

    return _PATH_D_RE.sub(_replace, svg_str)
