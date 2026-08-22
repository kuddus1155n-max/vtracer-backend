"""
VTracer Backend Server
-----------------------
A tiny local server that powers the "AI Vectorizer" section of Nur Meta AI.
It uses the real, open-source VTracer engine (https://github.com/visioncortex/vtracer)
to convert an uploaded JPG/PNG into clean SVG paths.

Before tracing, images go through a shape-detection preprocessing pass
(OpenCV) that finds the major shapes/edges and flattens away pixel-level
noise, JPEG artifacts, and gradient banding — so VTracer traces a handful
of clean regions instead of tracing every noisy pixel.

After tracing, the SVG goes through automatic path cleanup (see
path_cleanup.py): exact and near-exact duplicate paths are removed, tiny
"confetti" objects below an area threshold are dropped, broken/degenerate
paths (zero-length subpaths, stray single points) are discarded, redundant
collinear anchor points are collapsed, and same-colored paths that are
fully covered by another same-colored path (so removing them changes
nothing visible) are removed too.

Then every path's SVG is re-fit (see path_smoothing.py) so line-only
regions get smooth cubic-Bezier curves with far fewer anchor points, while
sharp corners are detected and kept crisp. Paths VTracer already drew as
curves are left untouched, since its own fit is already optimal.

An optional "Microstock Clean" mode (form field `microstock_clean`) layers
stronger versions of that same cleanup/smoothing on top -- tuned for
submitting to stock marketplaces (Adobe Stock, Shutterstock, Freepik,
etc.): more aggressive tiny-object removal, more aggressive duplicate/
near-duplicate shape removal, and more aggressive anchor-point reduction.
It never rasterizes or outlines anything, so the result stays fully
editable native SVG paths either way. See MICROSTOCK_CLEAN below.

The default parameters throughout this pipeline (SIMPLIFY_PRESETS below,
and the numeric defaults in _parse_numeric_params) are tuned for the kind
of source images this tool is built around -- flat illustrations, icons,
silhouettes, and AI-generated artwork: gentler mean-shift color blending
so genuinely distinct flat colors don't merge or shift (accurate colors),
a higher stray-object/anchor-point cleanup floor than VTracer's raw output
so shapes come out clean with the minimum anchors they actually need, and
VTracer's native "spline" mode across nearly the whole Detail slider for
naturally smooth Bezier curves. This is all still fully adjustable from
the HTML tool's sliders and toggles (or by any other caller sending its
own form fields) -- these are just the starting points a fresh install
or an un-set field falls back to.

The resulting SVG can also be downloaded as EPS (see eps_export.py, and
the /export_eps route below) for tools/marketplaces that specifically
require .eps. The EPS is generated directly from the SVG's own path data
-- each path stays its own separate object -- so it opens as a fully
editable vector in Adobe Illustrator, matching the SVG exactly.

Setup (one time):
    pip install vtracer flask flask-cors opencv-python-headless numpy

Run:
    python vtracer_server.py

By default this listens on http://localhost:8787
The HTML tool's "Backend URL" field should point to http://localhost:8787/vectorize
"""

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import vtracer
import cv2
import numpy as np

from path_smoothing import smooth_svg_paths
from path_cleanup import cleanup_svg_paths
from background_removal import remove_background
from eps_export import svg_to_eps
from color_segmentation import (
    detect_svg_color_palette,
    group_svg_by_color,
    build_color_layers_zip,
)

app = Flask(__name__)
CORS(app)  # allow the HTML file (opened from disk or any origin) to call this server

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Presets for the shape-detection preprocessing pass. Tuned per "detail level"
# so "low detail" simplifies aggressively (fewest, cleanest shapes, drops
# fine detail) and "high detail" simplifies gently (keeps small real detail
# like thin lines/dots, still removes pixel noise).
#   sp/sr          -> mean-shift segmentation spatial/color radius (merges
#                     pixel noise into flat regions while respecting edges)
#   k              -> number of colors kept after k-means quantization
#   min_area_frac  -> connected regions smaller than this fraction of the
#                     image area are dropped (merged into their biggest
#                     neighbor) as "unnecessary detail"
#   median         -> odd kernel size for median filtering the label map
#
# Tuned for flat illustrations, icons, silhouettes, and AI-generated
# artwork: sr (the mean-shift color radius) is lower than before at every
# level, since flat/AI art already has clean, deliberately-distinct flat
# colors that heavy color-blending would merge or shift -- bad for color
# accuracy. k (colors kept) is raised so more of those distinct colors
# survive quantization. min_area_frac is lower so small but real icon
# details (a dot, an eye, a thin accent) survive instead of being merged
# away; genuine stray noise/specks are still handled by filter_speckle and
# path cleanup further down the pipeline.
SIMPLIFY_PRESETS = {
    "low":    dict(sp=20, sr=45, k=8,  min_area_frac=0.005,  median=7),
    "medium": dict(sp=12, sr=30, k=12, min_area_frac=0.0018, median=5),
    "high":   dict(sp=6,  sr=20, k=16, min_area_frac=0.0008, median=3),
}

# "Microstock Clean" mode: a single toggle that layers stronger, stock-
# marketplace-tuned cleanup on top of whatever cleanup/smoothing values are
# already in effect (via max()/min() in run_vectorize_pipeline, so a user's
# own stricter setting is never loosened). Tuned for the kind of flat,
# node-light, duplicate-free SVGs sites like Adobe Stock/Shutterstock/
# Freepik expect from vector submissions: fewer stray "confetti" objects,
# fewer redundant/near-duplicate shapes, and fewer anchor points overall --
# while the output stays plain, native <path> elements the whole way
# through, so it's exactly as editable in Illustrator/Inkscape/etc as any
# other mode.
MICROSTOCK_CLEAN = dict(
    min_area_frac=0.00025,       # drop stray "confetti" specks (~6x the normal floor)
    node_epsilon=1.0,            # collapse near-collinear anchor points more aggressively
    overlap_containment=0.94,    # drop more same-colour hidden/redundant shapes
    dup_containment=0.97,        # catch near-duplicate shapes, not just near-pixel-identical ones
    min_smooth_tolerance=2.0,    # floor on path-simplification tolerance so node count always drops
)

# "Accuracy Mode": a single toggle that biases every stage of the pipeline
# toward preserving the source artwork's real shapes, proportions, angles,
# and geometry as closely as possible, instead of toward a lighter/cleaner
# file. It's the opposite dial from MICROSTOCK_CLEAN, and is layered on top
# of whatever the other fields already request via min()/max() in
# run_vectorize_pipeline (never loosening a caller's own stricter setting).
#
# Direction of every knob below is deliberate:
#   - VTracer's own corner/length/splice thresholds are lowered, so the
#     tracer itself is more sensitive to real corners and short real
#     segments instead of smoothing over them during tracing.
#   - max_iterations is raised for a closer curve fit to the source pixels.
#   - color_precision is raised and layer_difference lowered so distinct
#     colors/regions don't get merged into each other (a common source of
#     shape distortion).
#   - Post-trace cleanup thresholds are tightened (smaller min_area_frac,
#     smaller node_epsilon, higher containment ratios) so real small
#     shapes, real anchor points, and real distinct/overlapping shapes are
#     never mistaken for noise/redundancy and dropped.
#   - Smoothing tolerance is lowered and corner_angle_deg is lowered, so
#     the post-trace re-fit hugs the traced points closely and treats more
#     genuine direction changes as corners to keep crisp rather than
#     rounding them into a curve.
#   - simplify_shape_factor scales down the shape-preprocessing pass itself
#     (mean-shift radius, k-means merge floor, median blur) below even the
#     "high" detail preset, so shape-detection preprocessing flattens less
#     of the real geometry before VTracer ever sees the image.
ACCURACY_MODE = dict(
    corner_threshold=35,          # more sensitive corner detection while tracing
    length_threshold=2.0,         # keep short real segments instead of dropping them
    splice_threshold=25,          # tighter, more geometrically precise joins
    min_max_iterations=14,        # floor on curve-fit iterations (closer fit)
    min_color_precision=8,        # ceiling isn't relevant here -- always raise to max
    max_layer_difference=8,       # fewer merged layers, more distinct region boundaries
    max_cleanup_min_area_frac=0.00002,   # never drop small-but-real shapes as noise
    max_cleanup_node_epsilon=0.3,        # collapse far fewer near-collinear points
    min_cleanup_overlap_containment=0.995,  # only remove truly, fully hidden overlaps
    min_cleanup_dup_containment=0.999,      # only remove near-pixel-identical duplicates
    max_smooth_tolerance=0.6,     # hug the traced points closely when re-fitting
    max_smooth_corner_angle=16.0, # treat more real direction changes as corners
    simplify_shape_factor=0.55,   # shrink mean-shift/merge-floor/blur below the "high" preset
)


def _nearest_odd(x):
    """Rounds x to the nearest odd integer (cv2.medianBlur requires an odd
    kernel size), never going below 1."""
    n = 2 * round((x - 1) / 2) + 1
    return max(1, int(n))


def _interpolate_simplify_preset(detail):
    """Interpolates the low/medium/high SIMPLIFY_PRESETS for a continuous
    Detail-slider value (0 = Low ... 100 = High), using medium (at 50) as
    the midpoint anchor -- so slider positions of exactly 0/50/100 match
    the original discrete low/medium/high presets exactly, and everything
    in between blends smoothly."""
    detail = max(0.0, min(100.0, detail))
    low, med, high = SIMPLIFY_PRESETS["low"], SIMPLIFY_PRESETS["medium"], SIMPLIFY_PRESETS["high"]
    if detail <= 50:
        a, b, t = low, med, detail / 50.0
    else:
        a, b, t = med, high, (detail - 50) / 50.0

    def lerp(x, y):
        return x + (y - x) * t

    return dict(
        sp=lerp(a["sp"], b["sp"]),
        sr=lerp(a["sr"], b["sr"]),
        k=int(round(lerp(a["k"], b["k"]))),
        min_area_frac=lerp(a["min_area_frac"], b["min_area_frac"]),
        median=_nearest_odd(lerp(a["median"], b["median"])),
    )


def simplify_image(img_bytes, level="medium", color_count=None, detail=None, shape_factor=1.0, reconstruct_shapes=True):
    """Detect major shapes/edges and strip unnecessary pixel-level detail
    before tracing, so VTracer traces clean flat shapes instead of noise.

    1. Mean-shift segmentation merges noisy pixels into flat regions while
       preserving real shape boundaries/edges.
    2. K-means quantization reduces the image to a small palette, producing
       a discrete label map (not blended colors) so later steps never
       invent new colors at region boundaries.
    3. Median filtering on the label map cleans speckle at boundaries by
       picking an existing neighboring label (never blends colors).
    4. Small connected regions (below an area threshold) are merged into
       whichever neighboring region borders them most, so only the
       important major shapes survive.
    5. Shape-boundary reconstruction: each surviving region is recognized
       as one or more concrete shapes (via contour extraction) and redrawn
       from a minimal polygon approximation of its own boundary, instead of
       being left as the ragged pixel-staircase edge that quantization
       alone produces. This is what lets VTracer trace the artwork's actual
       recovered shapes instead of tracing individual pixels.

    color_count, when given, overrides the preset's k (number of colors
    kept by the k-means quantization step) with a user-chosen value.

    detail, when given (0-100), continuously interpolates between the
    low/medium/high presets for the Detail slider instead of using the
    fixed named `level` preset.

    shape_factor (0-1, default 1.0 = unchanged) scales the mean-shift
    spatial/color radius, the small-region merge floor, and the shape-
    reconstruction polygon tolerance down below whatever the level/detail
    preset already picked -- used by Accuracy Mode to flatten less real
    geometry during preprocessing than even the "high" detail preset does
    on its own. 1.0 leaves the preset untouched.

    reconstruct_shapes (default True) toggles step 5 above; turning it off
    falls back to tracing the quantized label map's raw pixel edges
    directly, exactly like before this stage existed.
    """
    if detail is not None:
        preset = _interpolate_simplify_preset(detail)
    else:
        preset = dict(SIMPLIFY_PRESETS.get(level, SIMPLIFY_PRESETS["medium"]))
    if color_count is not None:
        preset["k"] = color_count
    if shape_factor != 1.0:
        preset["sp"] = max(1, preset["sp"] * shape_factor)
        preset["sr"] = max(1, preset["sr"] * shape_factor)
        preset["min_area_frac"] = preset["min_area_frac"] * shape_factor
        preset["median"] = _nearest_odd(preset["median"] * shape_factor)

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image for preprocessing.")

    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if has_alpha:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
    elif img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img

    h, w = bgr.shape[:2]

    seg = cv2.pyrMeanShiftFiltering(bgr, sp=preset["sp"], sr=preset["sr"])

    pixels = seg.reshape((-1, 3)).astype(np.float32)
    k = preset["k"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.3)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS
    )
    label_map = labels.reshape(h, w).astype(np.uint8)  # k is always <= 16

    label_map = cv2.medianBlur(label_map, preset["median"])

    min_area = max(8, int(h * w * preset["min_area_frac"]))
    label_map = _remove_small_regions(label_map.astype(np.int32), k, min_area)

    if reconstruct_shapes:
        label_map = _reconstruct_shape_boundaries(label_map, k, shape_factor=shape_factor)

    centers_u8 = np.uint8(centers)
    out_bgr = centers_u8[label_map.flatten()].reshape(h, w, 3)

    if has_alpha:
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2BGRA)
        out[:, :, 3] = alpha
    else:
        out = out_bgr

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise ValueError("Could not re-encode preprocessed image.")
    return buf.tobytes()


def _remove_small_regions(label_map, k, min_area):
    """Merge every connected component smaller than min_area into whichever
    neighboring region borders it most. This drops unnecessary small
    detail/noise while preserving the important major shapes."""
    result = label_map.copy()
    for lbl in range(k):
        mask = (label_map == lbl).astype(np.uint8)
        if not mask.any():
            continue
        n, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp_id in range(1, n):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area >= min_area:
                continue
            comp_mask = (comp_labels == comp_id).astype(np.uint8)
            dilated = cv2.dilate(comp_mask, np.ones((3, 3), np.uint8), iterations=2)
            border = (dilated > 0) & (comp_mask == 0)
            neighbor_vals = result[border]
            if neighbor_vals.size == 0:
                continue
            vals, counts = np.unique(neighbor_vals, return_counts=True)
            new_label = vals[np.argmax(counts)]
            result[comp_mask.astype(bool)] = new_label
    return result


def _reconstruct_shape_boundaries(label_map, k, shape_factor=1.0, approx_epsilon_frac=0.0015, min_contour_area=6):
    """Recognizes each color region as one or more concrete shapes (via
    contour extraction) and redraws each one from a minimal polygon
    approximation of its own boundary, instead of leaving the ragged
    pixel-staircase edges that quantization alone produces. This is what
    lets VTracer trace the artwork's actual recovered shape boundaries
    afterward, rather than blindly tracing individual pixels.

    For each label: cv2.findContours recovers every outer shape and hole
    that region actually has; cv2.approxPolyDP then reduces each contour to
    the minimal set of vertices needed to represent it within `epsilon`
    pixels -- i.e. it recognizes the shape (rectangle, blob, star, hole,
    whatever it actually is) rather than assuming any particular form.
    Holes are cut back out after every outer shape for that label is
    filled, so a hole nested inside another hole's outer ring still
    resolves correctly.

    Regions are redrawn back onto a fresh label map in ascending label
    order (painter's algorithm): a later region's reconstructed boundary
    wins over an earlier one wherever two neighboring shapes' approximated
    polygons disagree at a shared edge. Any pixel no region's
    reconstruction claims (rare -- only right at such disagreements) simply
    keeps its original, pixel-accurate label, so reconstruction only ever
    replaces real boundary detail and never leaves an undefined gap.

    shape_factor (0-1, default 1.0) scales epsilon down along with the rest
    of Accuracy Mode's preprocessing tightening -- a smaller epsilon keeps
    more vertices, hugging the source boundary more closely.
    """
    h, w = label_map.shape
    diag = float(np.hypot(h, w))
    epsilon = max(0.5, diag * approx_epsilon_frac * shape_factor)
    out = label_map.copy()
    for lbl in range(k):
        mask = ((label_map == lbl).astype(np.uint8)) * 255
        if not mask.any():
            continue
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hier = hierarchy[0] if hierarchy is not None else []
        canvas = np.zeros((h, w), dtype=np.uint8)
        # pass 1: fill every recognized outer shape for this label
        for i, cnt in enumerate(contours):
            is_hole = len(hier) and hier[i][3] != -1
            if is_hole or cv2.contourArea(cnt) < min_contour_area:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(canvas, [approx], -1, 255, thickness=cv2.FILLED)
        # pass 2: cut every recognized hole back out, after all outer shapes
        # for this label are already filled
        for i, cnt in enumerate(contours):
            is_hole = len(hier) and hier[i][3] != -1
            if not is_hole or cv2.contourArea(cnt) < min_contour_area:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(canvas, [approx], -1, 0, thickness=cv2.FILLED)
        out[canvas > 0] = lbl
    return out


def guess_img_format(filename, content_type):
    name = (filename or "").lower()
    if name.endswith(".png"):
        return "png"
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "jpg"
    if content_type == "image/png":
        return "png"
    if content_type in ("image/jpeg", "image/jpg"):
        return "jpg"
    return "png"  # sensible default


class PipelineError(Exception):
    """Carries both a message and the HTTP status it should surface as,
    so every route that runs the pipeline reports errors the same way
    /vectorize always has."""
    def __init__(self, message, status=500):
        super().__init__(message)
        self.status = status


def _parse_numeric_params(form):
    try:
        return dict(
            # Baselines below match the frontend's Detail-slider "medium"
            # anchor (see detailToVtracerParams in the HTML tool) -- tuned
            # for flat illustrations/icons/silhouettes/AI art: more accurate
            # color reproduction (color_precision), fewer stray layers from
            # soft AI-art edges (layer_difference), cleaner shapes
            # (filter_speckle). Only used as a fallback when a caller
            # doesn't send these fields explicitly (the bundled HTML tool
            # always does).
            filter_speckle=int(float(form.get("filter_speckle", 6))),
            color_precision=int(float(form.get("color_precision", 7))),
            layer_difference=int(float(form.get("layer_difference", 20))),
            corner_threshold=int(float(form.get("corner_threshold", 60))),
            length_threshold=float(form.get("length_threshold", 4.0)),
            max_iterations=int(float(form.get("max_iterations", 10))),
            splice_threshold=int(float(form.get("splice_threshold", 45))),
            path_precision=int(float(form.get("path_precision", 8))),
            smooth_tolerance=float(form.get("smooth_tolerance", 1.8)),
            smooth_corner_angle=float(form.get("smooth_corner_angle", 30.0)),
            # Default anchor-point/stray-object cleanup raised a bit from
            # VTracer's raw-output level, so flat-color icon/illustration
            # shapes come out clean with the minimum anchors they actually
            # need -- but kept well below the MICROSTOCK_CLEAN floor below,
            # so that toggle still means something stronger on top of this.
            cleanup_min_area_frac=float(form.get("cleanup_min_area_frac", 0.00008)),
            cleanup_node_epsilon=float(form.get("cleanup_node_epsilon", 0.75)),
            cleanup_overlap_containment=float(form.get("cleanup_overlap_containment", 0.98)),
            cleanup_dup_containment=float(form.get("cleanup_dup_containment", 0.995)),
        )
    except ValueError:
        raise PipelineError("One or more numeric parameters were invalid.", 400)


def _parse_color_count(form):
    """Reads the optional 'color_count' form field: an approximate target
    palette size (2-32 colors) for the result. Returns None if not given."""
    raw = form.get("color_count", "").strip()
    if not raw:
        return None
    try:
        color_count = int(float(raw))
    except ValueError:
        raise PipelineError("color_count must be a number.", 400)
    return max(2, min(32, color_count))


def _parse_detail(form):
    """Reads the optional 'detail' form field: a continuous Detail-slider
    value from 0 (Low) to 100 (High). Returns None if not given, so
    callers can fall back to the discrete `simplify_level` preset."""
    raw = form.get("detail", "").strip()
    if not raw:
        return None
    try:
        d = float(raw)
    except ValueError:
        raise PipelineError("detail must be a number.", 400)
    return max(0.0, min(100.0, d))


def run_vectorize_pipeline(img_bytes, img_format, form):
    """Runs the full simplify -> VTracer -> cleanup -> smooth pipeline
    shared by /vectorize and /segment, and returns the final SVG string.
    Raises PipelineError (with the right HTTP status) on failure."""
    colormode = form.get("colormode", "color")
    hierarchical = form.get("hierarchical", "stacked")
    mode = form.get("mode", "spline")

    # "simplify" defaults ON: detect major shapes/edges and strip pixel-level
    # detail before tracing, instead of tracing every pixel.
    do_simplify = form.get("simplify", "true").lower() not in ("false", "0", "no")
    simplify_level = form.get("simplify_level", "medium")
    if simplify_level not in SIMPLIFY_PRESETS:
        simplify_level = "medium"

    # Continuous Detail slider (0 = Low ... 100 = High). When given, this
    # takes over from the discrete `simplify_level` above and interpolates
    # the low/medium/high shape-simplification presets smoothly; the
    # `simplify_level` field is kept only for backward compatibility with
    # callers that still send a named level instead of a slider value.
    detail = _parse_detail(form)

    # Optional explicit color-count control (approx. 2-32 colors). This is
    # the target palette size for the k-means quantization step, so it only
    # takes effect through the simplify pass -- turn simplify on if a color
    # count was requested even if the caller left it off/unset.
    color_count = _parse_color_count(form)
    if color_count is not None:
        do_simplify = True

    # Automatic background removal: detects a flat/uniform background
    # (white by default, or any solid color in "auto" mode) around the
    # edges of the image and makes it transparent before anything else
    # runs, so VTracer never traces a big background shape behind the
    # actual artwork. Defaults ON; skips itself automatically whenever the
    # border doesn't look like a flat background, so it's safe to leave on
    # for photos with no removable background too.
    do_remove_bg = form.get("remove_background", "true").lower() not in ("false", "0", "no")
    bg_mode = form.get("bg_mode", "auto")
    if bg_mode not in ("auto", "white"):
        bg_mode = "auto"
    try:
        bg_tolerance = float(form.get("bg_tolerance", 30))
    except ValueError:
        raise PipelineError("bg_tolerance must be a number.", 400)

    # Post-trace path smoothing/simplification: re-fits any line-only
    # subpaths (e.g. from "polygon" mode) as smooth cubic Beziers with fewer
    # anchor points, while leaving corners sharp. Subpaths VTracer already
    # drew as curves ("spline" mode) are left untouched -- its own fit is
    # already optimal, so re-deriving it would only risk losing accuracy.
    do_smooth_paths = form.get("smooth_paths", "true").lower() not in ("false", "0", "no")

    # Automatic SVG path cleanup: removes duplicate paths, tiny objects,
    # broken paths, redundant nodes, and unnecessary same-color overlaps
    # from VTracer's raw output, before any smoothing happens.
    do_cleanup_paths = form.get("cleanup_paths", "true").lower() not in ("false", "0", "no")

    # "Microstock Clean" mode: one toggle that guarantees a marketplace-
    # ready result. It forces simplify/cleanup/smoothing on -- a microstock
    # file can never skip any of those stages -- regardless of what the
    # individual toggles above were set to.
    do_microstock_clean = form.get("microstock_clean", "false").lower() not in ("false", "0", "no")
    if do_microstock_clean:
        do_simplify = True
        do_cleanup_paths = True
        do_smooth_paths = True

    # "Accuracy Mode": one toggle that biases the whole pipeline toward
    # preserving the source artwork's real shapes/proportions/angles as
    # closely as possible (see ACCURACY_MODE above for the full rationale).
    # Forces simplify/cleanup/smoothing on, same as microstock_clean, since
    # each stage still has a real, positive role to play (removing actual
    # sensor/JPEG/AI-artifact noise) -- Accuracy Mode makes each stage more
    # conservative rather than skipping stages outright.
    do_accuracy_mode = form.get("accuracy_mode", "false").lower() not in ("false", "0", "no")
    if do_accuracy_mode:
        do_simplify = True
        do_cleanup_paths = True
        do_smooth_paths = True

    params = _parse_numeric_params(form)

    if do_accuracy_mode:
        # Layer the stronger accuracy thresholds on top of whatever is
        # already in params -- min()/max() (picked per-field for the
        # "more accurate" direction) so a user's own already-stricter
        # value is never loosened back down.
        am = ACCURACY_MODE
        params["corner_threshold"] = min(params["corner_threshold"], am["corner_threshold"])
        params["length_threshold"] = min(params["length_threshold"], am["length_threshold"])
        params["splice_threshold"] = min(params["splice_threshold"], am["splice_threshold"])
        params["max_iterations"] = max(params["max_iterations"], am["min_max_iterations"])
        params["color_precision"] = max(params["color_precision"], am["min_color_precision"])
        params["layer_difference"] = min(params["layer_difference"], am["max_layer_difference"])
        params["cleanup_min_area_frac"] = min(params["cleanup_min_area_frac"], am["max_cleanup_min_area_frac"])
        params["cleanup_node_epsilon"] = min(params["cleanup_node_epsilon"], am["max_cleanup_node_epsilon"])
        params["cleanup_overlap_containment"] = max(params["cleanup_overlap_containment"], am["min_cleanup_overlap_containment"])
        params["cleanup_dup_containment"] = max(params["cleanup_dup_containment"], am["min_cleanup_dup_containment"])
        params["smooth_tolerance"] = min(params["smooth_tolerance"], am["max_smooth_tolerance"])
        params["smooth_corner_angle"] = min(params["smooth_corner_angle"], am["max_smooth_corner_angle"])

    if do_microstock_clean:
        # Layer the stronger microstock thresholds on top of whatever is
        # already in params -- max()/min() so a user's own stricter value
        # (e.g. a higher smoothness tolerance) is never weakened back down.
        mc = MICROSTOCK_CLEAN
        params["cleanup_min_area_frac"] = max(params["cleanup_min_area_frac"], mc["min_area_frac"])
        params["cleanup_node_epsilon"] = max(params["cleanup_node_epsilon"], mc["node_epsilon"])
        params["cleanup_overlap_containment"] = min(params["cleanup_overlap_containment"], mc["overlap_containment"])
        params["cleanup_dup_containment"] = min(params["cleanup_dup_containment"], mc["dup_containment"])
        params["smooth_tolerance"] = max(params["smooth_tolerance"], mc["min_smooth_tolerance"])

    if do_remove_bg:
        try:
            img_bytes, bg_applied, _bg_info = remove_background(
                img_bytes, mode=bg_mode, tolerance=bg_tolerance,
            )
            if bg_applied:
                img_format = "png"  # remove_background always re-encodes to PNG (needs alpha)
        except Exception as exc:
            raise PipelineError(f"Background removal failed: {exc}", 500)

    # Shape-boundary reconstruction: after simplify's quantization, each
    # region is recognized as one or more concrete shapes (via contour
    # extraction) and redrawn from a minimal polygon approximation of its
    # own boundary, instead of tracing the ragged pixel-staircase edges
    # quantization alone leaves behind. Defaults ON; only takes effect when
    # simplify itself runs, same as color_count above.
    do_reconstruct_shapes = form.get("reconstruct_shapes", "true").lower() not in ("false", "0", "no")

    if do_simplify:
        try:
            shape_factor = ACCURACY_MODE["simplify_shape_factor"] if do_accuracy_mode else 1.0
            img_bytes = simplify_image(
                img_bytes, level=simplify_level, color_count=color_count,
                detail=detail, shape_factor=shape_factor,
                reconstruct_shapes=do_reconstruct_shapes,
            )
            img_format = "png"  # simplify_image always re-encodes to PNG
            # Preprocessing already removed pixel-level noise, so relax the
            # speckle filter slightly further to avoid re-fragmenting the
            # clean flat regions it produced.
            params["filter_speckle"] = max(params["filter_speckle"], 4)
        except Exception as exc:
            raise PipelineError(f"Shape preprocessing failed: {exc}", 500)

    if color_count is not None:
        # The palette is already pinned to `color_count` distinct colors by
        # the quantization step above. Use max color precision so VTracer's
        # own color grouping doesn't merge any of those colors back
        # together while tracing -- the traced result should keep close to
        # the requested count.
        params["color_precision"] = 8

    try:
        svg_str = vtracer.convert_raw_image_to_svg(
            img_bytes,
            img_format=img_format,
            colormode=colormode,
            hierarchical=hierarchical,
            mode=mode,
            filter_speckle=params["filter_speckle"],
            color_precision=params["color_precision"],
            layer_difference=params["layer_difference"],
            corner_threshold=params["corner_threshold"],
            length_threshold=params["length_threshold"],
            max_iterations=params["max_iterations"],
            splice_threshold=params["splice_threshold"],
            path_precision=params["path_precision"],
        )
    except Exception as exc:  # surface VTracer/parse errors to the browser
        raise PipelineError(f"VTracer failed: {exc}", 500)

    if do_cleanup_paths:
        try:
            svg_str = cleanup_svg_paths(
                svg_str,
                min_area_frac=params["cleanup_min_area_frac"],
                node_epsilon=params["cleanup_node_epsilon"],
                overlap_containment=params["cleanup_overlap_containment"],
                dup_containment=params["cleanup_dup_containment"],
            )
        except Exception:
            pass  # if cleanup ever fails, fall back to VTracer's raw output

    if do_smooth_paths:
        try:
            svg_str = smooth_svg_paths(
                svg_str, tolerance=params["smooth_tolerance"],
                corner_angle_deg=params["smooth_corner_angle"],
            )
        except Exception:
            pass  # if smoothing ever fails, fall back to VTracer's raw output

    return svg_str


def _read_uploaded_image():
    """Shared request parsing for /vectorize and /segment. Returns
    (img_bytes, img_format) or raises PipelineError with a 400 status."""
    if "image" not in request.files:
        raise PipelineError("No image file provided (expected form field 'image').", 400)
    file = request.files["image"]
    img_bytes = file.read()
    if not img_bytes:
        raise PipelineError("Uploaded image is empty.", 400)
    return img_bytes, guess_img_format(file.filename, file.mimetype)


@app.route("/vectorize", methods=["POST"])
def vectorize():
    """Form fields accepted: see run_vectorize_pipeline(). Notably
    `microstock_clean` ("true"/"false") turns on "Microstock Clean" mode --
    stronger tiny-object/duplicate-shape/anchor-point cleanup, layered on
    top of whatever the other simplify/cleanup/smoothing fields already
    request, for a marketplace-ready result that's still fully editable."""
    try:
        img_bytes, img_format = _read_uploaded_image()
        svg_str = run_vectorize_pipeline(img_bytes, img_format, request.form)
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), exc.status

    return Response(svg_str, mimetype="image/svg+xml")


@app.route("/segment", methods=["POST"])
def segment():
    """Automatic color segmentation: runs the same vectorize pipeline as
    /vectorize, then detects the image's main colors and turns each one
    into its own addressable/exportable set of shapes.

    Form fields (in addition to everything /vectorize accepts):
      min_color_percent -- colors covering less than this % of the total
                            drawn area are treated as noise, not a "main
                            color" (default 0.5).
      format             -- "json" (default): returns
                               {"colors": [...], "svg": "<svg>...</svg>"}
                             where `svg` groups same-colored paths under
                             <g data-color="#rrggbb" ...> elements.
                             "zip": returns a .zip with one standalone .svg
                             per detected color instead.
    """
    try:
        img_bytes, img_format = _read_uploaded_image()
        svg_str = run_vectorize_pipeline(img_bytes, img_format, request.form)
    except PipelineError as exc:
        return jsonify({"error": str(exc)}), exc.status

    try:
        min_percent = float(request.form.get("min_color_percent", 0.5))
    except ValueError:
        return jsonify({"error": "min_color_percent must be a number."}), 400

    if request.form.get("format", "json").lower() == "zip":
        try:
            zip_bytes = build_color_layers_zip(svg_str, min_percent=min_percent)
        except Exception as exc:
            return jsonify({"error": f"Could not build color layers: {exc}"}), 500
        return Response(
            zip_bytes,
            mimetype="application/zip",
            headers={"Content-Disposition": "attachment; filename=color_layers.zip"},
        )

    try:
        colors = detect_svg_color_palette(svg_str, min_percent=min_percent)
        grouped_svg = group_svg_by_color(svg_str)
    except Exception as exc:
        return jsonify({"error": f"Color segmentation failed: {exc}"}), 500

    return jsonify({"colors": colors, "svg": grouped_svg})


@app.route("/export_eps", methods=["POST"])
def export_eps():
    """Converts an SVG document (the same one /vectorize or /segment just
    returned -- the browser sends it back, nothing is re-traced) into an
    EPS file, so the vector can be downloaded as SVG or EPS from the same
    result. See eps_export.py: every <path> stays its own separate,
    editable object in the .eps, and colors/shapes are read straight from
    the SVG's own path data, so it matches exactly and opens cleanly in
    Adobe Illustrator (or any other EPS-reading vector editor).

    Form fields:
      svg   -- required. The SVG document string to convert.
      title -- optional. EPS %%Title comment (default 'Vectorized Image').
    """
    svg_str = request.form.get("svg", "")
    if not svg_str.strip():
        return jsonify({"error": "No SVG provided (expected form field 'svg')."}), 400
    title = request.form.get("title", "Vectorized Image")
    try:
        eps_str = svg_to_eps(svg_str, title=title)
    except Exception as exc:
        return jsonify({"error": f"EPS conversion failed: {exc}"}), 500
    return Response(
        eps_str.encode("latin-1", errors="replace"),
        mimetype="application/postscript",
        headers={"Content-Disposition": "attachment; filename=vectorized.eps"},
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "vtracer"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8787))
    print(f"VTracer backend running at http://localhost:{port}")
    print(f"Point the AI Vectorizer's Backend URL to http://localhost:{port}/vectorize")
    app.run(host="0.0.0.0", port=port)
