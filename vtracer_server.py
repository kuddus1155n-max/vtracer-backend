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

# Force single-threaded OpenCV. On constrained containers (e.g. Render's
# free tier, which allocates a fraction of a CPU core), OpenCV's internal
# thread pool can segfault the whole process during heavy ops like
# pyrMeanShiftFiltering/kmeans. Single-threaded mode avoids that crash.
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from path_smoothing import smooth_svg_paths
from path_cleanup import cleanup_svg_paths

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
SIMPLIFY_PRESETS = {
    "low":    dict(sp=20, sr=60, k=6,  min_area_frac=0.006,  median=7),
    "medium": dict(sp=12, sr=40, k=10, min_area_frac=0.0025, median=5),
    "high":   dict(sp=6,  sr=25, k=16, min_area_frac=0.001,  median=3),
}


def simplify_image(img_bytes, level="medium"):
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
    """
    preset = SIMPLIFY_PRESETS.get(level, SIMPLIFY_PRESETS["medium"])

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image for preprocessing.")

    # Cap the working size before the heavy ops below. pyrMeanShiftFiltering
    # and kmeans scale badly with pixel count, and on a memory/CPU-constrained
    # container (e.g. Render's free tier) a large image can spike memory
    # enough to segfault the whole process. 1000px on the longest side keeps
    # results visually equivalent for tracing while staying safe.
    _MAX_DIM = 1000
    _h0, _w0 = img.shape[:2]
    if max(_h0, _w0) > _MAX_DIM:
        _scale = _MAX_DIM / float(max(_h0, _w0))
        img = cv2.resize(
            img, (max(1, int(_w0 * _scale)), max(1, int(_h0 * _scale))),
            interpolation=cv2.INTER_AREA,
        )

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


def _cap_image_size(img_bytes, max_dim=1000):
    """Downscale img_bytes so its longest side is at most max_dim, re-encoded
    as PNG. Used as a memory/CPU safety net on constrained hosting when the
    shape-simplify pass (which has its own cap) is skipped."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image.")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(
            img, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not re-encode image.")
    return buf.tobytes()


@app.route("/vectorize", methods=["POST"])
def vectorize():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (expected form field 'image')."}), 400

    file = request.files["image"]
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({"error": "Uploaded image is empty."}), 400

    img_format = guess_img_format(file.filename, file.mimetype)

    form = request.form
    colormode = form.get("colormode", "color")
    hierarchical = form.get("hierarchical", "stacked")
    mode = form.get("mode", "spline")

    # "simplify" defaults ON: detect major shapes/edges and strip pixel-level
    # detail before tracing, instead of tracing every pixel.
    do_simplify = form.get("simplify", "true").lower() not in ("false", "0", "no")
    simplify_level = form.get("simplify_level", "medium")
    if simplify_level not in SIMPLIFY_PRESETS:
        simplify_level = "medium"

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

    try:
        filter_speckle = int(float(form.get("filter_speckle", 4)))
        color_precision = int(float(form.get("color_precision", 6)))
        layer_difference = int(float(form.get("layer_difference", 16)))
        corner_threshold = int(float(form.get("corner_threshold", 60)))
        length_threshold = float(form.get("length_threshold", 4.0))
        max_iterations = int(float(form.get("max_iterations", 10)))
        splice_threshold = int(float(form.get("splice_threshold", 45)))
        path_precision = int(float(form.get("path_precision", 8)))
        smooth_tolerance = float(form.get("smooth_tolerance", 1.5))
        smooth_corner_angle = float(form.get("smooth_corner_angle", 30.0))
        cleanup_min_area_frac = float(form.get("cleanup_min_area_frac", 0.00004))
        cleanup_node_epsilon = float(form.get("cleanup_node_epsilon", 0.5))
        cleanup_overlap_containment = float(form.get("cleanup_overlap_containment", 0.98))
        cleanup_dup_containment = float(form.get("cleanup_dup_containment", 0.995))
    except ValueError:
        return jsonify({"error": "One or more numeric parameters were invalid."}), 400

    if do_simplify:
        try:
            img_bytes = simplify_image(img_bytes, level=simplify_level)
            img_format = "png"  # simplify_image always re-encodes to PNG
            # Preprocessing already removed pixel-level noise, so relax the
            # speckle filter slightly further to avoid re-fragmenting the
            # clean flat regions it produced.
            filter_speckle = max(filter_speckle, 4)
        except Exception as exc:
            return jsonify({"error": f"Shape preprocessing failed: {exc}"}), 500
    else:
        # No shape-preprocessing pass means no size cap has been applied yet.
        # Cap it here too, so a very large raw image can't spike memory/CPU
        # on constrained hosting and crash the process.
        try:
            img_bytes = _cap_image_size(img_bytes)
            img_format = "png"
        except Exception as exc:
            return jsonify({"error": f"Image preprocessing failed: {exc}"}), 500

    try:
        svg_str = vtracer.convert_raw_image_to_svg(
            img_bytes,
            img_format=img_format,
            colormode=colormode,
            hierarchical=hierarchical,
            mode=mode,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
            layer_difference=layer_difference,
            corner_threshold=corner_threshold,
            length_threshold=length_threshold,
            max_iterations=max_iterations,
            splice_threshold=splice_threshold,
            path_precision=path_precision,
        )
    except Exception as exc:  # surface VTracer/parse errors to the browser
        return jsonify({"error": f"VTracer failed: {exc}"}), 500

    if do_cleanup_paths:
        try:
            svg_str = cleanup_svg_paths(
                svg_str,
                min_area_frac=cleanup_min_area_frac,
                node_epsilon=cleanup_node_epsilon,
                overlap_containment=cleanup_overlap_containment,
                dup_containment=cleanup_dup_containment,
            )
        except Exception:
            pass  # if cleanup ever fails, fall back to VTracer's raw output

    if do_smooth_paths:
        try:
            svg_str = smooth_svg_paths(
                svg_str, tolerance=smooth_tolerance, corner_angle_deg=smooth_corner_angle
            )
        except Exception:
            pass  # if smoothing ever fails, fall back to VTracer's raw output

    return Response(svg_str, mimetype="image/svg+xml")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "vtracer"})


@app.route("/debug-vtracer", methods=["GET"])
def debug_vtracer():
    """Diagnostic-only route: runs VTracer on a tiny hardcoded 2x2 PNG,
    completely bypassing OpenCV/our preprocessing, to isolate whether
    crashes come from VTracer itself or from our OpenCV preprocessing code."""
    import base64
    # A trivial 2x2 red PNG, hardcoded as base64 so this route has zero
    # dependency on file upload or OpenCV image decoding.
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR42mP8z8"
        "BQz0AEYBxVSF+FABJADveWkH6oAAAAAElFTkSuQmCC"
    )
    img_bytes = base64.b64decode(tiny_png_b64)
    svg_str = vtracer.convert_raw_image_to_svg(
        img_bytes,
        img_format="png",
        colormode="color",
        hierarchical="stacked",
        mode="spline",
    )
    return jsonify({"status": "ok", "svg_length": len(svg_str)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8787))
    print(f"VTracer backend running at http://localhost:{port}")
    print(f"Point the AI Vectorizer's Backend URL to http://localhost:{port}/vectorize")
    app.run(host="0.0.0.0", port=port)
