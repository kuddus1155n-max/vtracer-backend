"""
Automatic background removal
-----------------------------
Detects a flat/uniform background (white, or any other solid color) around
the edges of an image and makes it transparent, while leaving the actual
artwork alone -- so VTracer traces just the subject instead of also tracing
a big background-colored shape behind it.

Approach (deliberately simple and safe, not a full salient-object-detection
model):

1. Sample a thin ring of pixels around the image border and take their
   median color as the background color guess ("auto" mode), or use a
   fixed near-white reference color ("white" mode).
2. Build a mask of pixels within `tolerance` color-distance of that
   reference color.
3. Keep only the parts of that mask that are connected (via 8-connectivity)
   to the image border -- so a white cat on a white background does NOT
   get its highlights erased, only the background itself, since the cat's
   white fur is not (usually) touching the frame.
4. Small morphological open/close cleans up JPEG-noise holes and speckle
   at the boundary.
5. The kept background mask becomes alpha=0; everything else keeps its
   original alpha (fully opaque, for a normal JPG/PNG input).

In "auto" mode, if the border ring isn't actually uniform (a busy photo
with no flat background), removal is skipped entirely rather than guessing
wrong and cutting into real artwork.
"""

import cv2
import numpy as np

# BGR reference used for "white" mode -- near-white rather than pure white
# so slightly off-white/cream backgrounds and JPEG compression noise near
# white still match.
WHITE_REF_BGR = np.array([246.0, 246.0, 246.0], dtype=np.float32)


def _border_ring_mask(shape, ring_px):
    """Boolean mask (h, w) that is True for the outer ring of pixels."""
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=bool)
    m[0:ring_px, :] = True
    m[h - ring_px:h, :] = True
    m[:, 0:ring_px] = True
    m[:, w - ring_px:w] = True
    return m


def _color_distance_mask(bgr, ref_color, tolerance):
    diff = bgr.astype(np.float32) - ref_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return dist <= tolerance


def _keep_mask_connected_to_border(mask):
    """Keeps only the connected components of `mask` that touch the image
    border, so background-colored regions fully enclosed inside the
    artwork (e.g. a white highlight) are left untouched."""
    mask_u8 = mask.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape
    border_labels = set(labels[0, :].tolist()) | set(labels[h - 1, :].tolist())
    border_labels |= set(labels[:, 0].tolist()) | set(labels[:, w - 1].tolist())
    border_labels.discard(0)  # 0 = background of the label map, not our mask
    if not border_labels:
        return np.zeros_like(mask)
    return np.isin(labels, list(border_labels))


def remove_background(img_bytes, mode="auto", tolerance=30.0, min_confidence=0.5):
    """Detects and removes a flat background, returning
    (out_png_bytes, applied, info).

    mode           -- "auto" (sample the border color) or "white" (force a
                       near-white reference color regardless of what the
                       border actually looks like).
    tolerance      -- color-distance threshold (0-100-ish scale on typical
                       0-255 channel values) for "close enough to the
                       background color".
    min_confidence -- ("auto" mode only) required fraction of the border
                       ring that must match the guessed background color
                       before removal proceeds; below this, the border is
                       assumed to not be a flat background and nothing is
                       changed, to avoid cutting into a busy photo.

    `applied` is False (and img_bytes returned unchanged) whenever removal
    was skipped or found nothing meaningful to remove.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image for background removal.")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        base_alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif img.shape[2] == 4:
        bgr = img[:, :, :3]
        base_alpha = img[:, :, 3]
        # Already has meaningful transparency -- assume the background was
        # handled upstream and leave it alone.
        if float((base_alpha < 250).mean()) > 0.02:
            return img_bytes, False, {"reason": "image already has transparency"}
    else:
        bgr = img
        base_alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)

    h, w = bgr.shape[:2]
    ring_px = max(2, round(0.01 * min(h, w)))
    ring = _border_ring_mask((h, w), ring_px)

    if mode == "white":
        ref_color = WHITE_REF_BGR
    else:
        ref_color = np.median(bgr[ring], axis=0)

    close_mask = _color_distance_mask(bgr, ref_color, tolerance)

    confidence = float(close_mask[ring].mean())
    if mode != "white" and confidence < min_confidence:
        return img_bytes, False, {
            "reason": "border is not a uniform background",
            "confidence": confidence,
        }

    bg_mask = _keep_mask_connected_to_border(close_mask)
    if not bg_mask.any():
        return img_bytes, False, {"reason": "no background-colored region touches the border"}

    bg_u8 = (bg_mask.astype(np.uint8)) * 255
    kernel = np.ones((3, 3), np.uint8)
    # Fill small foreground-colored noise specks inside the background.
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove small background-colored specks that poked into the artwork.
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    removed_fraction = float((bg_u8 > 0).mean())
    if removed_fraction < 0.005:
        return img_bytes, False, {"reason": "no meaningful background area found"}

    new_alpha = base_alpha.copy()
    new_alpha[bg_u8 > 0] = 0

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    out[:, :, 3] = new_alpha

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise ValueError("Could not re-encode background-removed image.")

    return buf.tobytes(), True, {
        "confidence": confidence,
        "removed_fraction": removed_fraction,
        "background_color_bgr": [float(c) for c in ref_color],
    }
