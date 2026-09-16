"""
Skyline extraction by dynamic programming.

Primary reference
-----------------
W.-N. Lie, T. C.-I. Lin, T.-C. Lin, K.-S. Hung,
"A robust dynamic programming algorithm to extract skyline in images for
navigation", Pattern Recognition Letters 26(2):221-230, 2005.
doi:10.1016/j.patrec.2004.08.021

The image is treated as a multi-stage graph: one stage per image column, one
node per row. The skyline is the minimum-cost path from the left edge to the
right edge of the image. Node cost is low where a pixel looks like a
sky/ground boundary; transition cost penalises large vertical jumps between
neighbouring columns. Because the search is global, the path bridges short
breaks in the edge map (haze, low contrast, blown-out sky) rather than
fragmenting into disconnected pieces -- the robustness property the original
paper is built around.

The same DP-over-columns back-end is still used by modern learned skyline
detectors, which only replace the hand-crafted node cost with a learned one:

T. Ahmad, E. Emami, M. Cadik, G. Bebis, "Resource Efficient Mountainous
Skyline Extraction using Shallow Learning", IJCNN 2021, arXiv:2107.10997.

This module deliberately keeps the hand-crafted cost, so it needs no training,
no GPU, and runs in a few milliseconds per frame -- which matches the
edge-hardware constraint of the thesis.

Node cost
---------
Two complementary cues are combined, both cheap:

1. Local edge cue -- the *signed* vertical derivative. At a sky/ground
   boundary intensity falls as you move down the image (bright sky above,
   darker terrain below), so dI/dy is strongly negative. Using the signed
   derivative instead of gradient magnitude suppresses cloud edges and ground
   texture, which come in both signs. This is the "preferred orientation"
   prior of Lie et al. made explicit.

2. Global region cue -- for a candidate boundary row, the mean intensity of
   everything above it minus the mean intensity of everything below it. This
   is computed for every (column, row) in O(1) from a column-wise cumulative
   sum. It stays informative when haze washes out the local edge, which is
   exactly the failure mode of the raw dataset used here.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SkylineResult:
    """Extracted skyline for one image.

    Attributes
    ----------
    y : np.ndarray
        Row index of the skyline for every column, in *original* image
        pixels. Length equals the original image width.
    y_small : np.ndarray
        The same path at the internal working resolution.
    cost : np.ndarray
        Node-cost map at working resolution (low = looks like skyline).
        Useful for debugging and for figures.
    shape : tuple[int, int]
        (height, width) of the original image.
    roll_deg : float
        Roll angle estimated by a robust straight-line fit to the extracted
        path. Only meaningful when the true horizon is distant and roughly
        straight; see notes in README.
    """

    y: np.ndarray
    y_small: np.ndarray
    cost: np.ndarray
    shape: tuple
    roll_deg: float = 0.0
    meta: dict = field(default_factory=dict)


def _norm01(a, lo_pct=1.0, hi_pct=99.0):
    """Percentile-robust rescale to [0, 1]."""
    lo = np.percentile(a, lo_pct)
    hi = np.percentile(a, hi_pct)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def sky_likelihood(image_bgr, tex_window=9, w_color=0.45, w_bright=0.25,
                   w_texture=0.30):
    """Per-pixel "how sky-like is this?" map in [0, 1].

    Three cues, all cheap and all chosen because they survive atmospheric
    haze, which is what defeats a plain intensity-gradient detector on this
    dataset:

    * colour -- blue minus red. Sky is blue, or blue-white when hazy; dry
      terrain is brown-orange. The sign of B-R separates them even when the
      two have almost the same brightness.
    * brightness -- sky is the bright end of the scene.
    * texture -- local standard deviation. Sky and cloud are smooth; terrain,
      vegetation and buildings are not. This is the cue that still works on a
      washed-out distant ridge, where colour and brightness have both been
      flattened by haze.
    """
    img = image_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)

    colour = _norm01(b - r)
    bright = _norm01((b + g + r) / 3.0)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    k = (tex_window, tex_window)
    mu = cv2.boxFilter(gray, -1, k)
    mu2 = cv2.boxFilter(gray * gray, -1, k)
    texture = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    smooth_cue = 1.0 - _norm01(texture)

    sky = w_color * colour + w_bright * bright + w_texture * smooth_cue
    return cv2.GaussianBlur(sky, (5, 5), 0)


def _node_cost(image_bgr, w_edge=0.45, w_region=0.55):
    """Cost of each pixel being on the skyline. Low cost = likely skyline."""
    sky = sky_likelihood(image_bgr)
    h, w = sky.shape

    # --- Cue 1: local edge. Sky-likelihood falls as you cross the boundary
    # downward, so the signed derivative is strongly negative there. Using the
    # signed derivative (not the magnitude) suppresses cloud edges and ground
    # texture, which come in both signs.
    gy = cv2.Sobel(sky, cv2.CV_32F, 0, 1, ksize=3)
    edge = _norm01(np.maximum(-gy, 0.0), hi_pct=99.5)

    # --- Cue 2: global region split. For a candidate boundary row, how much
    # more sky-like is everything above it than everything below it? O(1) per
    # pixel via a column-wise cumulative sum. This is what keeps the path on
    # the true horizon when a nearer, higher-contrast ridge competes with it.
    cum = np.cumsum(sky, axis=0)
    total = cum[-1:, :]
    n_above = np.arange(1, h + 1, dtype=np.float32)[:, None]
    n_below = np.maximum(h - n_above, 1.0)
    mean_above = cum / n_above
    mean_below = (total - cum) / n_below
    region = _norm01(mean_above - mean_below)

    return 1.0 - (w_edge * edge + w_region * region)


def _dp_shortest_path(cost, max_jump, smooth):
    """Minimum-cost left-to-right path, one row per column.

    cost : (H, W) node costs.
    Returns an array of length W with the chosen row per column.
    """
    h, w = cost.shape
    acc = np.empty((h, w), dtype=np.float32)
    back = np.zeros((h, w), dtype=np.int32)
    acc[:, 0] = cost[:, 0]

    shifts = np.arange(-max_jump, max_jump + 1)
    penalties = smooth * np.abs(shifts).astype(np.float32)

    for x in range(1, w):
        prev = acc[:, x - 1]
        # candidates[k, y] = cost of coming from row y-shifts[k] in prev column
        candidates = np.full((len(shifts), h), np.inf, dtype=np.float32)
        for k, (s, pen) in enumerate(zip(shifts, penalties)):
            # Landing on row y from row y - s.
            if s >= 0:
                candidates[k, s:] = prev[: h - s] + pen if s > 0 else prev + pen
            else:
                candidates[k, :h + s] = prev[-s:] + pen
        best_k = np.argmin(candidates, axis=0)
        acc[:, x] = cost[:, x] + candidates[best_k, np.arange(h)]
        back[:, x] = np.arange(h) - shifts[best_k]

    path = np.empty(w, dtype=np.int32)
    path[-1] = int(np.argmin(acc[:, -1]))
    for x in range(w - 1, 0, -1):
        path[x - 1] = back[path[x], x]
    return path


def _estimate_roll(path):
    """Robust line fit to the path; returns roll in degrees.

    Uses cv2.fitLine with an L1 norm so buildings and trees sticking above the
    horizon do not drag the fit.
    """
    pts = np.stack([np.arange(len(path)), path], axis=1).astype(np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L1, 0, 0.01, 0.01).ravel()
    return float(np.degrees(np.arctan2(vy, vx)))


def extract_skyline(
    image_bgr,
    work_width=480,
    max_jump=10,
    smooth=0.12,
    w_edge=0.45,
    w_region=0.55,
):
    """Extract the skyline from a BGR image.

    Parameters
    ----------
    work_width : int
        Internal processing width. The image is downscaled to this width for
        the DP (keeping aspect ratio) and the resulting path is scaled back
        up. 480 keeps a 1920x1080 frame at a few milliseconds per call.
    max_jump : int
        Largest vertical step, in working-resolution rows, allowed between
        adjacent columns. Must be big enough for building silhouettes.
    smooth : float
        Cost charged per row of vertical jump. Higher = straighter path.
    w_edge, w_region : float
        Relative weight of the local edge cue and the global region cue.

    Returns
    -------
    SkylineResult
    """
    h0, w0 = image_bgr.shape[:2]
    scale = work_width / float(w0)
    work_height = max(2, int(round(h0 * scale)))
    small = cv2.resize(image_bgr, (work_width, work_height), interpolation=cv2.INTER_AREA)

    cost = _node_cost(small, w_edge=w_edge, w_region=w_region)
    path_small = _dp_shortest_path(cost, max_jump=max_jump, smooth=smooth)

    # Scale the path back to original resolution.
    xs_small = np.arange(work_width)
    xs_full = np.linspace(0, work_width - 1, w0)
    path_full = np.interp(xs_full, xs_small, path_small) / scale
    path_full = np.clip(path_full, 0, h0 - 1)

    roll = _estimate_roll(path_small)

    return SkylineResult(
        y=path_full,
        y_small=path_small,
        cost=cost,
        shape=(h0, w0),
        roll_deg=roll,
        meta={"work_size": (work_height, work_width), "scale": scale},
    )
