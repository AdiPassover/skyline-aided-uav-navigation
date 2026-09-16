"""
Descriptors for an extracted skyline, and matching distances for each.

The skyline is reduced to a 1-D signal -- the height of the sky/ground
boundary as a function of horizontal image position -- and then described in
three complementary ways. All three are named in the thesis related-work
chapter as standard skyline representations.

1. Height profile (the raw contour)
   The signal itself, resampled to a fixed length and normalised. This is what
   contour-based skyline matching operates on, e.g.

     O. Saurer, G. Baatz, K. Koeser, L. Ladicky, M. Pollefeys,
     "Image Based Geo-localization in the Alps",
     International Journal of Computer Vision 116:213-225, 2016.

   Matching a query against a database is a sliding-window correlation over
   this signal, and the offset that wins is a direct estimate of relative yaw.

2. Fourier descriptor
   The magnitude spectrum of the profile. Discarding phase makes the
   descriptor invariant to horizontal shift, i.e. to yaw. That is useful for
   the *retrieval* stage -- shortlist candidate places regardless of heading --
   with the profile correlation above then recovering the heading itself.
   Compact: 32 numbers instead of 256.

3. Shape context
   A log-polar histogram of the contour's own points seen from each sample
   point, from

     S. Belongie, J. Malik, J. Puzicha, "Shape Matching and Object Recognition
     Using Shape Contexts", IEEE TPAMI 24(4):509-522, 2002.

   Unlike the two above it encodes the *spatial layout* of the silhouette
   rather than its pointwise height, so it degrades differently under
   occlusion and partial field of view.

A slope signature (first derivative of the profile) is also provided; it is
not a descriptor in its own right here, but it is the cue that makes peaks and
building edges legible in the figures.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Height profile
# ---------------------------------------------------------------------------

def height_profile(skyline_y, image_height, n_samples=256, normalize=True,
                   detrend=False):
    """Resample the skyline to a fixed-length 1-D signal.

    Parameters
    ----------
    skyline_y : array
        Row index of the skyline per column (as returned in SkylineResult.y).
    image_height : int
        Height of the image the path came from, used to normalise.
    n_samples : int
        Output length. Fixed length is what makes profiles comparable across
        images of different resolution.
    normalize : bool
        If True, subtract the mean so the descriptor is invariant to a
        constant vertical offset, which pitch or altitude would introduce.
    detrend : bool
        If True, also subtract the best-fit straight line. Camera roll tilts
        the whole skyline, adding a linear ramp that dominates the signal and
        swamps the terrain shape underneath it. Removing the ramp makes the
        descriptor roll-invariant, at the cost of discarding any genuine
        large-scale slope in the terrain. Compare both in run_demo.py.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Elevation in [0, 1] before mean-removal, where 1 is the top of the
        image. Higher value = skyline sits higher in the frame.
    """
    y = np.asarray(skyline_y, dtype=np.float64)
    xs_in = np.linspace(0.0, 1.0, len(y))
    xs_out = np.linspace(0.0, 1.0, n_samples)
    resampled = np.interp(xs_out, xs_in, y)
    elevation = 1.0 - resampled / float(image_height)
    if detrend:
        coeffs = np.polyfit(xs_out, elevation, 1)
        elevation = elevation - np.polyval(coeffs, xs_out)
    if normalize:
        elevation = elevation - elevation.mean()
    return elevation


def profile_distance(p, q):
    """1 - normalised cross-correlation between two profiles (0 = identical).

    No shift search here: this compares two profiles as-is. The sliding
    version used for yaw recovery lives in the matching stage, not the
    descriptor.
    """
    a = p - p.mean()
    b = q - q.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(1.0 - np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# 2. Fourier descriptor
# ---------------------------------------------------------------------------

def fourier_descriptor(profile, n_coeffs=32):
    """Shift-invariant magnitude spectrum of the profile.

    The DC term is dropped (it only carries the mean height) and the vector is
    L2-normalised, so the descriptor is invariant to horizontal shift, to a
    constant vertical offset, and to overall amplitude.
    """
    p = np.asarray(profile, dtype=np.float64)
    p = p - p.mean()
    spectrum = np.abs(np.fft.rfft(p))
    mag = spectrum[1: n_coeffs + 1]
    if len(mag) < n_coeffs:  # short profile: pad
        mag = np.pad(mag, (0, n_coeffs - len(mag)))
    return mag / (np.linalg.norm(mag) + 1e-12)


def fourier_distance(a, b):
    """Euclidean distance between two L2-normalised Fourier descriptors."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ---------------------------------------------------------------------------
# 3. Shape context (Belongie et al., 2002)
# ---------------------------------------------------------------------------

def shape_context(profile, n_points=64, n_radial=5, n_angular=12,
                  aspect=0.5):
    """Log-polar shape-context histograms for the skyline contour.

    Parameters
    ----------
    profile : array
        Height profile (any length); it is resampled to `n_points` sample
        points along the contour.
    n_points : int
        Number of contour points to describe.
    n_radial, n_angular : int
        Log-polar bin counts, giving an `n_radial * n_angular` histogram per
        point. Defaults follow the original paper (5 x 12 = 60 bins).
    aspect : float
        Vertical exaggeration applied when embedding the 1-D profile into 2-D.
        The profile is unit-width; without this the height variation would be
        tiny relative to the horizontal extent and the histograms would be
        nearly identical for every image.

    Returns
    -------
    np.ndarray of shape (n_points, n_radial * n_angular)
        Row-normalised histograms, one per contour point.
    """
    p = np.asarray(profile, dtype=np.float64)
    xs_in = np.linspace(0.0, 1.0, len(p))
    xs = np.linspace(0.0, 1.0, n_points)
    ys = np.interp(xs, xs_in, p)

    # Scale-normalise the height so shapes are comparable, then embed in 2-D.
    spread = ys.max() - ys.min()
    if spread > 1e-9:
        ys = (ys - ys.mean()) / spread * aspect
    pts = np.stack([xs, ys], axis=1)

    # Pairwise vectors and the mean distance used for scale normalisation.
    diff = pts[None, :, :] - pts[:, None, :]          # (i, j, 2): j relative to i
    dist = np.linalg.norm(diff, axis=2)
    mean_dist = dist[dist > 0].mean() if np.any(dist > 0) else 1.0
    dist_n = dist / (mean_dist + 1e-12)

    angle = np.arctan2(diff[:, :, 1], diff[:, :, 0])   # (-pi, pi]

    # Log-radial bin edges, as in the paper.
    r_inner, r_outer = 0.125, 2.0
    r_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), n_radial + 1)
    r_bin = np.digitize(dist_n, r_edges) - 1           # -1 .. n_radial-1

    a_edges = np.linspace(-np.pi, np.pi, n_angular + 1)
    a_bin = np.clip(np.digitize(angle, a_edges) - 1, 0, n_angular - 1)

    valid = (r_bin >= 0) & (r_bin < n_radial)
    np.fill_diagonal(valid, False)                     # a point is not its own context

    n_bins = n_radial * n_angular
    hist = np.zeros((n_points, n_bins), dtype=np.float64)
    flat_bin = r_bin * n_angular + a_bin
    for i in range(n_points):
        v = valid[i]
        if not np.any(v):
            continue
        counts = np.bincount(flat_bin[i][v], minlength=n_bins)
        hist[i] = counts / counts.sum()
    return hist


def shape_context_distance(sc_a, sc_b):
    """Mean chi-square distance between corresponding contour points.

    Both contours are sampled left-to-right at the same number of points, so
    point i in one corresponds to point i in the other. That makes the full
    Hungarian matching of the original paper unnecessary here.
    """
    a = np.asarray(sc_a)
    b = np.asarray(sc_b)
    num = (a - b) ** 2
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(den > 0, num / den, 0.0)
    return float(0.5 * chi.sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Auxiliary signal
# ---------------------------------------------------------------------------

def slope_signature(profile):
    """First derivative of the profile: where the silhouette rises and falls."""
    return np.gradient(np.asarray(profile, dtype=np.float64))
