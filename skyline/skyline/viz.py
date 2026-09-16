"""Figures for inspecting extracted skylines and their descriptors."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

SKY_TINT = np.array([80, 150, 255], dtype=np.float32)   # RGB
LINE_RGB = (1.0, 0.15, 0.15)


def overlay_skyline(image_bgr, result, tint_alpha=0.22):
    """RGB copy of the image with the sky region tinted and the path drawn."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = rgb.shape[:2]
    rows = np.arange(h)[:, None]
    sky_mask = rows < result.y[None, :]
    rgb[sky_mask] = (1 - tint_alpha) * rgb[sky_mask] + tint_alpha * SKY_TINT
    return np.clip(rgb, 0, 255).astype(np.uint8)


def figure_overview(names, images_bgr, results, ncols=3, tint=True):
    """Grid of every frame with its extracted skyline."""
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name, img, res in zip(axes, names, images_bgr, results):
        ax.imshow(overlay_skyline(img, res) if tint else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.plot(np.arange(len(res.y)), res.y, color=LINE_RGB, lw=1.8)
        ax.set_title(f"{name}   (roll {res.roll_deg:+.1f}°)", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Extracted skylines — DP shortest path (Lie et al., 2005)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def figure_frame_detail(name, image_bgr, result, profile, slope):
    """One frame in depth: overlay, node-cost map, profile and its derivative."""
    width_in = 13.0
    h, w = image_bgr.shape[:2]
    img_in = width_in * h / w          # height the photo needs at full width
    cost_in, plot_in = 3.2, 2.4
    fig = plt.figure(figsize=(width_in, img_in + cost_in + plot_in + 1.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[img_in, cost_in, plot_in],
                          hspace=0.3, wspace=0.14)

    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(overlay_skyline(image_bgr, result), aspect="auto")
    ax_img.plot(np.arange(len(result.y)), result.y, color=LINE_RGB, lw=2.2)
    ax_img.set_title(f"{name} — extracted skyline (roll {result.roll_deg:+.1f}°)")
    ax_img.axis("off")

    ax_cost = fig.add_subplot(gs[1, :])
    ax_cost.imshow(result.cost, cmap="magma", aspect="auto")
    ax_cost.plot(np.arange(len(result.y_small)), result.y_small, color="cyan", lw=1.6)
    ax_cost.set_title("Node cost (dark = sky/ground boundary) with the DP path")
    ax_cost.axis("off")

    ax_p = fig.add_subplot(gs[2, 0])
    ax_p.plot(profile, color="#1f4e79", lw=1.6)
    ax_p.set_title("Height profile (mean-removed)", fontsize=10)
    ax_p.set_xlabel("resampled column")
    ax_p.grid(alpha=0.3)

    ax_s = fig.add_subplot(gs[2, 1])
    ax_s.plot(slope, color="#a33", lw=1.2)
    ax_s.set_title("Slope signature (d/dx of profile)", fontsize=10)
    ax_s.set_xlabel("resampled column")
    ax_s.grid(alpha=0.3)

    return fig


def figure_profiles(names, profiles):
    """All height profiles on one axis, plus a stacked heat-map view."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1.2, 1])

    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(profiles)))
    for prof, c, name in zip(profiles, cmap, names):
        axes[0].plot(prof, color=c, lw=1.3, label=name)
    axes[0].set_title("Skyline height profiles — all frames (mean-removed)")
    axes[0].set_xlabel("resampled column")
    axes[0].set_ylabel("normalised elevation")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=5, loc="lower left")

    im = axes[1].imshow(np.stack(profiles), aspect="auto", cmap="RdYlBu_r",
                        interpolation="nearest")
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].set_xlabel("resampled column")
    axes[1].set_title("Same profiles stacked — smooth drift across the sequence")
    fig.colorbar(im, ax=axes[1], label="normalised elevation")

    fig.tight_layout()
    return fig


def figure_fourier(names, fouriers):
    """Fourier magnitude descriptors for every frame."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), width_ratios=[1.15, 1])

    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(fouriers)))
    for f, c in zip(fouriers, cmap):
        axes[0].plot(np.arange(1, len(f) + 1), f, color=c, lw=1.2, alpha=0.85)
    axes[0].set_title("Fourier descriptors (shift-invariant magnitude spectrum)")
    axes[0].set_xlabel("harmonic index")
    axes[0].set_ylabel("normalised magnitude")
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(np.stack(fouriers), aspect="auto", cmap="magma",
                        interpolation="nearest")
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].set_xlabel("harmonic index")
    axes[1].set_title("Descriptor matrix (frames × harmonics)")
    fig.colorbar(im, ax=axes[1])

    fig.tight_layout()
    return fig


def figure_shape_context(name, sc, n_radial=5, n_angular=12, points=(8, 32, 56)):
    """Log-polar histograms at a few contour points, as in Belongie et al."""
    fig, axes = plt.subplots(1, len(points) + 1, figsize=(3.1 * (len(points) + 1), 3.4))

    im0 = axes[0].imshow(sc, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title(f"{name}\nall points × 60 bins", fontsize=9)
    axes[0].set_xlabel("bin")
    axes[0].set_ylabel("contour point")
    fig.colorbar(im0, ax=axes[0])

    for ax, p in zip(axes[1:], points):
        ax.imshow(sc[p].reshape(n_radial, n_angular), cmap="viridis",
                  interpolation="nearest", aspect="auto")
        ax.set_title(f"point {p}", fontsize=9)
        ax.set_xlabel("angular bin")
        ax.set_ylabel("log-radial bin")

    fig.suptitle("Shape context (Belongie et al., 2002)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def figure_distance_matrices(names, matrices, titles):
    """Side-by-side descriptor distance matrices."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.0 * len(matrices), 4.6))
    axes = np.atleast_1d(axes)

    short = [n.replace("frame_", "").replace(".png", "") for n in names]
    for ax, m, t in zip(axes, matrices, titles):
        im = ax.imshow(m, cmap="viridis_r", interpolation="nearest")
        ax.set_title(t, fontsize=10)
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, fontsize=6, rotation=90)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Descriptor distance between frames (bright = similar)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
