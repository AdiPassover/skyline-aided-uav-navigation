"""Thesis Figure 4.2 -- per-frame residuals and their accumulation.

Thesis rendering of the closure figure F17 (`vo_closure/closure_figures.py`), with the
research-log framing removed and the arm labels put in thesis terms. Top row: what one frame
looks like. Bottom row: what a few thousand of them compose to.

Reads only committed artifacts: runs/<id>/logical_transform.csv, whose per-frame
`inc_anisotropy` / `inc_log_scale` / `rigid_accum_scale` columns and accumulated `g??`
transform are written by the Java capture side. The stdout line per arm reproduces the F17
legend values (mean increment, robust sigma, end-of-run accumulated scale) exactly.

    cd evaluation && python tools/thesis_figures/residual_accumulation.py
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "figures", "vo", "residual_accumulation.png")

CENTRE = np.array([306.0, 256.0])   # image centre in the shrunk (shrinkScale 0.5) render frame
BAND = (0.87, 1.30)                 # scale band the LiDAR-measured height range explains
TILT = {"HK-B": 1.0379, "AM-C": 1.0124}   # 1/cos(max boresight tilt), per window

ARMS = [
    ("HK-B", "homography", "hkairport01-b-homography-rigid-v1", "#1f6fb4", "-"),
    ("HK-B", "affine",     "hkairport01-b-affine-rigid-v1",     "#4bb3d6", "--"),
    ("AM-C", "homography", "amtown01-c-homography-rigid-v1",    "#c81e1e", "-"),
    ("AM-C", "affine",     "amtown01-c-affine-rigid-v1",        "#e08214", "--"),
]


def load(run_id):
    path = os.path.join(ROOT, "runs", run_id, "logical_transform.csv")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    # EXP-VO-011's filter: init and restart rows carry no meaningful increment.
    kept = [r for r in rows if r["event"] not in ("init", "restart")]
    return {
        "inc_anisotropy": np.array([float(r["inc_anisotropy"]) for r in kept]),
        "inc_log_scale": np.array([float(r["inc_log_scale"]) for r in kept]),
        "accum_scale": np.array([float(r["rigid_accum_scale"]) for r in rows]),
        "frame": np.array([float(r["frame_index"]) for r in rows]),
        "G": np.array([[[float(r["g%d%d" % (i, j)]) for j in range(3)] for i in range(3)]
                       for r in rows]),
    }


def jacobian_anisotropy(M):
    """sigma_1/sigma_2 of the Jacobian of the homography M at the image centre."""
    x, y = CENTRE
    w = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    if abs(w) < 1e-12:
        return np.nan
    u = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / w
    v = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / w
    J = np.array([[M[0, 0] - u * M[2, 0], M[0, 1] - u * M[2, 1]],
                  [M[1, 0] - v * M[2, 0], M[1, 1] - v * M[2, 1]]]) / w
    s = np.linalg.svd(J, compute_uv=False)
    return s[0] / s[1]


def main():
    data = {}
    for window, model, run_id, _colour, _ls in ARMS:
        d = load(run_id)
        accum = np.empty(len(d["G"]))
        for k, G in enumerate(d["G"]):
            try:
                accum[k] = jacobian_anisotropy(np.linalg.inv(G))
            except np.linalg.LinAlgError:
                accum[k] = np.nan
        d["accum_aniso"] = accum
        data[(window, model)] = d
        inc = d["inc_log_scale"]
        robust = 1.4826 * np.median(np.abs(inc - np.median(inc)))
        print("%-5s %-11s n=%5d mean=%+.2e robust=%.2e ends=%6.2fx aniso_med=%5.2f"
              % (window, model, inc.size, inc.mean(), robust,
                 d["accum_scale"][-1], np.nanmedian(accum[1:])))

    plt.rcParams.update({
        "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
        "axes.linewidth": 0.7, "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.borderpad": 0.35, "legend.labelspacing": 0.3,
    })
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(6.4, 4.6))

    # (a) per-frame anisotropy, against each flight's tilt-explained ceiling
    for window, model, _run, colour, ls in ARMS:
        v = np.sort(data[(window, model)]["inc_anisotropy"])
        ax_a.plot(v, np.arange(1, v.size + 1) / v.size, color=colour, ls=ls, lw=1.0,
                  label="%s, %s" % (window, model))
    for window, style in (("HK-B", ":"), ("AM-C", (0, (1, 2)))):
        ax_a.axvline(TILT[window], color="0.35", ls=style, lw=0.8)
        ax_a.text(TILT[window], 0.30, " %s tilt ceiling" % window, fontsize=6.2,
                  color="0.35", rotation=90, va="bottom", ha="left")
    ax_a.set_xlim(1.0, 1.045)
    ax_a.set_ylim(0, 1.02)
    ax_a.set_xlabel(r"per-frame anisotropy $\sigma_1/\sigma_2$ of $D_k^{-1}$")
    ax_a.set_ylabel("cumulative fraction")
    ax_a.set_title("(a) one frame: essentially a similarity")
    ax_a.legend(loc="center right")

    # (b) per-frame log-scale increment: the bias sits well inside the spread
    for window, model, _run, colour, ls in ARMS:
        v = data[(window, model)]["inc_log_scale"]
        ax_b.hist(v, bins=np.linspace(-0.006, 0.006, 160), histtype="step",
                  color=colour, ls=ls, lw=0.9,
                  label="%s, %s  (mean %+.1e)" % (window, model, v.mean()))
        ax_b.axvline(v.mean(), color=colour, lw=1.4, alpha=0.85)
    ax_b.axvline(0.0, color="k", lw=0.7)
    ax_b.set_xlim(-0.006, 0.006)
    ax_b.set_xlabel(r"per-frame log-scale increment $\log\sqrt{|\det J_k|}$")
    ax_b.set_ylabel("frames per bin")
    ax_b.set_title("(b) one frame: the bias hides in the noise")
    ax_b.legend(loc="upper left")

    # (c) accumulated scale, against exp(n * that arm's own mean increment)
    for window, model, _run, colour, ls in ARMS:
        d = data[(window, model)]
        ax_c.plot(d["frame"], d["accum_scale"], color=colour, ls=ls, lw=1.0,
                  label=u"%s, %s  (ends %.1f×)" % (window, model, d["accum_scale"][-1]))
        n = np.arange(d["frame"].size)
        ax_c.plot(d["frame"], np.exp(n * d["inc_log_scale"].mean()),
                  color=colour, ls=":", lw=0.7, alpha=0.75)
    ax_c.axhspan(BAND[0], BAND[1], color="#7fbf7b", alpha=0.28, lw=0)
    ax_c.axhline(1.0, color="k", lw=0.6)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("frame index")
    ax_c.set_ylabel("accumulated scale")
    ax_c.set_title(r"(c) composed: dotted $=\exp(n\bar{\ell})$ from (b)")
    ax_c.legend(loc="upper left")

    # (d) accumulated anisotropy: the composed transform is not an isotropic scale
    for window, model, _run, colour, ls in ARMS:
        d = data[(window, model)]
        ax_d.plot(d["frame"], np.clip(d["accum_aniso"], 1.0, 30.0),
                  color=colour, ls=ls, lw=1.0,
                  label="%s, %s  (median %.2f)"
                        % (window, model, np.nanmedian(d["accum_aniso"][1:])))
    ax_d.axhspan(1.0, TILT["HK-B"], color="#7fbf7b", alpha=0.28, lw=0)
    ax_d.set_yscale("log")
    ax_d.set_ylim(0.98, 32)
    ax_d.set_xlabel("frame index")
    ax_d.set_ylabel(r"accumulated $\sigma_1/\sigma_2$ of $G_k^{-1}$")
    ax_d.set_title("(d) composed: not an isotropic scale")
    ax_d.legend(loc="upper left")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=2.5, pad=1.5)
        ax.grid(True, lw=0.3, alpha=0.35)

    fig.tight_layout(pad=0.5, w_pad=1.2, h_pad=1.4)
    fig.savefig(OUT, dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
