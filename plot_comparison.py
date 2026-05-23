"""Render a comparison figure for all six Fitzpatrick skin types.

Layout
------
Rows    : Fitzpatrick types I → VI
Columns : four illumination angles (0°, 30°, 60°, 80°) + Rd(r) profile

Usage
-----
    python plot_comparison.py                          # default 128 px
    python plot_comparison.py --px 256 --out fig.png  # higher resolution
"""

import argparse
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from sss.skin_params import FITZPATRICK_TYPES, build_skin_bssrdf
from sss.renderer import PatchRenderer, build_r_grid, hemispherical_reflectance


ANGLES     = [0, 30, 60, 80]   # degrees
SKIN_ORDER = ["I", "II", "III", "IV", "V", "VI"]

# Approximate sRGB for the radial profile plot background per skin type.
# These are hand-tuned perceptual references, not outputs of the model.
SKIN_REF_COLORS = {
    "I":   "#f2d5c8",
    "II":  "#e8b89a",
    "III": "#c68642",
    "IV":  "#8d5524",
    "V":   "#5c3317",
    "VI":  "#2d1b0e",
}


def gamma_correct(arr: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma to a linear float array in [0, 1]."""
    return np.where(arr <= 0.0031308,
                    12.92 * arr,
                    1.055 * arr ** (1.0 / 2.4) - 0.055)


def tensor_to_rgb(img: torch.Tensor, global_scale: float = 1.0) -> np.ndarray:
    """(H, W, 3) float → (H, W, 3) uint8, gamma-corrected.

    global_scale is computed across all images so that relative brightness
    differences between skin types are preserved.
    """
    arr = img.detach().cpu().float().numpy() / max(global_scale, 1e-8)
    arr = np.clip(arr, 0.0, 1.0)
    arr = gamma_correct(arr)
    return (arr * 255).astype(np.uint8)


def render_all(px: int = 128, device: torch.device = torch.device("cpu")):
    renderer = PatchRenderer(patch_size_mm=8.0, img_px=px, device=device)
    r_grid   = build_r_grid(r_max=4.0, n=400, device=device)

    results = {}   # {skin_type: {"images": [img...], "Rd": tensor, "R_hemi": tensor}}

    for key in SKIN_ORDER:
        print(f"  Computing Fitzpatrick type {key} ...")
        model, ftype = build_skin_bssrdf(key, learnable=False, device=device)
        model.eval()

        with torch.no_grad():
            Rd = model(r_grid)       # (N_r, 3)
            R_hemi = hemispherical_reflectance(Rd, r_grid)

            images = []
            for theta in ANGLES:
                img = renderer.render(model, theta_i_deg=theta,
                                      phi_i_deg=0.0, irradiance=1.0, n_r=400)
                images.append(img)

        results[key] = {"images": images, "Rd": Rd, "R_hemi": R_hemi, "ftype": ftype}
        print(f"    R_hemi(RGB) = {R_hemi.cpu().numpy().round(4)}")

    return results, r_grid


def make_figure(results, r_grid, out_path: str = "skin_comparison.png"):
    # Global scale = 99th-percentile across ALL rendered images so that
    # relative brightness between skin types is preserved in the figure.
    all_vals = np.concatenate([
        np.concatenate([img.detach().cpu().numpy().ravel() for img in res["images"]])
        for res in results.values()
    ])
    global_scale = float(np.percentile(all_vals, 99))
    n_rows = len(SKIN_ORDER)
    n_cols = len(ANGLES) + 1   # +1 for Rd profile

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows), facecolor="#1a1a1a")
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.08, wspace=0.06,
        left=0.06, right=0.98, top=0.93, bottom=0.04,
    )

    # Column headers
    col_titles = [f"θᵢ = {a}°" for a in ANGLES] + ["Rd(r) profile"]
    for j, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")

    # Per-row content
    for i, key in enumerate(SKIN_ORDER):
        res    = results[key]
        ftype  = res["ftype"]
        Rd_arr = res["Rd"].cpu().numpy()          # (N_r, 3)
        r_arr  = r_grid.cpu().numpy()

        # ── Rendered patch images ──────────────────────────────────────────
        for j, img_t in enumerate(res["images"]):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(tensor_to_rgb(img_t, global_scale))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")
                sp.set_linewidth(0.5)

            # Row label on the leftmost image
            if j == 0:
                ax.set_ylabel(
                    f"Type {key}\n{ftype.label}",
                    color="white", fontsize=8, rotation=90,
                    labelpad=6, va="center",
                )

        # ── Rd(r) profile plot ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[i, len(ANGLES)])
        ax.set_facecolor("#0d0d0d")

        colors = {"R": "#e84040", "G": "#40e840", "B": "#4080ff"}
        for ci, (ch, col) in enumerate(colors.items()):
            ax.semilogy(r_arr, Rd_arr[:, ci], color=col,
                        linewidth=1.4, label=ch if i == 0 else "")

        ax.set_xlim(0, 4.0)
        ax.set_ylim(1e-5, 2.0)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(True, color="#333", linewidth=0.4)

        if i == 0:
            ax.legend(loc="upper right", framealpha=0.3,
                      labelcolor="white", fontsize=8)
        if i < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("r  [mm]", color="white", fontsize=8)

        R_hemi = res["R_hemi"].cpu().numpy()
        ax.text(0.97, 0.96,
                f"R={R_hemi[0]:.3f} G={R_hemi[1]:.3f} B={R_hemi[2]:.3f}",
                transform=ax.transAxes, color="white",
                fontsize=6, ha="right", va="top",
                bbox=dict(facecolor="#000", alpha=0.4, pad=2, edgecolor="none"))

    # Global title
    fig.suptitle(
        "Dipole BSSRDF · Donner & Jensen 2005 · Fitzpatrick Skin Types I–VI",
        color="white", fontsize=13, y=0.97,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--px",  type=int, default=128, help="patch render resolution")
    p.add_argument("--out", default="skin_comparison.png")
    p.add_argument("--device", default="")
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    results, r_grid = render_all(px=args.px, device=device)
    make_figure(results, r_grid, out_path=args.out)


if __name__ == "__main__":
    main()
