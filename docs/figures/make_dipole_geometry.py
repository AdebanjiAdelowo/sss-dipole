"""Draw the dipole source configuration to scale for one layer of the implemented model.

Source depths are computed with the same formulas as sss/dipole.py (z_r = 1/sigma'_t,
z_v = z_r + 4AD, A from boundary_A), using the Fitzpatrick type III dermis parameters from
sss/skin_params.py in the red channel.

    python docs/figures/make_dipole_geometry.py   ->   docs/figures/dipole_geometry.svg
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sss.dipole import boundary_A
from sss.skin_params import build_skin_bssrdf

SKIN_TYPE, CHANNEL, R_EXIT = "III", 0, 1.0   # red channel; exit point 1 mm from entry
model, _ = build_skin_bssrdf(SKIN_TYPE, learnable=False)
g, eta = model.g, model.mlss.dipole.eta
sa = float(model.sigma_a_derm[CHANNEL])
ssp = float(model.sigma_s_derm[CHANNEL]) * (1 - g)
st_p = sa + ssp
D = 1.0 / (3.0 * st_p)
A = boundary_A(eta)
z_r = 1.0 / st_p
z_v = z_r + 4.0 * A * D
z_b = 2.0 * A * D                     # extrapolated boundary, midway between the sources
d_r, d_v = np.hypot(R_EXIT, z_r), np.hypot(R_EXIT, z_v)

plt.rcParams.update({"font.size": 9, "svg.fonttype": "path", "svg.hashsalt": "fig"})
fig, ax = plt.subplots(figsize=(7.4, 4.6))
fig.patch.set_facecolor("white")
x0, x1 = -1.0, 3.0
ax.fill_between([x0, x1], [-z_v - 0.1] * 2, [0, 0], color="#f4e1d8", zorder=0)
ax.plot([x0, x1], [0, 0], color="#6b3b2a", lw=1.6)
ax.plot([x0, x1], [z_b, z_b], color="0.45", lw=0.9, ls="--")
ax.text(x1, z_b - 0.04, f"extrapolated boundary\n$z_b = 2AD$ = {z_b:.2f} mm", ha="right",
        va="top", fontsize=8, color="0.35")
ax.text(x1, -0.05, "scattering medium (semi-infinite)", ha="right", va="top", fontsize=8,
        color="#6b3b2a")

ax.annotate("", xy=(0, 0), xytext=(-0.45, 0.55), arrowprops=dict(arrowstyle="-|>", lw=1.4))
ax.text(-0.5, 0.6, "incident light at $x_i$", ha="right", va="bottom", fontsize=8.5)
ax.annotate("", xy=(R_EXIT + 0.45, 0.55), xytext=(R_EXIT, 0), arrowprops=dict(arrowstyle="-|>", lw=1.4))
ax.text(R_EXIT + 0.5, 0.6, "exit at $x_o$", ha="left", va="bottom", fontsize=8.5)

ax.plot(0, -z_r, "o", ms=9, color="#c0392b", zorder=5)
ax.text(-0.1, -z_r, f"real source (+)\n$z_r = 1/\\sigma'_t$ = {z_r:.2f} mm", ha="right",
        va="center", fontsize=8.5, color="#c0392b")
ax.plot(0, z_v, "o", ms=9, mfc="white", mec="#1f5fa8", mew=1.8, zorder=5)
ax.text(-0.1, z_v, f"virtual source (−)\n$z_v = z_r + 4AD$ = {z_v:.2f} mm", ha="right",
        va="center", fontsize=8.5, color="#1f5fa8")
ax.plot([0, R_EXIT], [-z_r, 0], color="#c0392b", lw=1)
ax.plot([0, R_EXIT], [z_v, 0], color="#1f5fa8", lw=1)
ax.text(R_EXIT * 0.5, -z_r * 0.5 - 0.03, "$d_r$", color="#c0392b", ha="left", va="top")
ax.text(R_EXIT * 0.55, z_v * 0.5, "$d_v$", color="#1f5fa8", ha="left", va="bottom")
ax.plot([R_EXIT, R_EXIT], [0, -z_r - 0.12], color="0.6", lw=0.6, ls=":")
ax.annotate("", xy=(0, -z_r - 0.12), xytext=(R_EXIT, -z_r - 0.12),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.3"))
ax.text(R_EXIT / 2, -z_r - 0.15, f"r = {R_EXIT:g} mm", ha="center", va="top", fontsize=8, color="0.3")
ax.plot([0, 0], [-z_r - 0.12, z_v], color="0.6", lw=0.6, ls=":")

ax.text(1.25, -0.2,
        f"Type {SKIN_TYPE} dermis, red channel\n"
        f"$\\sigma_a$ = {sa:.3f}, $\\sigma'_s$ = {ssp:.2f} mm$^{{-1}}$\n"
        f"$D = 1/(3\\sigma'_t)$ = {D:.3f} mm,  A = {A:.2f} ($\\eta$ = {eta})",
        fontsize=8, va="top", bbox=dict(fc="white", ec="0.7", lw=0.6, boxstyle="round,pad=0.35"))

ax.set_xlim(x0 - 1.3, x1)
ax.set_ylim(-z_r - 0.75, z_v + 0.25)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
out = Path(__file__).with_name("dipole_geometry.svg")
fig.savefig(out, facecolor="white", metadata={"Date": None})  # deterministic output
if len(sys.argv) > 1:  # optional raster preview path
    fig.savefig(sys.argv[1], dpi=150, facecolor="white")
print(f"wrote {out.relative_to(ROOT)}  (z_r={z_r:.4f}, z_v={z_v:.4f}, A={A:.4f}, D={D:.4f})")
