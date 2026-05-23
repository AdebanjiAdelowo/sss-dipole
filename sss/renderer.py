"""Flat-surface-patch renderer using the BSSRDF.

Rendering equation for a flat Lambertian-scatter surface lit by a
distant directional light (irradiance E [W/m²] at angle θ_i from normal):

    L_o(x_o) = ∫_surface S_d(‖x_o − x_i‖) · (n·ω_i) · E  dA_i

For an infinite flat surface with isotropic BSSRDF Rd(r):

    L_o = E · cos(θ_i) · ∫₀^∞ Rd(r) · 2π r dr        (single-scattering)

The last integral is the hemispherical reflectance R_hemi. To show spatial
variation we render a finite patch lit by a point source overhead, which
requires the BSSRDF convolution:

    I(x_o) = ∫ Rd(‖x_o − x_i‖) · F_t(θ_i) · L_i(x_i) dA_i

We discretise the 1-D (radially symmetric) integral and also allow a 2-D
image rendering for display.  All ops are differentiable.
"""

import math
import torch
import torch.nn as nn


# ── 1-D radial profile helpers ────────────────────────────────────────────────

def build_r_grid(
    r_max: float = 5.0,
    n: int = 512,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Linearly spaced radial grid from ε to r_max [mm]."""
    return torch.linspace(1e-3, r_max, n, device=device)


def hemispherical_reflectance(
    Rd: torch.Tensor,   # (N, C)
    r:  torch.Tensor,   # (N,)
) -> torch.Tensor:      # (C,)
    """∫₀^r_max Rd(r) · 2π r dr  via trapezoidal quadrature."""
    integrand = Rd * (2.0 * math.pi * r).unsqueeze(-1)   # (N, C)
    return torch.trapezoid(integrand, r, dim=0)            # (C,)


# ── Fresnel transmittance ─────────────────────────────────────────────────────

def fresnel_transmittance(cos_theta: torch.Tensor, eta: float = 1.4) -> torch.Tensor:
    """Unpolarised Fresnel transmittance for an air → skin interface.

    Uses the exact Fresnel equations.  cos_theta is the cosine of the
    incident angle (in air).
    """
    cos_t_sq = 1.0 - (1.0 / eta ** 2) * (1.0 - cos_theta ** 2)
    cos_t_sq = cos_t_sq.clamp(min=0.0)   # total internal reflection guard
    cos_t = cos_t_sq.sqrt()

    # s and p polarisation reflectances
    rs = ((cos_theta - eta * cos_t) / (cos_theta + eta * cos_t + 1e-8)) ** 2
    rp = ((eta * cos_theta - cos_t) / (eta * cos_theta + cos_t + 1e-8)) ** 2
    Fr = 0.5 * (rs + rp)
    return (1.0 - Fr).clamp(min=0.0, max=1.0)


# ── 2-D patch renderer ────────────────────────────────────────────────────────

class PatchRenderer(nn.Module):
    """Render a square surface patch under a directional light.

    The renderer evaluates the outgoing radiance at each surface pixel
    by convolving the BSSRDF profile with the incident irradiance field.

    For an infinite slab lit at angle θ_i the convolution reduces to:
        I(x_o) = E · F_t(θ_i) · cos(θ_i) · ∫ Rd(r) 2π r dr
              = E · F_t(θ_i) · cos(θ_i) · R_hemi

    We render a finite patch by discretising the BSSRDF as a lookup table
    and convolving it with the per-pixel irradiance using the separability
    of the radial kernel (direct integration along each output pixel).

    Parameters
    ----------
    patch_size_mm : float   physical side length of the rendered patch [mm]
    img_px        : int     output image resolution (square)
    eta           : float   surface refractive index (for Fresnel correction)
    """

    def __init__(
        self,
        patch_size_mm: float = 8.0,
        img_px: int = 128,
        eta: float = 1.4,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.patch_size_mm = patch_size_mm
        self.img_px        = img_px
        self.eta           = eta
        self.device        = device

        # Pixel spacing [mm]
        self.px_mm = patch_size_mm / img_px

        # Pre-build pixel-centre grid [mm]  shape: (img_px, img_px, 2)
        c = torch.linspace(-patch_size_mm / 2 + self.px_mm / 2,
                            patch_size_mm / 2 - self.px_mm / 2,
                            img_px, device=device)
        gx, gy = torch.meshgrid(c, c, indexing="xy")
        self.register_buffer("grid_x", gx)
        self.register_buffer("grid_y", gy)

    def render(
        self,
        bssrdf_model: nn.Module,
        theta_i_deg: float = 0.0,
        phi_i_deg:   float = 0.0,
        irradiance:  float = 1.0,
        n_r:         int   = 256,
    ) -> torch.Tensor:
        """Render the patch for a single directional light.

        Parameters
        ----------
        bssrdf_model  : callable  r → (N, C) returning Rd profile
        theta_i_deg   : illumination polar angle from normal [°]
        phi_i_deg     : illumination azimuth [°] (affects shadow projection)
        irradiance    : incident irradiance E [W/m²] (or arbitrary units)
        n_r           : radial profile samples

        Returns
        -------
        image : (H, W, C)  outgoing radiance in [0, 1]-ish units
        """
        theta_i = math.radians(theta_i_deg)
        phi_i   = math.radians(phi_i_deg)
        cos_ti  = math.cos(theta_i)

        # Incident direction projected onto the surface plane
        dx = math.sin(theta_i) * math.cos(phi_i)
        dy = math.sin(theta_i) * math.sin(phi_i)

        # Fresnel transmittance at the entry point (same for all pixels)
        Ft = fresnel_transmittance(
            torch.tensor([cos_ti], device=self.device)
        ).item()

        # Build radial lookup table
        r_max = self.patch_size_mm * math.sqrt(2.0)
        r_grid = build_r_grid(r_max, n_r, self.device)  # (n_r,)

        with torch.set_grad_enabled(bssrdf_model.training):
            Rd_profile = bssrdf_model(r_grid)  # (n_r, C)

        C = Rd_profile.shape[-1]
        H = W = self.img_px

        # For each output pixel, the incident point is shifted by the light
        # direction times the "shadow offset" — for a distant source every
        # surface pixel receives the same hemispherical convolution,
        # modulated only by a directional shadow mask (none for flat surface).
        # We therefore compute a single R_hemi and scale by Ft·cos(θ_i)·E.
        R_hemi = hemispherical_reflectance(Rd_profile, r_grid)   # (C,)

        # Outgoing radiance (uniform over the flat patch)
        L_o = irradiance * Ft * cos_ti * R_hemi                  # (C,)

        # For spatial variation, convolve Rd with a soft disc representing
        # the light footprint shifted by illumination angle.
        # We render the per-pixel irradiance by projecting a grid of incident
        # positions and accumulating Rd(distance).
        #
        # Here we use a simple 2-D brute-force integration:
        # For each output pixel o and sample incident pixel i:
        #   r = ||x_o - x_i||, add Rd(r) * E_i
        # Memory-efficient: evaluate one output pixel at a time in the radial
        # direction (since Rd is isotropic, we use the 1-D lookup + 2-D grid).

        image = self._convolve_2d(Rd_profile, r_grid, irradiance, Ft, cos_ti)
        return image   # (H, W, C)

    def _convolve_2d(
        self,
        Rd_profile: torch.Tensor,   # (n_r, C)
        r_grid:     torch.Tensor,   # (n_r,)
        E:          float,
        Ft:         float,
        cos_ti:     float,
    ) -> torch.Tensor:              # (H, W, C)
        """Convolve Rd with a uniform irradiance disc (vectorised over rows)."""
        H = W = self.img_px
        px = self.px_mm

        # Build interpolation: Rd(r) as a function, evaluated by torch.searchsorted.
        # For each pixel pair (o, i): r = sqrt((xo-xi)^2 + (yo-yi)^2)
        # We integrate over source pixels i on a coarser grid for speed.
        src_step = max(1, H // 32)   # source grid subsampling
        xi = self.grid_x[::src_step, ::src_step].reshape(-1)   # (M,)
        yi = self.grid_y[::src_step, ::src_step].reshape(-1)

        # Output pixels: (H*W, 2)
        xo = self.grid_x.reshape(-1)   # (H*W,)
        yo = self.grid_y.reshape(-1)

        # Pairwise distances: (H*W, M)
        dx = xo.unsqueeze(1) - xi.unsqueeze(0)   # (H*W, M)
        dy = yo.unsqueeze(1) - yi.unsqueeze(0)
        r_pair = (dx ** 2 + dy ** 2).sqrt()       # (H*W, M)

        # Look up Rd at each r via linear interpolation of the profile.
        Rd_vals = _interp1d(r_grid, Rd_profile, r_pair)  # (H*W, M, C)

        # Sum over source pixels, scale by pixel area and irradiance.
        src_area = (px * src_step) ** 2
        I = Rd_vals.sum(dim=1) * src_area * E * Ft * cos_ti   # (H*W, C)
        return I.reshape(H, W, -1)


def _interp1d(
    x: torch.Tensor,   # (N,)   strictly increasing
    y: torch.Tensor,   # (N, C)
    xi: torch.Tensor,  # (...)  query points
) -> torch.Tensor:     # (..., C)
    """Differentiable 1-D linear interpolation."""
    shape = xi.shape
    xi_flat = xi.reshape(-1).clamp(x[0], x[-1])   # (Q,)

    idx = torch.searchsorted(x.contiguous(), xi_flat.contiguous(), right=True)
    idx = idx.clamp(1, len(x) - 1)

    x0 = x[idx - 1]
    x1 = x[idx]
    t  = ((xi_flat - x0) / (x1 - x0 + 1e-12)).unsqueeze(-1)   # (Q, 1)

    y0 = y[idx - 1]   # (Q, C)
    y1 = y[idx]       # (Q, C)
    yi = y0 + t * (y1 - y0)                                    # (Q, C)
    return yi.reshape(*shape, y.shape[-1])
