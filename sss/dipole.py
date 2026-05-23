"""Single-layer dipole diffusion BSSRDF.

Physics
-------
Subsurface scattering in a semi-infinite homogeneous medium is described by
the diffusion equation.  Jensen et al. (2001) showed that the solution can be
expressed as two point-source dipoles placed symmetrically about the surface:

  Rd(r) = (α' / 4π) · [ z_r (σ_tr d_r + 1) e^{-σ_tr d_r} / d_r³
                        + z_v (σ_tr d_v + 1) e^{-σ_tr d_v} / d_v³ ]

where
  σ'_t  = σ_a + σ'_s                        reduced extinction
  α'    = σ'_s / σ'_t                        reduced albedo
  D     = 1 / (3 σ'_t)                       diffusion coefficient
  σ_tr  = √(3 σ_a σ'_t)                      effective transport coefficient
  z_r   = 1 / σ'_t                           depth of real source dipole
  z_v   = z_r + 4 A D                        depth of virtual source dipole
  d_r   = √(r² + z_r²)                       distance from real source
  d_v   = √(r² + z_v²)                       distance from virtual source
  A     = (1 + F_dr) / (1 - F_dr)            internal reflection coefficient

The boundary parameter A accounts for total internal reflection at the
air-medium interface via the diffuse Fresnel reflectance F_dr.

All computations use per-channel (RGB) tensors so that the model naturally
handles wavelength-dependent scattering.  Every quantity is a real-valued
torch.Tensor, making the entire model differentiable.

References
----------
Jensen H.W. et al. (2001). "A practical model for subsurface light transport."
SIGGRAPH 2001.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fresnel utilities ──────────────────────────────────────────────────────────

def fresnel_moment1(eta: float) -> float:
    """First Fresnel moment — diffuse internal reflectance F_dr.

    Uses the polynomial fit from Jensen et al. 2001 (eq. 5).
    Valid for η > 1 (optically denser medium, e.g. skin η ≈ 1.4).
    """
    if eta >= 1.0:
        return -1.4399 / eta ** 2 + 0.7099 / eta + 0.6681 + 0.0636 * eta
    else:  # vacuum → medium (shouldn't occur for skin)
        return 0.0


def fresnel_moment2(eta: float) -> float:
    """Second Fresnel moment — for Fresnel-corrected flux boundary condition."""
    if eta >= 1.0:
        return (-0.4399 + 0.7099 / eta - 0.3319 / eta ** 2
                + 0.0636 * eta)
    return 0.0


def boundary_A(eta: float) -> float:
    """Boundary condition parameter A = (1 + F_dr) / (1 - F_dr)."""
    fdr = fresnel_moment1(eta)
    return (1.0 + fdr) / max(1.0 - fdr, 1e-6)


# ── Core dipole ───────────────────────────────────────────────────────────────

class DipoleBSSRDF(nn.Module):
    """Differentiable single-layer dipole BSSRDF.

    Parameters
    ----------
    eta : float
        Relative refractive index of the medium (air=1).  For skin ≈ 1.4.

    Forward inputs
    --------------
    r           : (N,)      radial distances on the surface [mm]
    sigma_a     : (C,)      absorption coefficients [mm⁻¹]
    sigma_s_prime:(C,)      reduced scattering coefficients [mm⁻¹]
                            σ'_s = σ_s (1 − g)

    Returns
    -------
    Rd : (N, C) diffuse reflectance profile [mm⁻²]
    """

    def __init__(self, eta: float = 1.4):
        super().__init__()
        self.eta = eta
        self._A = boundary_A(eta)

    def extra_repr(self) -> str:
        return f"eta={self.eta}, A={self._A:.4f}"

    def forward(
        self,
        r: torch.Tensor,              # (N,)
        sigma_a: torch.Tensor,        # (C,)
        sigma_s_prime: torch.Tensor,  # (C,)
    ) -> torch.Tensor:                # (N, C)

        sigma_a       = sigma_a.clamp(min=1e-8)
        sigma_s_prime = sigma_s_prime.clamp(min=1e-8)

        sigma_t_prime = sigma_a + sigma_s_prime               # (C,)
        alpha_prime   = sigma_s_prime / sigma_t_prime         # (C,)
        D             = 1.0 / (3.0 * sigma_t_prime)          # (C,) diffusion coeff
        sigma_tr      = (3.0 * sigma_a * sigma_t_prime).sqrt()# (C,) transport coeff

        z_r = 1.0 / sigma_t_prime                            # (C,) real source depth
        z_v = z_r + 4.0 * self._A * D                        # (C,) virtual source depth

        r2  = r.unsqueeze(-1) ** 2                            # (N, 1)
        d_r = (r2 + z_r ** 2).sqrt()                          # (N, C)
        d_v = (r2 + z_v ** 2).sqrt()                          # (N, C)

        def _dipole_term(z, d):
            return z * (sigma_tr * d + 1.0) * torch.exp(-sigma_tr * d) / d ** 3

        Rd = (alpha_prime / (4.0 * math.pi)) * (
            _dipole_term(z_r, d_r) + _dipole_term(z_v, d_v)
        )
        return Rd  # (N, C)

    def diffuse_transmittance(
        self,
        thickness: torch.Tensor,     # scalar or (C,)
        sigma_a: torch.Tensor,       # (C,)
        sigma_s_prime: torch.Tensor, # (C,)
    ) -> torch.Tensor:               # (C,)
        """Approximate diffuse transmittance through a slab of given thickness.

        Uses the two-flux approximation:
            Td ≈ exp(−σ_tr · d)
        which is accurate when d is small compared to 1/σ_tr.
        """
        sigma_t_prime = sigma_a + sigma_s_prime
        sigma_tr      = (3.0 * sigma_a * sigma_t_prime).sqrt()
        return torch.exp(-sigma_tr * thickness)
