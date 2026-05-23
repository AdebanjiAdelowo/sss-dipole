"""Multi-layered dipole BSSRDF — Donner & Jensen 2005.

Extension to layered media
--------------------------
Donner & Jensen (2005) extend the single-layer dipole to an N-layer slab
stack by treating each layer as contributing its own dipole response,
attenuated by the transmittance of all layers above it.

For two layers — epidermis (layer 0) + dermis (layer 1, semi-infinite) —
the total diffuse reflectance is:

    Rd_total(r) = Rd_0(r)
                + Td_0² · Rd_1(r) / (1 − Rd_1_hemi · Rd_0_back)

Where:
  Rd_0(r)     single-layer BSSRDF profile of the epidermis
  Td_0        diffuse transmittance through the epidermis (scalar per channel)
  Rd_1(r)     semi-infinite BSSRDF of the dermis
  Rd_1_hemi   hemispherical reflectance of the dermis (1D integral of Rd_1)
  Rd_0_back   hemispherical reflectance of epidermis seen from below

The Td_0² factor accounts for light passing through the epidermis twice
(once down to the dermis, once back up to the surface).  The denominator
corrects for inter-layer multiple scattering using the adding-doubling
principle.  For skin (low inter-layer reflectance) the denominator ≈ 1.

For N layers the recurrence extends naturally:
    Rd_stack[i:] = Rd_i + Td_i² · Rd_stack[i+1:] / (1 − R_hemi_i+1 · R_hemi_i_back)

This implementation is fully differentiable: sigma_a, sigma_s_prime, and
thicknesses are all PyTorch tensors that can carry gradients.

Reference
---------
Donner C., Jensen H.W. (2005). "Light diffusion in multi-layered translucent
materials." SIGGRAPH 2005.
"""

import math
import torch
import torch.nn as nn

from .dipole import DipoleBSSRDF


class MultiLayerBSSRDF(nn.Module):
    """Differentiable multi-layer dipole BSSRDF for stratified media.

    Parameters
    ----------
    n_layers : int    Number of layers (last layer is semi-infinite).
    eta      : float  Refractive index relative to air (≈1.4 for skin).

    Forward
    -------
    r                 : (N,)         radial distances [mm]
    sigma_a           : (L, C)       absorption coefficients per layer [mm⁻¹]
    sigma_s_prime     : (L, C)       reduced scattering per layer [mm⁻¹]
    thicknesses       : (L−1,)       slab thicknesses [mm]; last layer ∞

    Returns
    -------
    Rd : (N, C)  total diffuse reflectance profile [mm⁻²]
    """

    def __init__(self, n_layers: int = 2, eta: float = 1.4):
        super().__init__()
        self.n_layers = n_layers
        self.dipole   = DipoleBSSRDF(eta=eta)

    # ── Hemispherical integrals ─────────────────────────────────────────────

    @staticmethod
    def _hemispherical_Rd(
        Rd_profile: torch.Tensor,   # (N, C)
        r: torch.Tensor,            # (N,)
    ) -> torch.Tensor:              # (C,)
        """Integrate Rd(r) · 2πr dr over the profile samples (trapezoidal)."""
        integrand = Rd_profile * (2.0 * math.pi * r).unsqueeze(-1)  # (N, C)
        return torch.trapezoid(integrand, r, dim=0)                  # (C,)

    # ── Public forward ──────────────────────────────────────────────────────

    def forward(
        self,
        r: torch.Tensor,              # (N,)
        sigma_a: torch.Tensor,        # (L, C)
        sigma_s_prime: torch.Tensor,  # (L, C)
        thicknesses: torch.Tensor,    # (L-1,)  last layer semi-infinite
    ) -> torch.Tensor:                # (N, C)

        L = self.n_layers
        assert sigma_a.shape[0] == L
        assert thicknesses.shape[0] == L - 1

        # Compute per-layer dipole profiles — semi-infinite for each layer.
        layer_Rd = [
            self.dipole(r, sigma_a[i], sigma_s_prime[i])
            for i in range(L)
        ]  # list of (N, C)

        # Diffuse transmittance through each finite-thickness layer.
        Td = [
            self.dipole.diffuse_transmittance(
                thicknesses[i], sigma_a[i], sigma_s_prime[i]
            )
            for i in range(L - 1)
        ]  # list of (C,)

        # Adding-doubling recurrence (back to front).
        # Rd_stack starts as the deepest (semi-infinite) layer.
        Rd_stack = layer_Rd[-1]   # (N, C)

        for i in range(L - 2, -1, -1):
            # Hemispherical reflectance of the stack below layer i
            R_below_hemi = self._hemispherical_Rd(Rd_stack, r)  # (C,)
            # Hemispherical reflectance of layer i seen from below
            R_i_back     = self._hemispherical_Rd(layer_Rd[i], r)  # (C,)

            denom = (1.0 - R_below_hemi * R_i_back).clamp(min=1e-6)  # (C,)

            # Td_i² accounts for two passes through layer i.
            # Shape broadcast: Td[i]²: (C,) → (1, C)
            Rd_stack = (
                layer_Rd[i]
                + (Td[i] ** 2).unsqueeze(0) * Rd_stack / denom.unsqueeze(0)
            )  # (N, C)

        return Rd_stack  # (N, C)


# ── Convenience: skin-specific two-layer model ──────────────────────────────

class SkinBSSRDF(nn.Module):
    """Two-layer skin model: epidermis + dermis, fully differentiable.

    Parameters can be held fixed (requires_grad=False) or optimised
    (requires_grad=True) for inverse-rendering / fitting tasks.

    Units: all optical coefficients in mm⁻¹, thicknesses in mm.
    """

    def __init__(
        self,
        sigma_a_epi:     torch.Tensor,   # (C,) epidermis absorption
        sigma_s_epi:     torch.Tensor,   # (C,) epidermis scattering
        sigma_a_derm:    torch.Tensor,   # (C,) dermis absorption
        sigma_s_derm:    torch.Tensor,   # (C,) dermis scattering
        thickness_epi:   torch.Tensor,   # scalar  epidermis thickness [mm]
        g:               float = 0.8,    # Henyey-Greenstein anisotropy
        eta:             float = 1.4,
        learnable:       bool = False,
    ):
        super().__init__()
        def _p(x):
            return nn.Parameter(x.float().clone(), requires_grad=learnable)

        self.log_sigma_a_epi   = _p(sigma_a_epi.log())
        self.log_sigma_s_epi   = _p(sigma_s_epi.log())
        self.log_sigma_a_derm  = _p(sigma_a_derm.log())
        self.log_sigma_s_derm  = _p(sigma_s_derm.log())
        self.log_thickness_epi = _p(thickness_epi.log())

        self.g    = g
        self.mlss = MultiLayerBSSRDF(n_layers=2, eta=eta)

    # Keep optical coefficients strictly positive via log parameterisation.
    @property
    def sigma_a_epi(self):    return self.log_sigma_a_epi.exp()
    @property
    def sigma_s_epi(self):    return self.log_sigma_s_epi.exp()
    @property
    def sigma_a_derm(self):   return self.log_sigma_a_derm.exp()
    @property
    def sigma_s_derm(self):   return self.log_sigma_s_derm.exp()
    @property
    def thickness_epi(self):  return self.log_thickness_epi.exp()

    def forward(self, r: torch.Tensor) -> torch.Tensor:  # (N,) → (N, C)
        sigma_a       = torch.stack([self.sigma_a_epi,
                                     self.sigma_a_derm])          # (2, C)
        sigma_s_prime = torch.stack([self.sigma_s_epi  * (1 - self.g),
                                     self.sigma_s_derm * (1 - self.g)])  # (2, C)
        thicknesses   = self.thickness_epi.unsqueeze(0)            # (1,)
        return self.mlss(r, sigma_a, sigma_s_prime, thicknesses)
