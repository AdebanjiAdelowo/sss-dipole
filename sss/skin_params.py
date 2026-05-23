"""Fitzpatrick skin-type optical parameters.

Sources
-------
The values here are synthesised from:

1. Krishnaswamy A., Baranoski G.V.G. (2004). "A biophysically-based
   spectral model of light interaction with human skin."  Eurographics 2004.
   — Provides melanin fraction (f_mel) per Fitzpatrick type.

2. Meglinski I.V., Matcher S.J. (2002). "Quantitative assessment of skin
   layers absorption and skin reflectance spectra simulation in the visible
   and near-infrared spectral regions." Physiol. Meas. 23, 741–753.
   — Layer-by-layer σ_a and σ_s for epidermis/dermis at multiple wavelengths.

3. Donner C., Jensen H.W. (2005). "Light diffusion in multi-layered
   translucent materials." SIGGRAPH 2005.
   — Effective σ_a / σ_s for rendered skin in the RGB channels.

4. Jacques S.L. (2013). "Optical properties of biological tissues: a review."
   Phys. Med. Biol. 58, R37–R61.
   — Wavelength-dependent baseline for blood and tissue.

RGB channel mapping
-------------------
We use three representative wavelengths:
    R ≈ 650 nm    (red, low melanin/blood absorption)
    G ≈ 532 nm    (green, high oxyhemoglobin peak)
    B ≈ 450 nm    (blue, high melanin and Rayleigh scatter)

All coefficients are in mm⁻¹.

Epidermis layer
---------------
Thickness: 0.008–0.020 mm (increases slightly with pigmentation)
σ_a_epi = f_mel · σ_a_mel + (1 − f_mel) · σ_a_tissue
σ_s_epi ≈ constant across types (melanosomes are similar in size)

Dermis layer (semi-infinite for our model)
-------------------------------------------
σ_a_derm = f_blood · σ_a_oxy + (1 − f_blood) · σ_a_tissue
σ_s_derm ≈ constant, slightly higher than epidermis

Melanin absorption (mm⁻¹, from Krishnaswamy & Baranoski 2004)
    R(650nm): 1.70
    G(532nm): 2.80
    B(450nm): 4.80

Oxyhemoglobin absorption (mm⁻¹ for 1% blood volume fraction)
    R(650nm): 0.28
    G(532nm): 8.40
    B(450nm): 3.20

Baseline tissue absorption (mm⁻¹)
    R(650nm): 0.028
    G(532nm): 0.055
    B(450nm): 0.130

Epidermis scattering (mm⁻¹, Mie + Rayleigh)
    R: 14.0,  G: 18.0,  B: 24.0    (g ≈ 0.79)

Dermis scattering (mm⁻¹)
    R: 18.0,  G: 22.0,  B: 28.0    (g ≈ 0.78)
"""

import torch
from dataclasses import dataclass
from typing import Dict


# ── Base spectra ──────────────────────────────────────────────────────────────

# Melanin (eumelanin) — dominant pigment; mm⁻¹ per unit concentration
_SIGMA_A_MEL    = torch.tensor([1.70, 2.80, 4.80])   # R, G, B
# Oxyhemoglobin — dominant absorber in dermis; mm⁻¹ for 1 vol.% blood
_SIGMA_A_OXY    = torch.tensor([0.28, 8.40, 3.20])
# Baseline tissue water + lipid absorption; mm⁻¹
_SIGMA_A_TISSUE = torch.tensor([0.028, 0.055, 0.130])

# Scattering (before (1-g) reduction)
_SIGMA_S_EPI    = torch.tensor([14.0, 18.0, 24.0])   # mm⁻¹
_SIGMA_S_DERM   = torch.tensor([18.0, 22.0, 28.0])   # mm⁻¹


# ── Fitzpatrick type definitions ──────────────────────────────────────────────
# f_mel   : melanin volume fraction (dimensionless, 0–1)
# f_blood : blood volume fraction in dermis (dimensionless, 0–1)
# thickness_epi : epidermis thickness in mm

@dataclass
class FitzpatrickType:
    name: str        # Roman numeral label
    label: str       # descriptive label
    f_mel: float     # melanin fraction
    f_blood: float   # blood volume fraction in dermis
    thickness_epi: float  # epidermis thickness [mm]


FITZPATRICK_TYPES: Dict[str, FitzpatrickType] = {
    "I":   FitzpatrickType("I",   "Very fair (Celtic)",        f_mel=0.010, f_blood=0.010, thickness_epi=0.008),
    "II":  FitzpatrickType("II",  "Fair (Northern European)",  f_mel=0.040, f_blood=0.015, thickness_epi=0.010),
    "III": FitzpatrickType("III", "Medium (Middle European)",  f_mel=0.100, f_blood=0.020, thickness_epi=0.012),
    "IV":  FitzpatrickType("IV",  "Olive (Mediterranean)",     f_mel=0.180, f_blood=0.025, thickness_epi=0.014),
    "V":   FitzpatrickType("V",   "Brown (Middle Eastern)",    f_mel=0.270, f_blood=0.030, thickness_epi=0.016),
    "VI":  FitzpatrickType("VI",  "Dark (Sub-Saharan)",        f_mel=0.430, f_blood=0.040, thickness_epi=0.020),
}


# ── Parameter computation ─────────────────────────────────────────────────────

def compute_optical_params(ftype: FitzpatrickType) -> dict:
    """Return the optical parameters for a Fitzpatrick type as tensors.

    Returns a dict with keys:
        sigma_a_epi     : (3,) absorption of epidermis [mm⁻¹]
        sigma_s_epi     : (3,) scattering of epidermis (before (1-g)) [mm⁻¹]
        sigma_a_derm    : (3,) absorption of dermis [mm⁻¹]
        sigma_s_derm    : (3,) scattering of dermis (before (1-g)) [mm⁻¹]
        thickness_epi   : scalar tensor, epidermis thickness [mm]
    """
    f  = ftype.f_mel
    fb = ftype.f_blood

    sigma_a_epi  = f * _SIGMA_A_MEL + (1.0 - f) * _SIGMA_A_TISSUE
    sigma_a_derm = fb * _SIGMA_A_OXY + (1.0 - fb) * _SIGMA_A_TISSUE

    return {
        "sigma_a_epi":   sigma_a_epi.clone(),
        "sigma_s_epi":   _SIGMA_S_EPI.clone(),
        "sigma_a_derm":  sigma_a_derm.clone(),
        "sigma_s_derm":  _SIGMA_S_DERM.clone(),
        "thickness_epi": torch.tensor(ftype.thickness_epi),
    }


def build_skin_bssrdf(
    fitz_type: str,
    learnable: bool = False,
    device: torch.device = torch.device("cpu"),
):
    """Construct a SkinBSSRDF model for the given Fitzpatrick type.

    Parameters
    ----------
    fitz_type : one of 'I', 'II', 'III', 'IV', 'V', 'VI'
    learnable : if True, all parameters carry gradients (for optimisation)
    device    : target device

    Returns
    -------
    SkinBSSRDF instance
    """
    from .multilayer import SkinBSSRDF

    ftype  = FITZPATRICK_TYPES[fitz_type]
    params = compute_optical_params(ftype)

    model = SkinBSSRDF(
        sigma_a_epi    = params["sigma_a_epi"].to(device),
        sigma_s_epi    = params["sigma_s_epi"].to(device),
        sigma_a_derm   = params["sigma_a_derm"].to(device),
        sigma_s_derm   = params["sigma_s_derm"].to(device),
        thickness_epi  = params["thickness_epi"].to(device),
        g              = 0.79,
        eta            = 1.4,
        learnable      = learnable,
    )
    return model, ftype
