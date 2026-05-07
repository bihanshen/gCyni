import numpy as np
import astropy.units as u
import naima
from naima.models import (
    ExponentialCutoffPowerLaw,
    BrokenPowerLaw,
    PionDecay,
    Synchrotron,
    InverseCompton,
    Bremsstrahlung,
)


def HadronicModel(
    pars, data, model_type="ECPL", gas_density=88.0 * u.cm**-3, distance=5.5 * u.kpc
):
    """
    Compute the hadronic model flux, proton distribution, and total energy.

    Parameters
    ----------
    pars : array-like
        Model parameters depending on model_type:
        - ECPL: [log10(norm), alpha, log10(cutoff_energy)]
        - BPL: [log10(norm), alpha, beta, log10(break_energy)]
    data : array-like
        Energy array or data points to evaluate the flux at.
    model_type : str, optional
        Type of model to use: "ECPL" (default) or "BPL".

    Returns
    -------
    model : array
        Predicted flux at `data` energies.
    (proton_energy, proton_dist) : tuple
        Proton energy grid and corresponding particle distribution.
    Wp : Quantity
        Total energy in protons above 1 GeV.
    """

    amplitude = 10 ** pars[0] / u.eV

    if model_type.upper() == "ECPL":
        alpha = pars[1]
        ecutoff = (10 ** pars[2]) * u.TeV
        proton_spectrum = ExponentialCutoffPowerLaw(
            amplitude=amplitude, e_0=1 * u.TeV, alpha=alpha, e_cutoff=ecutoff, beta=1.0
        )

    elif model_type.upper() == "BPL":
        alpha = pars[1]
        beta = pars[2]
        e_break = (10 ** pars[3]) * u.TeV
        proton_spectrum = BrokenPowerLaw(
            amplitude=amplitude,
            e_0=1.0 * u.TeV,
            e_break=e_break,
            alpha_1=alpha,
            #alpha_2=2.91,
            alpha_2=alpha+beta,
        )

    else:
        raise ValueError("model_type must be 'ECPL' or 'BPL'")

    # Pion decay gamma-ray production
    PP = PionDecay(proton_spectrum, nh=gas_density)

    # Compute gamma-ray model flux
    model = PP.sed(data, distance=distance)

    # Proton distribution
    proton_energy = np.logspace(-4, 2, 50) * u.TeV
    proton_dist = PP.particle_distribution(proton_energy)

    # Total energy in protons above 1 GeV
    Wp = PP.compute_Wp(Epmin=1 * u.GeV)

    return model, (proton_energy, proton_dist), Wp


# ===============================================================
#   PRIOR DEFINITIONS
# ===============================================================


def lnprior_HadronicModel(pars, model_type="ECPL"):
    """
    Log-prior for hadronic model parameters.

    Parameters
    ----------
    pars : array-like
        Model parameters.
    model_type : str
        "ECPL" or "BPL"
    """
    if model_type.upper() == "ECPL":
        logprob = (
            naima.uniform_prior(pars[0], 30, 40)
            + naima.uniform_prior(pars[1], 1, 5)
            + naima.uniform_prior(pars[2], -1, 4)
        )

    elif model_type.upper() == "BPL":
        logprob = (
            naima.uniform_prior(pars[0], 25, 40)
            + naima.uniform_prior(pars[1], 0, 2.5)
            + naima.uniform_prior(pars[2], 0, 1.5)
            + naima.uniform_prior(pars[3], -1, 4)
        )

    else:
        raise ValueError("model_type must be 'ECPL' or 'BPL'")

    return logprob


'''
# Fix the second index in BPL
default_params_HadronicModel = {
    "ECPL": {
        "p0": np.array([36, 2, np.log10(20.0)]),
        "labels": ["log10(norm)", "alpha", "log10(cutoff_energy)"],
    },
    "BPL": {
        "p0": np.array([36, 2, 0.8]),
        "labels": ["log10(norm)", "alpha", "log10(break_energy)"],
    },
}
'''
# ===============================================================
#   LEPTONIC (IC) MODEL
# ===============================================================

def LeptonicModel(
    pars,
    data,
    model_type="ECPL",
    distance=2.0 * u.kpc,
    B=20 * u.uG,
    seed_photon_fields=None,
    gas_density=88.0 * u.cm**-3,
    fit_B=True,
):
    """
    Compute the leptonic (Synchrotron + Inverse Compton + Bremsstrahlung)
    model flux, electron distribution, and total energy in electrons.

    Parameters
    ----------
    pars : array-like
        Model parameters depending on model_type:
        Model parameters depending on model_type and fit_B:
        - ECPL, fit_B=True : [log10(norm), alpha, log10(cutoff_energy), log10(B/uG)]
        - ECPL, fit_B=False: [log10(norm), alpha, log10(cutoff_energy)]
        - BPL,  fit_B=True : [log10(norm), alpha, beta, log10(break_energy), log10(B/uG)]
        - BPL,  fit_B=False: [log10(norm), alpha, beta, log10(break_energy)]
      where 'norm' is the electron normalization at 1 TeV (in 1/eV).
    data : array-like (Quantity)
        Photon energies at which to evaluate the flux, with energy units.
    model_type : str, optional
        Type of electron spectrum: "ECPL" (default) or "BPL".
    distance : Quantity, optional
        Source distance (default 5.5 kpc).
    seed_photon_fields : list of str, optional
        Seed photon fields for Inverse Compton. Example:
        ["CMB"], ["CMB", "IR", "FIR"]. If None, defaults to ["CMB"].
    gas_density : Quantity, optional
        Target gas density for the Bremsstrahlung component.

    Returns
    -------
    model : Quantity
        Predicted differential photon flux at `data` energies,
        in units of 1 / (cm2 s energy).
    (electron_energy, electron_dist) : tuple
        Electron energy grid (Quantity) and corresponding differential
        particle distribution N(E) (same spectral model units).
    We : Quantity
        Approximate total energy in electrons above 1 GeV.
    """

    if seed_photon_fields is None:
        seed_photon_fields = ["CMB"]

    # Treat B as a free parameter when requested. The last parameter is log10(B/uG).
    if fit_B:
        B = (10 ** pars[-1]) * u.uG

    # Electron normalization at 1 TeV, in 1/eV (same convention as HadronicModel)
    amplitude = 10 ** pars[0] / u.eV

    # --- Electron spectrum definition ---
    if model_type.upper() == "ECPL":
        alpha = pars[1]
        ecutoff = (10 ** pars[2]) * u.TeV
        electron_spectrum = ExponentialCutoffPowerLaw(
            amplitude=amplitude,
            e_0=1.0 * u.TeV,
            alpha=alpha,
            e_cutoff=ecutoff,
            beta=1.0,
        )

    elif model_type.upper() == "BPL":
        alpha = pars[1]
        beta = pars[2]
        e_break = (10 ** pars[3]) * u.TeV
        electron_spectrum = BrokenPowerLaw(
            amplitude=amplitude,
            e_0=1.0 * u.TeV,
            e_break=e_break,
            alpha_1=alpha,
            alpha_2=alpha + beta,
            #alpha_2=4.82,
        )

    else:
        raise ValueError("model_type must be 'ECPL' or 'BPL'")

    # --- Radiative components ---
    SYN = Synchrotron(electron_spectrum, B=B)
    IC = InverseCompton(electron_spectrum, seed_photon_fields=seed_photon_fields)
    Brem = Bremsstrahlung(electron_spectrum, n0=gas_density)

    # Gamma-ray model flux at the input energies
    syn_sed = SYN.sed(data, distance=distance)
    ic_sed = IC.sed(data, distance=distance)
    brem_sed = Brem.sed(data, distance=distance)
    model = syn_sed + ic_sed + brem_sed

    # Electron distribution (for diagnostics; similar to proton distribution in hadronic case)
    electron_energy = np.logspace(-8, 2, 50) * u.TeV  # 10^-4 to 10^2 TeV
    # differential distribution N(E): use the electron spectrum directly
    electron_dist = electron_spectrum(electron_energy)

    '''
    # --- Approximate total energy in electrons above 1 GeV ---
    Emin = 1.0 * u.GeV
    Emax = 1e5 * u.GeV
    n_int = 200

    # Do the integral in eV so amplitude (1/eV) and E share units
    E_grid = np.logspace(
        np.log10(Emin.to_value(u.eV)),
        np.log10(Emax.to_value(u.eV)),
        n_int,
    ) * u.eV

    N_E = electron_spectrum(E_grid)      # units: 1/eV

    # Integrand: E * N(E)  → dimensionless
    integrand = E_grid * N_E            # (eV * 1/eV) = dimensionless

    # Integral: ∫ (E * N(E)) dE → units of E (eV)
    We = np.trapz(integrand, E_grid)    # result in eV

    # Convert to whatever you like:
    We = We.to(u.TeV)                   # or .to(u.TeV)
    '''
    We = IC.compute_We(Eemin=1 * u.GeV)

    return model, (electron_energy, electron_dist), We

# ===============================================================
#   PRIOR DEFINITIONS (LEPTONIC)
# ===============================================================

def lnprior_LeptonicModel(pars, model_type="ECPL", fit_B=True):
    """
    Log-prior for leptonic (IC + synchrotron + Bremsstrahlung) model parameters.

    Parameters
    ----------
    pars : array-like
        Model parameters.
    model_type : str
        "ECPL" or "BPL"
    fit_B : bool
        If True, include log10(B/uG) as the final free parameter.
    """
    if model_type.upper() == "ECPL":
        # pars = [log10(norm), alpha, log10(cutoff_energy), log10(B/uG)]
        logprob = (
            naima.uniform_prior(pars[0], 10, 40)
            + naima.uniform_prior(pars[1], -1, 4)
            + naima.uniform_prior(pars[2], -2, 4)
        )
        if fit_B:
            logprob += naima.uniform_prior(pars[3], -1, 3)  # B ~ 0.1--1000 uG

    elif model_type.upper() == "BPL":
        # pars = [log10(norm), alpha, beta, log10(break_energy), log10(B/uG)]
        logprob = (
            naima.uniform_prior(pars[0], 15, 40)
            + naima.uniform_prior(pars[1], -1, 3)
            + naima.uniform_prior(pars[2], -1, 5)
            + naima.uniform_prior(pars[3], -1, 6)
        )
        if fit_B:
            logprob += naima.uniform_prior(pars[4], -1, 3)  # B ~ 0.1--1000 uG

    else:
        raise ValueError("model_type must be 'ECPL' or 'BPL'")

    return logprob





# Suggested starting points when fitting the magnetic field as a free parameter
default_params_LeptonicModel = {
    "ECPL": {
        "p0": np.array([30, 2.0, np.log10(20.0), np.log10(20.0)]),
        "labels": ["log10(norm)", "alpha", "log10(cutoff_energy)", "log10(B_uG)"],
    },
    "BPL": {
        "p0": np.array([30, 1.8, 1.0, np.log10(0.3), np.log10(20.0)]),
        "labels": ["log10(norm)", "alpha", "beta", "log10(break_energy)", "log10(B_uG)"],
    },
}
