import numpy as np
import astropy.units as u
import naima
from naima.models import (
    ExponentialCutoffPowerLaw,
    BrokenPowerLaw,
    PionDecay,
    Synchrotron,
    InverseCompton,
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


# ===============================================================
#   DEFAULT INITIAL PARAMETERS
# ===============================================================

default_params_HadronicModel = {
    "ECPL": {
        "p0": np.array([36, 2, np.log10(20)]),
        "labels": ["log10(norm)", "alpha", "log10(cutoff_energy)"],
    },
    "BPL": {
        "p0": np.array([26, 2, 0.8, 1.0]),
        "labels": ["log10(norm)", "alpha", "beta", "log10(break_energy)"],
    },
}
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
):
    """
    Compute the leptonic (Inverse Compton) model flux, electron distribution,
    and total energy in electrons.

    Parameters
    ----------
    pars : array-like
        Model parameters depending on model_type:
        - ECPL: [log10(norm), alpha, log10(cutoff_energy)]
        - BPL : [log10(norm), alpha, beta, log10(break_energy)]
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

     # Synchrotron
    SYN = Synchrotron(electron_spectrum, B=B)

    # --- Inverse Compton gamma-ray production ---
    IC = InverseCompton(electron_spectrum, seed_photon_fields=seed_photon_fields)

    # Gamma-ray model flux at the input energies
    syn_sed = SYN.sed(data, distance=distance)
    ic_sed = IC.sed(data, distance=distance)
    model = syn_sed + ic_sed
    #model = ic_sed

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

def lnprior_LeptonicModel(pars, model_type="ECPL"):
    """
    Log-prior for leptonic (IC) model parameters.

    Parameters
    ----------
    pars : array-like
        Model parameters.
    model_type : str
        "ECPL" or "BPL"
    """
    if model_type.upper() == "ECPL":
        # pars = [log10(norm), alpha, log10(cutoff_energy)]
        logprob = (
            naima.uniform_prior(pars[0], 30, 40)  # normalization
            + naima.uniform_prior(pars[1], 1, 5)  # electron index
            + naima.uniform_prior(pars[2], -1, 4) # log10(E_cut / TeV)
        )

    elif model_type.upper() == "BPL":
        # pars = [log10(norm), alpha, beta, log10(break_energy)]
        logprob = (
            naima.uniform_prior(pars[0], 15, 40)  # normalization
            + naima.uniform_prior(pars[1], 1, 3)  # low-energy index
            + naima.uniform_prior(pars[2], -1, 6) # curvature / delta-index
            + naima.uniform_prior(pars[3], -1, 3) # log10(E_break / TeV)
            #+ naima.uniform_prior(pars[2], -1, 3)
        )

    else:
        raise ValueError("model_type must be 'ECPL' or 'BPL'")

    return logprob


# ===============================================================
#   DEFAULT INITIAL PARAMETERS (LEPTONIC)
# ===============================================================

default_params_LeptonicModel = {
    "ECPL": {
        "p0": np.array([36, 2.0, np.log10(10.0)]),
        "labels": ["log10(norm)", "alpha", "log10(cutoff_energy)"],
    },
    "BPL": {
        "p0": np.array([30, 2.0, 1.0, 0.6]),
        "labels": ["log10(norm)", "alpha", "beta", "log10(break_energy)"],
    },
}    

