import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from naima.models import ExponentialCutoffPowerLaw, BrokenPowerLaw, PionDecay, InverseCompton, Synchrotron, Bremsstrahlung
from naima.plot import find_ML


def get_parameters(sampler, ML=True):
    """
    Extract model parameters from a Naima MCMC sampler.

    Parameters
    ----------
    sampler : naima.sampling.MCMCSampler
        The Naima MCMC sampler object returned by `naima.run_sampler()`.

    ML : bool, optional (default=True)
        If True, return the maximum-likelihood (ML) parameter set.
        If False, return the median posterior values estimated
        from the flattened MCMC chain.

    Returns
    -------
    pars : dict
        Dictionary of model parameters:
        - If ML=True, contains parameters at maximum likelihood.
        - If ML=False, contains median posterior values.

    Notes
    -----
    - For ML=True, parameters are taken from `sampler.run_info["p0"]`,
      which stores the best-fit parameters at maximum likelihood.
    - For ML=False, this function computes the 16th, 50th, and 84th
      percentiles from the posterior chain to estimate uncertainties.
      Currently, it returns only the median values, but can be modified
      to also return uncertainties if desired.

    Example
    -------
    >>> sampler, pos = naima.run_sampler(...)
    >>> ml_params = get_parameters(sampler, ML=True)
    >>> posterior_medians = get_parameters(sampler, ML=False)
    >>> print(ml_params)
    """

    labels = sampler.labels  # Parameter names
    pars = {}

    if ML:
        # --- Maximum Likelihood parameters ---
        ML_params = sampler.run_info.get("p0", None)
        if ML_params is None:
            raise ValueError("No ML parameters found in sampler.run_info['p0']")

        for i, label in enumerate(labels):
            pars[label] = ML_params[i]

    else:
        # --- Posterior median estimates ---
        dists = sampler.get_chain(flat=True)
        quantiles = [16, 50, 84]

        for i, label in enumerate(labels):
            dist = dists[:, i]
            q_values = np.percentile(dist, quantiles)

            median = q_values[1]
            low_err = median - q_values[0]
            high_err = q_values[2] - median

            # Store median only (like original)
            #pars[label] = median

            # Optionally, if you want to include errors:
            pars[label] = {"median": median, "err_low": low_err, "err_high": high_err}

    return pars

def _extract_param_sets(params_dict):
    """
    Convert params_dict into median / low / high parameter dictionaries.

    Supports either:
      - scalar values
      - {"median": ..., "err_low": ..., "err_high": ...}
    """
    median, low, high = {}, {}, {}

    for key, val in params_dict.items():
        if isinstance(val, dict):
            m = val["median"]
            elo = val["err_low"]
            ehi = val["err_high"]
            median[key] = m
            low[key] = m - elo
            high[key] = m + ehi
        else:
            median[key] = val
            low[key] = val
            high[key] = val

    return median, low, high

def _gamma_index_from_model(model, distance, e1=0.5 * u.TeV, e2=10.0 * u.TeV):
    """
    Estimate local gamma-ray spectral index from two nearby energies.
    """
    E_test = np.array([e1.to_value(u.TeV), e2.to_value(u.TeV)]) * u.TeV
    flux_test = model.flux(E_test, distance=distance)
    return float(-np.log(flux_test[1] / flux_test[0]) / np.log(e2 / e1))

def _flatten_params_dict(params_dict, mode="median"):
    """
    Convert params_dict entries to scalars.

    Supports either:
      - scalar values
      - {"median": ..., "err_low": ..., "err_high": ...}
    """
    flat = {}
    for key, val in params_dict.items():
        if isinstance(val, dict):
            if mode not in val:
                raise KeyError(f"Key '{mode}' not found in parameter entry for '{key}': {val}")
            flat[key] = val[mode]
        else:
            flat[key] = val
    return flat

def estimate_gamma_break_from_model(
    radiative_model,
    distance,
    energy_range=(1e-3, 1e2),   # TeV
    ngrid=400,
    low_frac=0.15,
    high_frac=0.15,
    use_sed=False,
):
    """
    Estimate the gamma-ray break energy directly from the modeled spectrum.

    Parameters
    ----------
    radiative_model : naima radiative model
        e.g. PionDecay(...) or InverseCompton(...)
    distance : astropy.units.Quantity
        Source distance
    energy_range : tuple
        Photon energy range in TeV used for the estimate
    ngrid : int
        Number of grid points
    low_frac, high_frac : float
        Fractions of the grid used to estimate the low/high asymptotic slopes
    use_sed : bool
        If True, estimate break from the SED slope.
        If False, estimate break from dN/dE (recommended).

    Returns
    -------
    E_break : astropy.units.Quantity
        Estimated gamma-ray break energy
    gamma_low : float
        Low-energy asymptotic gamma-ray index
    gamma_high : float
        High-energy asymptotic gamma-ray index
    """
    E = np.logspace(np.log10(energy_range[0]), np.log10(energy_range[1]), ngrid) * u.TeV

    if use_sed:
        y = radiative_model.sed(E, distance=distance).to_value("TeV cm-2 s-1")
        # local slope of SED = d log(SED) / d log(E)
        local_slope = np.gradient(np.log(y), np.log(E.to_value(u.TeV)))
        # convert to equivalent dN/dE photon index if desired:
        # SED = E^2 dN/dE  =>  dlog(SED)/dlogE = 2 - Gamma
        local_index = 2.0 - local_slope
    else:
        y = radiative_model.flux(E, distance=distance).to_value("1 / (TeV cm2 s)")
        local_index = -np.gradient(np.log(y), np.log(E.to_value(u.TeV)))

    nlow = max(3, int(low_frac * ngrid))
    nhigh = max(3, int(high_frac * ngrid))

    gamma_low = np.mean(local_index[:nlow])
    gamma_high = np.mean(local_index[-nhigh:])

    gamma_mid = 0.5 * (gamma_low + gamma_high)

    idx = np.argmin(np.abs(local_index - gamma_mid))
    E_break = E[idx]

    return E_break, gamma_low, gamma_high

def compute_We_from_spectrum(electron_spectrum, Eemin=1.0 * u.GeV, Eemax=100.0 * u.TeV, ngrid=400):
    """
    Compute total electron energy:
        We = ∫ E N(E) dE
    for a Naima particle spectrum.

    Parameters
    ----------
    electron_spectrum : callable
        e.g. BrokenPowerLaw or ExponentialCutoffPowerLaw
    Eemin : Quantity
        Minimum electron energy
    Eemax : Quantity
        Maximum electron energy
    ngrid : int
        Number of integration points

    Returns
    -------
    We : Quantity
        Total electron energy in erg
    """
    Egrid = np.logspace(
        np.log10(Eemin.to_value(u.eV)),
        np.log10(Eemax.to_value(u.eV)),
        ngrid,
    ) * u.eV

    Ne = electron_spectrum(Egrid)   # expected units ~ 1/eV
    integrand = Egrid * Ne          # dimensionless
    We = np.trapz(integrand, Egrid) # energy units

    return We.to(u.erg)


def get_BIC(sampler):
    """
    Compute the Bayesian Information Criterion (BIC) for a Naima MCMC sampler.

    The BIC is a model selection criterion that balances model fit and complexity:

        BIC = k * ln(n) - 2 * ln(L_max)

    where:
        - k : number of free parameters (model dimensionality)
        - n : number of data points
        - L_max : maximum likelihood of the model

    Parameters
    ----------
    sampler : naima.sampling.MCMCSampler
        The Naima MCMC sampler object obtained from `naima.run_sampler()`.

    Returns
    -------
    BIC : float
        The computed Bayesian Information Criterion value.

    Notes
    -----
    Lower BIC values indicate a better balance between goodness-of-fit and
    model simplicity. When comparing two models, the one with the lower BIC
    is generally preferred.

    Example
    -------
    >>> sampler, pos = naima.run_sampler(...)
    >>> bic_value = get_BIC(sampler)
    >>> print(f"BIC = {bic_value:.2f}")
    """

    # Find maximum likelihood and corresponding parameters
    ML, ML_params, ML_errors, _ = find_ML(sampler, None)

    # Number of model parameters
    k = len(ML_params)

    # Number of data points
    n = len(sampler.data)

    # Compute Bayesian Information Criterion
    BIC = k * np.log(n) - 2 * ML

    print(f"BIC value for model: {BIC:.3f}")

    return BIC


def plot_data(ax, data, color, marker, energy_unit, flux_unit, label):
    """
    Plot spectral data with symmetric/asymmetric errors and upper limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    data : astropy.table.Table or pandas.DataFrame
        Table containing columns:
            - 'energy' : photon energy
            - 'flux' : flux values
            - 'flux_error' OR 'flux_error_lo' & 'flux_error_hi' : uncertainties
            - Optional 'ul' : boolean indicating upper limits
    color : str
        Color of the points and error bars.
    marker : str
        Marker style for data points.
    energy_unit : astropy.units.Unit
        Desired unit for the x-axis (energy).
    flux_unit : astropy.units.Unit
        Desired unit for the y-axis (flux).
    label : str
        Label for the dataset in the legend.

    Notes
    -----
    - If 'ul' column exists, data points with `ul=True` are plotted as upper limits.
    - Automatically handles symmetric or asymmetric flux errors.
    """
    keywords_ul = {
        "capsize": 3,
        "marker": ".",
        "elinewidth": 1,
        "zorder": 2,
        "markersize": 0,
        "xerr": None,
    }
    # UL mask (if not present, assume all are detections)
    mask = data["ul"] == True if "ul" in data.colnames else np.zeros(len(data), dtype=bool)

    has_sym = "flux_error" in data.colnames
    has_asym = ("flux_error_lo" in data.colnames) and ("flux_error_hi" in data.colnames)

    if has_sym:
        ax.errorbar(
            data["energy"].to(energy_unit)[~mask],
            data["flux"].to(flux_unit)[~mask],
            data["flux_error"].to(flux_unit)[~mask],
            marker=marker, markersize=6, markerfacecolor="white",
            color=color, ls="", capsize=3, label=label,
        )
    elif has_asym:
        ax.errorbar(
            data["energy"].to(energy_unit)[~mask],
            data["flux"].to(flux_unit)[~mask],
            yerr=[
                data["flux_error_lo"].to(flux_unit)[~mask],
                data["flux_error_hi"].to(flux_unit)[~mask],
            ],
            marker=marker, markersize=6, markerfacecolor="white",
            color=color, ls="", capsize=3, label=label,
        )
    else:
        raise KeyError(
            "No recognized flux error columns found. "
            "Expected 'flux_error' or 'flux_error_lo'/'flux_error_hi'."
        )

    if np.sum(mask) > 0:
        ax.errorbar(
            data["energy"].to(energy_unit)[mask],
            data["flux"].to(flux_unit)[mask],
            0.3 * data["flux"].to(flux_unit)[mask],
            uplims=True,
            color=color,
            ls="",
            **keywords_ul,
        )
    else:
        pass


def plot_multi_instrument(ax, data_dict, energy_unit, flux_unit, energy_range, show=True):
    color = ["blue", "green", "coral", "red", "purple", "sienna", "olive", "teal", "crimson", "grey"]
    marker = ["o", "s", "^", "D", "v", "p", "2", "x", "8", "4"]

    i = 0
    for keys in data_dict.keys():
        plot_data(
            ax,
            data_dict[keys],
            color=color[i],
            marker=marker[i],
            energy_unit=energy_unit,
            flux_unit=flux_unit,
            label=keys,
        )
        i += 1

    # --- Axis formatting ---
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(energy_range)
    ax.set_ylim(1e-14, 1e-9)
    ax.set_xlabel(r"Photon energy [TeV]", fontsize=12)
    ax.set_ylabel(r"$E^2 \, dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
    ax.grid(which="both", color="gray", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", ncol=2)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_hadronic_model(
    params_dict,
    model_type="ECPL",
    gas_density=88.0 * u.cm**-3,
    distance=5.5 * u.kpc,
    energy_range=(1e-5, 1e3),  # in TeV
    ax=None,
    label="Hadronic Model",
    color="black",
    linestyle="-",
    alpha=0.8,
    data_dict=None,
    show=True,
):
    """
    Plot hadronic model SED (BPL or ECPL) and optionally observed data.

    Parameters
    ----------
    params_dict : dict
        Model parameters depending on model_type:
        - ECPL: {"log10(norm)": ..., "alpha": ..., "log10(cutoff_energy)": ...}
        - BPL: {"log10(norm)": ..., "alpha": ..., "beta": ..., "log10(break_energy)": ...}
    model_type : str
        Either "ECPL" or "BPL".
    gas_density : Quantity
        Target density (default 88 cm^-3).
    distance : Quantity
        Source distance (default 5.5 kpc).
    energy_range : tuple
        (emin, emax) in TeV for plotting range.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to plot on (creates new one if None).
    label : str
        Label for the model curve.
    color : str
        Color for the model curve.
    linestyle : str
        Line style for model curve.
    alpha : float
        Transparency for the model curve.
    data_dict : dict, optional
        Optional data dictionary containing observed data.
    show : bool
        If True, calls plt.show() at the end.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axis with the plotted SED.
    """

    spectrum_energy = np.logspace(
        np.log10(energy_range[0]), np.log10(energy_range[1]), 200
    ) * u.TeV

    params_med = _flatten_params_dict(params_dict, mode="median")
    amplitude = 10 ** params_med["log10(norm)"] / u.eV

    if model_type.upper() == "ECPL":
        model = ExponentialCutoffPowerLaw(
            amplitude=amplitude,
            e_0=1 * u.TeV,
            alpha=params_med["alpha"],
            e_cutoff=10 ** params_med["log10(cutoff_energy)"] * u.TeV,
            beta=1.0,
        )
    elif model_type.upper() == "BPL":
        model = BrokenPowerLaw(
            amplitude=amplitude,
            e_0=1 * u.TeV,
            e_break=10 ** params_med["log10(break_energy)"] * u.TeV,
            alpha_1=params_med["alpha"],
            alpha_2=params_med["alpha"] + params_med["beta"],
        )
    else:
        raise ValueError("model_type must be 'ECPL' or 'BPL'")

    PP = PionDecay(model, nh=gas_density)
    sed = PP.sed(spectrum_energy, distance=distance)

    # ---- Compute and print gamma-ray quantities with approximate errors ----
    p_med, p_low, p_high = _extract_param_sets(params_dict)

    def build_hadronic_model_local(pdict):
        amplitude = 10 ** pdict["log10(norm)"] / u.eV

        if model_type.upper() == "ECPL":
            spec = ExponentialCutoffPowerLaw(
                amplitude=amplitude,
                e_0=1 * u.TeV,
                alpha=pdict["alpha"],
                e_cutoff=10 ** pdict["log10(cutoff_energy)"] * u.TeV,
                beta=1.0,
            )
        elif model_type.upper() == "BPL":
            spec = BrokenPowerLaw(
                amplitude=amplitude,
                e_0=1 * u.TeV,
                e_break=10 ** pdict["log10(break_energy)"] * u.TeV,
                alpha_1=pdict["alpha"],
                alpha_2=pdict["alpha"] + pdict["beta"],
            )
        else:
            raise ValueError("model_type must be 'ECPL' or 'BPL'")

        return PionDecay(spec, nh=gas_density)

    PP_med = build_hadronic_model_local(p_med)
    PP_low = build_hadronic_model_local(p_low)
    PP_high = build_hadronic_model_local(p_high)

    Eref = np.array([1.0]) * u.TeV

    dnde_med = PP_med.flux(Eref, distance=distance)[0].to("1/(TeV cm2 s)")
    dnde_low = PP_low.flux(Eref, distance=distance)[0].to("1/(TeV cm2 s)")
    dnde_high = PP_high.flux(Eref, distance=distance)[0].to("1/(TeV cm2 s)")

    sed_med = PP_med.sed(Eref, distance=distance)[0].to("TeV cm-2 s-1")
    sed_low = PP_low.sed(Eref, distance=distance)[0].to("TeV cm-2 s-1")
    sed_high = PP_high.sed(Eref, distance=distance)[0].to("TeV cm-2 s-1")

    # Gamma-ray break estimated numerically from modeled spectrum
    Ebreak_med, _, _ = estimate_gamma_break_from_model(PP_med, distance=distance)
    Ebreak_low, _, _ = estimate_gamma_break_from_model(PP_low, distance=distance)
    Ebreak_high, _, _ = estimate_gamma_break_from_model(PP_high, distance=distance)

    # ---- Spectral indices ----
    if model_type.upper() == "BPL":
        # Approximate gamma-ray indices track proton indices for hadronic BPL
        gamma1_med = p_med["alpha"]
        gamma1_low = p_low["alpha"]
        gamma1_high = p_high["alpha"]

        gamma2_med = p_med["alpha"] + p_med["beta"]
        gamma2_low = p_low["alpha"] + p_low["beta"]
        gamma2_high = p_high["alpha"] + p_high["beta"]

        gamma_at_1tev_med = _gamma_index_from_model(PP_med, distance)
        gamma_at_1tev_low = _gamma_index_from_model(PP_low, distance)
        gamma_at_1tev_high = _gamma_index_from_model(PP_high, distance)
    else:
        gamma_at_1tev_med = _gamma_index_from_model(PP_med, distance)
        gamma_at_1tev_low = _gamma_index_from_model(PP_low, distance)
        gamma_at_1tev_high = _gamma_index_from_model(PP_high, distance)

        gamma1_med = gamma_at_1tev_med
        gamma1_low = gamma_at_1tev_low
        gamma1_high = gamma_at_1tev_high

        gamma2_med = np.nan
        gamma2_low = np.nan
        gamma2_high = np.nan

    # ---- Total proton energy above 1 GeV ----
    Wp_med = PP_med.compute_Wp(Epmin=1.0 * u.GeV).to(u.erg)
    Wp_low = PP_low.compute_Wp(Epmin=1.0 * u.GeV).to(u.erg)
    Wp_high = PP_high.compute_Wp(Epmin=1.0 * u.GeV).to(u.erg)

    print("\n[Gamma-ray spectrum (Hadronic)]")
    print(
        f"dN/dE (1 TeV): {dnde_med:.3e} "
        f"(-{(dnde_med - dnde_low):.3e}, +{(dnde_high - dnde_med):.3e})"
    )
    print(
        f"E^2 dN/dE (1 TeV): {sed_med:.3e} "
        f"(-{(sed_med - sed_low):.3e}, +{(sed_high - sed_med):.3e})"
    )
    print(
        f"Estimated gamma-ray break from modeled spectrum: {Ebreak_med:.3f} "
        f"(-{(Ebreak_med - Ebreak_low):.3f}, +{(Ebreak_high - Ebreak_med):.3f})"
    )

    if model_type.upper() == "BPL":
        print(
            f"Gamma-ray spectral index below break Γ1: {gamma1_med:.3f} "
            f"(-{(gamma1_med - gamma1_low):.3f}, +{(gamma1_high - gamma1_med):.3f})"
        )
        print(
            f"Gamma-ray spectral index above break Γ2: {gamma2_med:.3f} "
            f"(-{(gamma2_med - gamma2_low):.3f}, +{(gamma2_high - gamma2_med):.3f})"
        )
        print(
            f"Local gamma-ray index @10 TeV: {gamma_at_1tev_med:.3f} "
            f"(-{(gamma_at_1tev_med - gamma_at_1tev_low):.3f}, +{(gamma_at_1tev_high - gamma_at_1tev_med):.3f})"
        )
    else:
        print(
            f"Gamma-ray index @10 TeV: {gamma_at_1tev_med:.3f} "
            f"(-{(gamma_at_1tev_med - gamma_at_1tev_low):.3f}, +{(gamma_at_1tev_high - gamma_at_1tev_med):.3f})"
        )

    print(
        f"Total proton energy Wp (>1 GeV): {Wp_med:.3e} "
        f"(-{(Wp_med - Wp_low):.3e}, +{(Wp_high - Wp_med):.3e})"
    )

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(
        spectrum_energy.to("TeV").value,
        sed.to("TeV cm-2 s-1").value,
        lw=2,
        color=color,
        label=label,
        ls=linestyle,
        alpha=alpha,
    )

    return ax



def plot_leptonic_model(
    params_dict,
    model_type="ECPL",
    distance=5.5 * u.kpc,
    energy_range=(1e-5, 1e3),  # in TeV
    seed_photon_fields=None,
    B=10.0 * u.uG,
    gas_density=88.0 * u.cm**-3,
    fit_B=True,
    ax=None,
    label="Leptonic Model",
    color="black",
    linestyle="-",
    alpha=0.8,
    data_dict=None,
    show=True,
):
    """
    Plot leptonic (Inverse Compton + Synchrotron + Bremsstrahlung) SED
    (BPL or ECPL) and optionally observed data. If fit_B=True, the magnetic
    field is read from the final parameter entry (log10(B/uG)).
    """

    if seed_photon_fields is None:
        seed_photon_fields = ["CMB"]

    # If present, read the fitted magnetic field from params_dict.
    if fit_B:
        if "log10(B_uG)" in params_dict:
            Bval = params_dict["log10(B_uG)"]
        elif "log10(B)" in params_dict:
            Bval = params_dict["log10(B)"]
        else:
            Bval = None
        if Bval is not None:
            if isinstance(Bval, dict):
                B = (10 ** Bval["median"]) * u.uG
            else:
                B = (10 ** Bval) * u.uG

    spectrum_energy = (
        np.logspace(np.log10(energy_range[0]), np.log10(energy_range[1]), 200) * u.TeV
    )

    params_med = _flatten_params_dict(params_dict, mode="median")
    amplitude = 10 ** params_med["log10(norm)"] / u.eV

    # --- Build electron spectrum ---
    if model_type.upper() == "ECPL":
        electron_spectrum = ExponentialCutoffPowerLaw(
            amplitude=amplitude,
            e_0=1 * u.TeV,
            alpha=params_med["alpha"],
            e_cutoff=10 ** params_med["log10(cutoff_energy)"] * u.TeV,
            beta=1.0,
        )
    elif model_type.upper() == "BPL":
        electron_spectrum = BrokenPowerLaw(
            amplitude=amplitude,
            e_0=1 * u.TeV,
            e_break=10 ** params_med["log10(break_energy)"] * u.TeV,
            alpha_1=params_med["alpha"],
            alpha_2=params_med["alpha"] + params_med["beta"],
        )
    else:
        raise ValueError("model_type must be 'ECPL' or 'BPL'")

    # --- Radiative models ---
    SYN = Synchrotron(electron_spectrum, B=B)
    IC = InverseCompton(electron_spectrum, seed_photon_fields=seed_photon_fields)
    Brem = Bremsstrahlung(electron_spectrum, n0=gas_density)

    sed_syn = SYN.sed(spectrum_energy, distance=distance)
    sed_ic = IC.sed(spectrum_energy, distance=distance)
    sed_brem = Brem.sed(spectrum_energy, distance=distance)
    sed = sed_syn + sed_ic + sed_brem

    # ---- Compute and print gamma-ray quantities with approximate errors ----
    p_med, p_low, p_high = _extract_param_sets(params_dict)

    def build_leptonic_model_local(pdict):
        amplitude = 10 ** pdict["log10(norm)"] / u.eV
        Bloc = B
        if fit_B:
            if "log10(B_uG)" in pdict:
                Bloc = (10 ** pdict["log10(B_uG)"]) * u.uG
            elif "log10(B)" in pdict:
                Bloc = (10 ** pdict["log10(B)"]) * u.uG

        if model_type.upper() == "ECPL":
            spec = ExponentialCutoffPowerLaw(
                amplitude=amplitude,
                e_0=1 * u.TeV,
                alpha=pdict["alpha"],
                e_cutoff=10 ** pdict["log10(cutoff_energy)"] * u.TeV,
                beta=1.0,
            )
        elif model_type.upper() == "BPL":
            spec = BrokenPowerLaw(
                amplitude=amplitude,
                e_0=1 * u.TeV,
                e_break=10 ** pdict["log10(break_energy)"] * u.TeV,
                alpha_1=pdict["alpha"],
                alpha_2=pdict["alpha"] + pdict["beta"],
            )
        else:
            raise ValueError("model_type must be 'ECPL' or 'BPL'")

        syn = Synchrotron(spec, B=Bloc)
        ic = InverseCompton(spec, seed_photon_fields=seed_photon_fields)
        brem = Bremsstrahlung(spec, n0=gas_density)
        return spec, syn, ic, brem, Bloc

    spec_med, SYN_med, IC_med, Brem_med, B_med = build_leptonic_model_local(p_med)
    spec_low, SYN_low, IC_low, Brem_low, B_low = build_leptonic_model_local(p_low)
    spec_high, SYN_high, IC_high, Brem_high, B_high = build_leptonic_model_local(p_high)

    Eref = np.array([1.0]) * u.TeV

    total_med = IC_med.flux(Eref, distance=distance)[0] + Brem_med.flux(Eref, distance=distance)[0]
    total_low = IC_low.flux(Eref, distance=distance)[0] + Brem_low.flux(Eref, distance=distance)[0]
    total_high = IC_high.flux(Eref, distance=distance)[0] + Brem_high.flux(Eref, distance=distance)[0]

    dnde_med = total_med.to("1/(TeV cm2 s)")
    dnde_low = total_low.to("1/(TeV cm2 s)")
    dnde_high = total_high.to("1/(TeV cm2 s)")

    sed_med = (IC_med.sed(Eref, distance=distance)[0] + Brem_med.sed(Eref, distance=distance)[0]).to("TeV cm-2 s-1")
    sed_low = (IC_low.sed(Eref, distance=distance)[0] + Brem_low.sed(Eref, distance=distance)[0]).to("TeV cm-2 s-1")
    sed_high = (IC_high.sed(Eref, distance=distance)[0] + Brem_high.sed(Eref, distance=distance)[0]).to("TeV cm-2 s-1")

    # Numerical break estimate from IC spectrum
    Ebreak_med, _, _ = estimate_gamma_break_from_model(
        IC_med, distance=distance, energy_range=(1e-4, 1e2), ngrid=500, use_sed=False
    )
    Ebreak_low, _, _ = estimate_gamma_break_from_model(
        IC_low, distance=distance, energy_range=(1e-4, 1e2), ngrid=500, use_sed=False
    )
    Ebreak_high, _, _ = estimate_gamma_break_from_model(
        IC_high, distance=distance, energy_range=(1e-4, 1e2), ngrid=500, use_sed=False
    )

    # ---- Spectral indices ----
    if model_type.upper() == "BPL":
        # Electron indices
        p1_med = p_med["alpha"]
        p2_med = p_med["alpha"] + p_med["beta"]

        p1_low = p_low["alpha"]
        p2_low = p_low["alpha"] + p_low["beta"]

        p1_high = p_high["alpha"]
        p2_high = p_high["alpha"] + p_high["beta"]

        # Thomson-regime IC mapping: Gamma = (p + 1)/2
        gamma1_med = 0.5 * (p1_med + 1.0)
        gamma2_med = 0.5 * (p2_med + 1.0)

        gamma1_low = 0.5 * (p1_low + 1.0)
        gamma2_low = 0.5 * (p2_low + 1.0)

        gamma1_high = 0.5 * (p1_high + 1.0)
        gamma2_high = 0.5 * (p2_high + 1.0)

        gamma_at_1tev_med = _gamma_index_from_model(IC_med, distance)
        gamma_at_1tev_low = _gamma_index_from_model(IC_low, distance)
        gamma_at_1tev_high = _gamma_index_from_model(IC_high, distance)
    else:
        gamma_at_1tev_med = _gamma_index_from_model(IC_med, distance)
        gamma_at_1tev_low = _gamma_index_from_model(IC_low, distance)
        gamma_at_1tev_high = _gamma_index_from_model(IC_high, distance)

        gamma1_med = gamma_at_1tev_med
        gamma1_low = gamma_at_1tev_low
        gamma1_high = gamma_at_1tev_high

        gamma2_med = np.nan
        gamma2_low = np.nan
        gamma2_high = np.nan

    # ---- Total electron energy above 1 GeV ----
    We_med = compute_We_from_spectrum(spec_med, Eemin=1.0 * u.GeV, Eemax=100.0 * u.TeV)
    We_low = compute_We_from_spectrum(spec_low, Eemin=1.0 * u.GeV, Eemax=100.0 * u.TeV)
    We_high = compute_We_from_spectrum(spec_high, Eemin=1.0 * u.GeV, Eemax=100.0 * u.TeV)

    print("\n[Gamma-ray spectrum (Leptonic IC + Bremsstrahlung)]")
    print(
        f"dN/dE (1 TeV): {dnde_med:.3e} "
        f"(-{(dnde_med - dnde_low):.3e}, +{(dnde_high - dnde_med):.3e})"
    )
    print(
        f"E^2 dN/dE (1 TeV): {sed_med:.3e} "
        f"(-{(sed_med - sed_low):.3e}, +{(sed_high - sed_med):.3e})"
    )
    print(
        f"Estimated gamma-ray break from modeled IC spectrum: {Ebreak_med:.3f} "
        f"(-{(Ebreak_med - Ebreak_low):.3f}, +{(Ebreak_high - Ebreak_med):.3f})"
    )

    if model_type.upper() == "BPL":
        print(
            f"Gamma-ray spectral index below break Γ1: {gamma1_med:.3f} "
            f"(-{(gamma1_med - gamma1_low):.3f}, +{(gamma1_high - gamma1_med):.3f})"
        )
        print(
            f"Gamma-ray spectral index above break Γ2: {gamma2_med:.3f} "
            f"(-{(gamma2_med - gamma2_low):.3f}, +{(gamma2_high - gamma2_med):.3f})"
        )
        print(
            f"Local gamma-ray index @10 TeV: {gamma_at_1tev_med:.3f} "
            f"(-{(gamma_at_1tev_med - gamma_at_1tev_low):.3f}, +{(gamma_at_1tev_high - gamma_at_1tev_med):.3f})"
        )
    else:
        print(
            f"Gamma-ray index @10 TeV: {gamma_at_1tev_med:.3f} "
            f"(-{(gamma_at_1tev_med - gamma_at_1tev_low):.3f}, +{(gamma_at_1tev_high - gamma_at_1tev_med):.3f})"
        )

    print(
        f"Total electron energy We (>1 GeV): {We_med:.3e} "
        f"(-{(We_med - We_low):.3e}, +{(We_high - We_med):.3e})"
    )
    if fit_B:
        print(
            f"Magnetic field B: {B_med.to(u.uG):.3f} "
            f"(-{(B_med - B_low).to(u.uG):.3f}, +{(B_high - B_med).to(u.uG):.3f})"
        )

    # --- Plot model ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(
        spectrum_energy.to("TeV").value,
        sed_syn.to("TeV cm-2 s-1").value,
        lw=2,
        color="grey",
        ls="--",
        alpha=alpha,
        label="Synchrotron",
    )

    ax.loglog(
        spectrum_energy.to("TeV").value,
        sed_ic.to("TeV cm-2 s-1").value,
        lw=2,
        color=color,
        ls="--",
        alpha=alpha,
        label="IC",
    )

    ax.loglog(
        spectrum_energy.to("TeV").value,
        sed_brem.to("TeV cm-2 s-1").value,
        lw=2,
        color="peru",
        ls="--",
        alpha=alpha,
        label="Bremsstrahlung",
    )

    ax.loglog(
        spectrum_energy.to("TeV").value,
        sed.to("TeV cm-2 s-1").value,
        lw=2,
        color="black",
        ls=linestyle,
        alpha=alpha,
        label="Total",
    )

    if data_dict is not None:
        colors = ["blue", "green", "indigo", "darkorange", "red", "teal"]
        markers = ["o", "s", "^", "D", "v", "p"]

        for i, key in enumerate(data_dict.keys()):
            plot_data(
                ax,
                data_dict[key],
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                energy_unit="TeV",
                flux_unit="TeV cm-2 s-1",
                label=key,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(energy_range)
    ax.set_ylim(1e-14, 1e-9)
    ax.set_xlabel(r"Photon energy [TeV]", fontsize=12)
    ax.set_ylabel(r"$E^2 \, dN/dE$ [TeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
    ax.grid(which="both", color="gray", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", ncol=2)

    if show:
        plt.tight_layout()
        plt.show()

    return ax
