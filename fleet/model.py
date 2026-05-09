import scipy.special as sp
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import emcee
import warnings

# Central wavelengths for ZTF and LSST filters
ztf_refs = {'g': 4746.48,
            'r': 6366.38,
            'i': 7829.03}

lsst_refs = {'u': 3751.20,
             'g': 4740.66,
             'r': 6172.34,
             'i': 7500.97,
             'z': 8678.90,
             'y': 9711.82}

generic_refs = {'u': 3751.20,
                'g': 4740.66,
                'r': 6172.34,
                'i': 7500.97,
                'z': 8678.90,
                'y': 9711.82,
                'U': 3659.88,
                'B': 4380.74,
                'V': 5445.43,
                'R': 6411.47,
                'I': 7982.09}

# Define colors for each filter
filter_colors = {
    'u': 'navy',
    'g': 'g',
    'r': 'r',
    'i': 'maroon',
    'z': 'saddlebrown',
    'y': 'indigo',
    'U': 'navy',
    'B': 'darkcyan',
    'V': 'lawngreen',
    'R': 'r',
    'I': 'maroon',
}

# Define priors
priors = {
    'lc_width': (-0.6, 0.0),
    'lc_decline': (0.01, 1.0),
    'phase_offset': (-40.0, 40.0),
    'mag_offset': (5.0, 30.0),
    'initial_temp': (1000.0, 9000.0),
    'cooling_rate': (1.0, 3000.0)
}

# Gaussian prior parameters
lc_decline_mean = 0.3
lc_decline_std = 0.2
cooling_rate_mean = 200.0
cooling_rate_std = 100.0

#######
# Define functions for light curve models
#######
def normalize_label(value):
    """Normalize telescope/source/instrument labels used to infer filter systems."""
    if value is None:
        return ''
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip().lower()


def normalize_filter_name(filter_name):
    """Normalize a filter name while preserving standard uppercase Johnson-Cousins bands."""
    if isinstance(filter_name, bytes):
        filter_name = filter_name.decode()
    filter_name = str(filter_name).strip()

    if filter_name in ztf_refs or filter_name in lsst_refs or filter_name in generic_refs:
        return filter_name

    lower_name = filter_name.lower()
    if lower_name in ztf_refs or lower_name in lsst_refs or lower_name in generic_refs:
        return lower_name

    upper_name = filter_name.upper()
    if upper_name in generic_refs:
        return upper_name

    return filter_name

def get_filter_cenwave(filter_name, telescope=None, source=None, instrument=None):
    """
    Return the central wavelength for one photometric row.

    ZTF and Rubin/LSST use slightly different g/r/i effective wavelengths,
    so mixed-survey light curves should choose the reference per row rather
    than once per whole table.
    """
    filter_name = normalize_filter_name(filter_name)
    labels = {normalize_label(telescope), normalize_label(source), normalize_label(instrument)}

    if labels.intersection({'rubin', 'lsst'}):
        refs = lsst_refs
    elif 'ztf' in labels:
        refs = ztf_refs
    elif 'alerce' in labels and filter_name in ztf_refs:
        # Backward-compatible fallback for old ZTF tables with Source='Alerce'
        # and no Telescope/Instrument column.
        refs = ztf_refs
    else:
        refs = generic_refs

    if filter_name in refs:
        return refs[filter_name]
    if filter_name in generic_refs:
        return generic_refs[filter_name]
    if filter_name in lsst_refs:
        return lsst_refs[filter_name]
    if filter_name in ztf_refs:
        return ztf_refs[filter_name]

    raise KeyError(f"No central wavelength defined for filter '{filter_name}'")


def get_table_cenwaves(input_table):
    """Return central wavelengths for every row in an input light-curve table."""
    telescopes = input_table['Telescope'] if 'Telescope' in input_table.colnames else [None] * len(input_table)
    sources = input_table['Source'] if 'Source' in input_table.colnames else [None] * len(input_table)
    instruments = input_table['Instrument'] if 'Instrument' in input_table.colnames else [None] * len(input_table)

    return np.array([
        get_filter_cenwave(filter_name, telescope=telescope, source=source, instrument=instrument)
        for filter_name, telescope, source, instrument in zip(input_table['Filter'], telescopes, sources, instruments)
    ])


def get_filter_wavelengths(filters_used, input_table=None):
    """
    Return one representative central wavelength per plotted filter.

    If a table with a Cenwave column is supplied, use the median wavelength
    actually present for that filter. This keeps mixed ZTF/Rubin light curves
    from being forced entirely onto one survey's filter system.
    """
    wavelengths = []

    if input_table is not None and len(input_table) > 0 and 'Filter' in input_table.colnames:
        table_filters = np.array([normalize_filter_name(filter_name) for filter_name in input_table['Filter']])
        if 'Cenwave' in input_table.colnames:
            table_cenwaves = np.array(input_table['Cenwave'], dtype=float)
        else:
            table_cenwaves = get_table_cenwaves(input_table)

        for filter_name in filters_used:
            clean_filter = normalize_filter_name(filter_name)
            matched = table_filters == clean_filter
            matched_cenwaves = table_cenwaves[matched]
            matched_cenwaves = matched_cenwaves[np.isfinite(matched_cenwaves)]
            if len(matched_cenwaves) > 0:
                wavelengths.append(np.nanmedian(matched_cenwaves))
            else:
                wavelengths.append(get_filter_cenwave(clean_filter))
        return np.array(wavelengths)

    return np.array([get_filter_cenwave(normalize_filter_name(filter_name)) for filter_name in filters_used])


def linex(phase, lc_width, lc_decline, phase_offset, mag_offset):
    '''
    Function to generate a light curve based on a linear-exponential model.

    Parameters
    ----------
    phase : float or array-like
        Phase (time) in days
    lc_width : float
        Light curve width parameter (controls the width of the light curve)
        More negative numbers result in a narrower light curve.
    lc_decline : float
        Light curve asymmetry parameter (controls the decline rate)
        Larger values result in a steeper decline.
    phase_offset : float
        Reference phase offset in days
    mag_offset : float
        Brightest magnitude offset (shifts the light curve vertically)

    Returns
    -------
    mag_array : float or array-like
        Magnitude at the given phase, calculated using the linear-exponential model.
    '''

    slope = np.exp(lc_width * (phase - phase_offset)) - lc_width * lc_decline * (phase - phase_offset)
    mag_array = slope - 1 + mag_offset
    return mag_array


def d_linex(lc_width, lc_decline, phase_offset):
    """
    Function that calculates the derivative of the linear-exponential model
    to determine its minimum.

    Parameters
    ----------
    lc_width : float
        Light curve width parameter (controls the width of the light curve)
        More negative numbers result in a narrower light curve.
    lc_decline : float
        Light curve asymmetry parameter (controls the decline rate)
        Larger values result in a steeper decline.
    phase_offset : float
        Reference phase offset in days

    Returns
    -------
    phase_peak : float
        The minimum value of the linear-exponential model.
    """
    # Solve the equation for phase
    phase_peak = phase_offset + np.log(lc_decline) / lc_width
    return phase_peak


def bb_magnitude(wavelengths, temperature, bol_ref=1):
    """
    Calculate the luminosity at a specified wavelength or wavelengths
    given a blackbody temperature and a reference luminosity. The output
    is in roughly arbitrary units, and should not be interpreted as a
    physical luminosity.

    Parameters
    ----------
    wavelengths : np.array
        Central wavelengths in angstroms
    temperature : float
        Blackbody temperature in Kelvin
    bol_ref : float, default=1
        Reference luminosity in erg/s/AA

    Returns
    -------
    AB_mag : np.array
        AB magnitude at the specified wavelengths
    """

    # Calculate radius using Stefan-Boltzmann
    R = np.sqrt(bol_ref / (4*np.pi*5.67e-5*temperature**4))

    # Constants
    h = 6.62607E-27  # Planck Constant in cm^2 * g / s
    c = 2.99792458E10  # Speed of light in cm/s
    k_B = 1.38064852E-16  # Boltzmann Constant in cm^2 * g / s^2 / K

    # Convert wavelength to cm
    lam_cm = wavelengths * 1E-8

    # Calculate exponential term
    exponential = (h * c) / (lam_cm * k_B * temperature)

    # Constant factor in B_lam calculation
    prefactor = (2 * np.pi * h * c ** 2) / (lam_cm ** 5)

    # Calculate B_lam with better numerical stability
    # Use log-space for large exponentials to avoid overflow
    log_B_lam = np.log(prefactor) - np.log(np.expm1(exponential))
    log_B_lam = np.where(
        exponential > 100,
        np.log(prefactor) - exponential,  # log(prefactor * exp(-exponential))
        log_B_lam
    )

    # Multiply by the surface area
    A = 4*np.pi*R**2

    # Output luminosity in erg / s / Angstrom (in log space)
    log_luminosity = log_B_lam + np.log(A) - np.log(1E8)

    # Calculate corresponding magnitude (avoiding numerical issues)
    # Convert from log space to magnitude
    AB_mag = -2.5 * log_luminosity / np.log(10)

    return AB_mag


def model_mag(phases, wavelengths, lc_width, lc_decline, phase_offset, mag_offset,
              initial_temp, cooling_rate):
    """
    Combined function that calculates the magnitude of a supernova light curve
    at a given phase and wavelength, taking into account the temperature
    evolution and assumes a blackbody spectrum. The code also assumes the temperature
    decreases exponentially with time. The minimum temperature is 100K

    Parameters
    ----------
    phases : float or array-like
        Light curve phase in days
    wavelengths : float or array-like
        Wavelengths in angstroms
    lc_width : float
        Light curve width parameter
    lc_decline : float
        Light curve asymmetry parameter
    phase_offset : float
        Phase offset in time
    mag_offset : float
        Magnitude offset (brightest magnitude)
    initial_temp : float
        Initial temperature in Kelvin
    cooling_rate : float
        Cooling rate in Kelvin/day

    Returns
    -------
    magnitude : float or array-like
        AB magnitude at the given phases and wavelengths
    """
    # Calculate the temperature as a function of phase
    temperature = initial_temp * np.exp(-(phases-phase_offset)/cooling_rate) + 100

    # Calculate the overall brightness evolution using linex
    base_mag = linex(phases, lc_width, lc_decline, phase_offset, mag_offset)

    # Calculate the SED shape at the current temperature
    # We'll normalize the SED so it doesn't affect the overall brightness
    sed_shape = bb_magnitude(wavelengths, temperature)

    # Calculate the reference SED shape at the reference wavelength (r-band)
    sed_ref = bb_magnitude(6172.34, initial_temp)

    # Adjust the shape so the mean matches the base magnitude
    magnitude = base_mag + (sed_shape - sed_ref)

    return magnitude

#######
# Define likelihood functions for MCMC fitting
#######


def lnlike_ul(obs_mags, model_mags, err_mags, obs_limits):
    """
    Function to calculate the log likelihood of a model, but accounting
    for upper limits in the data.

    Parameters
    ----------
    obs_mags : np.array
        Observed magnitudes
    model_mags : np.array
        Model magnitudes
    err_mags : np.array
        Uncertainties in the observed magnitudes
    obs_limits : np.array
        Boolean array of upper limits

    Returns
    -------
    ln_like : float
        Log likelihood
    """

    ln_like = 0.0

    # Handle detections as usual
    is_detection = ~obs_limits
    if np.any(is_detection):
        det_error = obs_mags[is_detection] - model_mags[is_detection]
        det_weight = 1.0 / (err_mags[is_detection] ** 2)
        ln_like -= 0.5 * np.sum(det_weight * det_error ** 2)

        # Include normalization term for the detections
        ln_like -= 0.5 * np.sum(np.log(2.0 * np.pi * err_mags[is_detection] ** 2))

    # Handle upper limits
    if np.any(obs_limits):
        # For each upper limit, calculate the integral term
        for j in np.where(obs_limits)[0]:
            # Calculate how many sigma the model is from the limit
            z = (model_mags[j] - obs_mags[j]) / np.abs(err_mags[j])

            # Use the complementary error function to calculate the integral term
            # of Equation 8 in https://arxiv.org/pdf/1210.0285
            prob = 0.5 * (1 + sp.erf(z / np.sqrt(2)))
            # Add the log of this probability to the likelihood
            ln_like += np.log(prob) if prob > 0 else -np.inf

    return ln_like


def lnlike_single(theta, phases, obs_mags, err_mags, obs_limits, lc_decline):
    """
    Likelihood function for fitting a single exponential light curve,
    assuming a fixed decline rate (lc_decline).

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, phase_offset, mag_offset)
    phases : np.array
        Phase values in days
    obs_mags : np.array
        Observed magnitudes
    err_mags : np.array
        Observed uncertainties
    obs_limits : np.array
        Boolean array of upper limits
    lc_decline : float
        Light curve asymmetry parameter

    Returns
    -------
    float
        Log likelihood value
    """
    lc_width, phase_offset, mag_offset = theta

    # Fit the light curve using the linear-exponential model
    model_mags = linex(phases, lc_width, lc_decline, phase_offset, mag_offset)

    # Calculate the likelihood
    ln_like = lnlike_ul(obs_mags, model_mags, err_mags, obs_limits)
    return ln_like


def lnlike_double(theta, phases, obs_mags, err_mags, obs_limits):
    """
    Likelihood function for fitting a light curve based on the old
    linear-exponential model.

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, lc_decline, phase_offset, mag_offset)
    phases : np.array
        Phase values in days
    obs_mags : np.array
        Observed magnitudes
    err_mags : np.array
        Observed uncertainties
    obs_limits : np.array
        Boolean array of upper limits

    Returns
    -------
    float
        Log likelihood value
    """
    lc_width, lc_decline, phase_offset, mag_offset = theta

    # Fit the light curve using the linear-exponential model
    model_mags = linex(phases, lc_width, lc_decline, phase_offset, mag_offset)

    # Calculate the likelihood
    ln_like = lnlike_ul(obs_mags, model_mags, err_mags, obs_limits)
    return ln_like


def lnlike_full(theta, phases, wavelengths, obs_mags, err_mags, obs_limits):
    """
    Likelihood function for fitting a light curve using the full model,
    which includes the temperature evolution and blackbody spectrum.

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, lc_decline, phase_offset, mag_offset,
        initial_temp, cooling_rate)
    phases : np.array
        Phase values in days
    wavelengths : np.array
        Wavelength values in angstroms
    obs_mags : np.array
        Observed magnitudes
    err_mags : np.array
        Observed uncertainties
    obs_limits : np.array
        Boolean array of upper limits

    Returns
    -------
    float
        Log likelihood value
    """
    lc_width, lc_decline, phase_offset, mag_offset, initial_temp, cooling_rate = theta

    # Fit the light curve using the linear-exponential model
    model_mags = model_mag(phases, wavelengths, lc_width, lc_decline, phase_offset, mag_offset,
                           initial_temp, cooling_rate)

    # Calculate the likelihood
    ln_like = lnlike_ul(obs_mags, model_mags, err_mags, obs_limits)
    return ln_like

#######
# Define prior functions for MCMC fitting
#######


def lnprior_single(theta):
    """
    Log prior function for the single exponential model.

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, phase_offset, mag_offset)

    Returns
    -------
    float
        Log prior value
    """
    lc_width, phase_offset, mag_offset = theta

    # Define the prior ranges for each parameter
    if priors['lc_width'][0] < lc_width < priors['lc_width'][1] and \
       priors['phase_offset'][0] < phase_offset < priors['phase_offset'][1] and \
       priors['mag_offset'][0] < mag_offset < priors['mag_offset'][1]:
        return 0.0
    return -np.inf


def lnprior_double(theta):
    """
    Log prior function for the linear-exponential model.

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, lc_decline, phase_offset, mag_offset)

    Returns
    -------
    float
        Log prior value
    """
    lc_width, lc_decline, phase_offset, mag_offset = theta

    # Define the prior ranges for each parameter
    if not (priors['lc_width'][0] < lc_width < priors['lc_width'][1] and
            priors['lc_decline'][0] < lc_decline < priors['lc_decline'][1] and
            priors['phase_offset'][0] < phase_offset < priors['phase_offset'][1] and
            priors['mag_offset'][0] < mag_offset < priors['mag_offset'][1]):
        return -np.inf

    # If within bounds, compute log probability for Gaussian priors
    ln_prior = 0.0

    # Gaussian prior for lc_decline
    ln_prior += -0.5 * ((lc_decline - lc_decline_mean) / lc_decline_std)**2

    return ln_prior


def lnprior_full(theta):
    """
    Log prior function for the full model, which includes temperature
    evolution. Uses flat priors for most parameters but Gaussianpriors
    for lc_decline and cooling_rate.

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, lc_decline, phase_offset, mag_offset,
        initial_temp, cooling_rate)

    Returns
    -------
    float
        Log prior value
    """
    lc_width, lc_decline, phase_offset, mag_offset, initial_temp, cooling_rate = theta

    # First check if all parameters are within the allowed ranges
    if not (priors['lc_width'][0] < lc_width < priors['lc_width'][1] and
            priors['lc_decline'][0] < lc_decline < priors['lc_decline'][1] and
            priors['phase_offset'][0] < phase_offset < priors['phase_offset'][1] and
            priors['mag_offset'][0] < mag_offset < priors['mag_offset'][1] and
            priors['initial_temp'][0] < initial_temp < priors['initial_temp'][1] and
            priors['cooling_rate'][0] < cooling_rate < priors['cooling_rate'][1]):
        return -np.inf

    # If within bounds, compute log probability for Gaussian priors
    ln_prior = 0.0

    # Gaussian prior for lc_decline
    ln_prior += -0.5 * ((lc_decline - lc_decline_mean) / lc_decline_std)**2

    # Gaussian prior for cooling_rate
    ln_prior += -0.5 * ((cooling_rate - cooling_rate_mean) / cooling_rate_std)**2

    return ln_prior

########
# Define posterior functions for MCMC fitting
########


def lnprob_single(theta, phases, obs_mags, err_mags, obs_limits, lc_decline):
    """
    Posterior function for the single exponential model.

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, phase_offset, mag_offset)
    phases : np.array
        Phase values in days
    obs_mags : np.array
        Observed magnitudes
    err_mags : np.array
        Observed uncertainties
    obs_limits : np.array
        Boolean array of upper limits
    lc_decline : float
        Decline rate for the light curve

    Returns
    -------
    float
        Log posterior value
    """
    lp = lnprior_single(theta)
    if not np.isfinite(lp):
        return -np.inf  # Outside prior range
    return lp + lnlike_single(theta, phases, obs_mags, err_mags, obs_limits, lc_decline)


def lnprob_double(theta, phases, obs_mags, err_mags, obs_limits):
    """
    Posterior function for the linear-exponential model.

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, lc_decline, phase_offset, mag_offset)
    phases : np.array
        Phase values in days
    obs_mags : np.array
        Observed magnitudes
    err_mags : np.array
        Observed uncertainties
    obs_limits : np.array
        Boolean array of upper limits

    Returns
    -------
    float
        Log posterior value
    """
    lp = lnprior_double(theta)
    if not np.isfinite(lp):
        return -np.inf  # Outside prior range
    return lp + lnlike_double(theta, phases, obs_mags, err_mags, obs_limits)


def lnprob_full(theta, phases, wavelengths, obs_mags, err_mags, obs_limits):
    """
    Posterior function for the full model, which includes temperature evolution.

    Parameters
    ----------
    theta : list
        Parameters for the model (lc_width, lc_decline, phase_offset, mag_offset,
        initial_temp, cooling_rate)
    phases : np.array
        Phase values in days
    wavelengths : np.array
        Wavelength values in angstroms
    obs_mags : np.array
        Observed magnitudes
    err_mags : np.array
        Observed uncertainties
    obs_limits : np.array
        Boolean array of upper limits

    Returns
    -------
    float
        Log posterior value
    """
    lp = lnprior_full(theta)
    if not np.isfinite(lp):
        return -np.inf  # Outside prior range
    return lp + lnlike_full(theta, phases, wavelengths, obs_mags, err_mags, obs_limits)

########
# Define other functions now
#########


def calc_chi2(data, model, n_parameters, sigma, limits):
    """"
    Calculate the reduced chi squared of a data set and a model for
    a given number of parameters.

    Parameters
    ----------
    data : np.array
        Observed data values
    model : np.array
        Model values
    n_parameters : int
        Number of free parameters in the model
    sigma : np.array
        Uncertainties in the data (errors)
    limits : np.array
        Boolean array indicating upper limits

    Returns
    -------
    chisq : float
        Reduced chi squared value
    """
    # Ensure sigma is not zero or negative
    sigma[sigma <= 0] = np.nan

    # Select only the values that are not upper limits
    data = data[~limits]
    model = model[~limits]
    sigma = sigma[~limits]

    # Calculate the chi squared value
    chisq = np.nansum(((data-model)/sigma)**2.0)
    nu = data.size-n_parameters-1.0

    # Ensure we don't divide by zero or negative degrees of freedom
    if nu > 0:
        return chisq / nu
    else:
        return chisq


def format_data(input_table, default_err=0.1, clean=True, remove_ul=False,
                remove_late_ul=True, n_sigma_limit=3, phase_min=-200, phase_max=75):
    """
    Format the data for plotting and analysis in a format the FLEET will
    like.

    Parameters
    ----------
    input_table : astropy.table.Table
        Input data table containing photometry
    clean : bool, default True
        Remove ignored data and nan values?
    remove_ul : bool, default False
        Remove upper limits from fit?
    remove_late_ul : bool, default True
        Remove upper limits after the first detection?
        (Highly recommended since spurious upper limits can bias the fit)
    n_sigma_limit : int, default 3
        Default upper limit sigma to calculate
        the error on the limit
    phase_min : float, default -200
        Minimum phase in days from first detection to consider
    phase_max : float, default 75
        Maximum phase in days from first detection to consider

    Returns
    -------
    output_table : astropy.table.Table
        Formatted data table with additional columns for central wavelength and phase
    bright_mjd : float
        MJD of the brightest observation
    first_mjd : float
        MJD of the first observation
    """

    # Add central wavelength to each row
    if 'Cenwave' not in input_table.colnames:
        input_table['Cenwave'] = get_table_cenwaves(input_table)

    # Filter out unwanted rows based on UL and Ignore flags
    if clean:
        output_table = input_table[(input_table['Ignore'] == 'False') & np.isfinite(input_table['Mag']) &
                                   np.isfinite(input_table['MJD'])]
    else:
        output_table = input_table

    # Remove upper limits, or assign them a reasonable error
    if remove_ul:
        output_table = output_table[(output_table['UL'] == 'False')]
    else:
        upperlimits = output_table['UL'] == 'True'
        output_table['MagErr'][upperlimits] = np.round(2.5 * np.log10(1 + 1/n_sigma_limit), 4)

    # Remove late-time upper limits
    detections_for_limits = output_table[output_table['UL'] == 'False']
    if len(detections_for_limits) == 0:
        print("No detections found after filtering.")
        return None, None, None

    if remove_late_ul:
        first_mjd = np.nanmin(detections_for_limits['MJD'])
        output_table = output_table[~((output_table['UL'] == 'True') & (output_table['MJD'] > first_mjd))]

    # Continue only if len(output_table) > 0
    if len(output_table) == 0:
        print("No valid data points found after filtering.")
        return None, None, None
    else:
        # Override any missing errors with the default value
        output_table['MagErr'] = np.where(np.isnan(output_table['MagErr']),
                                          default_err,
                                          output_table['MagErr'])
        output_table['MagErr'] = np.where(output_table['MagErr'] <= 0,
                                          default_err,
                                          output_table['MagErr'])

        # Calculate the phase of each observation relative to the brightest observation
        dets = (output_table['UL'] == 'False') & (output_table['Ignore'] == 'False')
        first_mjd = np.nanmin(output_table[dets]['MJD'])
        output_table['Phase_boom'] = output_table['MJD'] - first_mjd

        # Set Ignore = True for data outside the specified phase range
        output_table['Ignore'] = np.where((output_table['Phase_boom'] < phase_min) |
                                          (output_table['Phase_boom'] > phase_max),
                                          'True', output_table['Ignore'])

        # Calculate the brightest observation time, after filtering
        dets = (output_table['UL'] == 'False') & (output_table['Ignore'] == 'False')
        bright_mjd = output_table[dets]['MJD'][np.nanargmin(output_table[dets]['Mag'])]
        output_table['Phase_peak'] = output_table['MJD'] - bright_mjd

        return output_table, bright_mjd, first_mjd


def plot_trace(param_chain, param_values, param_values_log, min_val, max_val,
               title_name, param, log, n_steps, burn_in, output_dir, repeats,
               object_name):
    '''
    This function plots the trace of a parameter chain.

    Parameters
    ----------
    param_chain : np.ndarray
        The chain of the parameter with shape (nwalkers, nsteps).
    param_values : np.ndarray
        The median, upper and lower limits of the parameter.
    param_values_log : np.ndarray
        The median, upper and lower limits of the log of the parameter.
    min_val : float
        The minimum value of the parameter.
    max_val : float
        The maximum value of the parameter.
    title_name : str
        The name of the parameter.
    param : str
        The name of the parameter.
    log : bool
        Whether the parameter is in log scale.
    n_steps : int
        The number of steps in the chain.
    burn_in : float
        The fraction of steps to burn in.
    output_dir : str
        The directory to save the plot.
    repeats: int
        Number of times the emcee was repeated
    '''

    # Average walker position
    print(f'Plotting {param} Trace...')
    Averageline = np.average(param_chain.T, axis=1)

    # Plot Trace
    plt.subplots_adjust(wspace=0)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax0.axhline(param_values[0], color='r', lw=2.0, linestyle='--', alpha=0.75)
    ax0.axhline(param_values[0] - param_values[2], color='r', lw=1.0, linestyle='--', alpha=0.50)
    ax0.axhline(param_values[0] + param_values[1], color='r', lw=1.0, linestyle='--', alpha=0.50)
    ax0.plot(Averageline, lw=1.0, color='b', alpha=0.75)
    ax0.plot(param_chain.T, '-', color='k', alpha=0.2, lw=0.5)
    plt.xlim(0, (repeats * n_steps) - 1)
    if log:
        plt.ylim(np.log10(min_val), np.log10(max_val))
    else:
        plt.ylim(min_val, max_val)

    title_string = r"$%s^{+%s}_{-%s}$" % (np.round(param_values[0], 5), np.round(param_values[1], 5),
                                          np.round(param_values[2], 5))
    if log:
        title_string += '  = log(' + r"$%s^{+%s}_{-%s}$" % (np.round(param_values_log[0], 5),
                                                            np.round(param_values_log[1], 5),
                                                            np.round(param_values_log[2], 5)) + ')'
    plt.title(title_string)
    plt.ylabel(title_name)
    plt.xlabel("Step")

    # Plot Histogram
    ax1 = plt.subplot(gs[1])
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                    labeltop=False, labelright=False, labelbottom=False)
    if log:
        plt.ylim(np.log10(min_val), np.log10(max_val))
    else:
        plt.ylim(min_val, max_val)
    ax1.hist(np.ndarray.flatten(param_chain[:, -int(n_steps*(1-burn_in)):]), bins='auto',
             orientation="horizontal", color='k', range=(min_val, max_val))
    ax1.axhline(Averageline[-1], color='b', lw=1.0, linestyle='-', alpha=0.75)
    ax1.axhline(param_values[0], color='r', lw=2.0, linestyle='--', alpha=0.75)
    ax1.axhline(param_values[0] - param_values[2], color='r', lw=1.0, linestyle='--', alpha=0.50)
    ax1.axhline(param_values[0] + param_values[1], color='r', lw=1.0, linestyle='--', alpha=0.50)

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{object_name}_{param}_Trace.jpg")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close('all')


def plot_model(input_table, last_samples, best_params, model, object_name,
               plot_dir='plots', default_err=0.1, n_sigma_limit=3):
    """
    Plot the model and the data.

    Parameters
    ----------
    input_table : astropy.table.Table
        Input data table containing photometry
    last_samples : np.ndarray
        MCMC samples of the parameters
    model : str
        Model to use for plotting. Options are 'single', 'double', or 'full'.
    object_name : str
        Name of the object being analyzed
    plot_dir : str, default 'plots'
        Directory to save the plots
    default_err : float, default 0.1
        Default error value for the data
    n_sigma_limit : int, default 3
        Default upper limit sigma to calculate the error on the limit
    """
    # Make sure folder exists
    os.makedirs(plot_dir, exist_ok=True)

    # Create a grid of phases and wavelengths for plotting
    phases = np.linspace(-50, 150, 100)

    # Clean data
    output_table, bright_mjd, first_mjd = format_data(input_table, default_err=default_err,
                                                      n_sigma_limit=n_sigma_limit)

    if output_table is None or len(output_table) == 0:
        print("No valid data available for model plot.")
        return

    # Define representative filter wavelengths from the data itself.
    filters_used = np.unique(output_table['Filter'])
    wavelengths = get_filter_wavelengths(filters_used, output_table)

    for filter_name, filter_wave in zip(filters_used, wavelengths):
        mask = (output_table['Filter'] == filter_name)
        det = (output_table['UL'] == 'False')

        # Plot the data
        plt.errorbar(output_table['Phase_peak'][mask & det], output_table['Mag'][mask & det], alpha=0.8,
                     yerr=output_table['MagErr'][mask & det], fmt='o',
                     color=filter_colors[filter_name], label=filter_name)
        plt.errorbar(output_table['Phase_peak'][mask & ~det], output_table['Mag'][mask & ~det], alpha=0.4,
                     fmt='v', color=filter_colors[filter_name])

        # Plot the model
        if model == 'full':
            model_data = model_mag(phases, filter_wave, *best_params)
            plt.plot(phases, model_data, color=filter_colors[filter_name], linewidth=1,
                     linestyle='--')

    if (model == 'single') or (model == 'double'):
        model_g = linex(phases, *best_params[0])
        model_r = linex(phases, *best_params[1])
        plt.plot(phases, model_g, color='g', linewidth=1, linestyle='--')
        plt.plot(phases, model_r, color='r', linewidth=1, linestyle='--')

    if model == 'full':
        for item in last_samples:
            for filter_name, filter_wave in zip(filters_used, wavelengths):
                # Plot the model
                model_data = model_mag(phases, filter_wave, *item)
                plt.plot(phases, model_data, color=filter_colors[filter_name], linewidth=1,
                         alpha=0.1)
    elif model == 'double':
        for item in last_samples[0]:
            # Plot the model
            model_data = linex(phases, *item)
            plt.plot(phases, model_data, color='g', linewidth=1, alpha=0.1)
        for item in last_samples[1]:
            # Plot the model
            model_data = linex(phases, *item)
            plt.plot(phases, model_data, color='r', linewidth=1, alpha=0.1)
    elif model == 'single':
        for item in last_samples[0]:
            # Plot the model
            model_data = linex(phases, item[0], best_params[0][1], item[1], item[2])
            plt.plot(phases, model_data, color='g', linewidth=1, alpha=0.1)
        for item in last_samples[1]:
            # Plot the model
            model_data = linex(phases, item[0], best_params[1][1], item[1], item[2])
            plt.plot(phases, model_data, color='r', linewidth=1, alpha=0.1)

    plt.gca().invert_yaxis()
    plt.xlabel('Phase (days)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.xlim(-50, 200)
    plt.ylim(26, 16)
    plt.savefig(f'{plot_dir}/{object_name}_model.pdf', bbox_inches='tight')
    plt.clf()
    plt.close('all')


def no_warnings(sampler, pos, n_steps, emcee_progress=True):
    """
    Run MCMC sampler while ignoring specific warnings.

    Parameters:
    ----------
    sampler : emcee.EnsembleSampler
        The MCMC sampler to run
    pos : array
        Initial positions for the walkers
    n_steps : int
        Number of steps to take
    emcee_progress : bool, optional
        Whether to show progress bar (default: True)

    Returns:
    -------
    The result of sampler.run_mcmc
    """
    import warnings

    with warnings.catch_warnings():
        # Filter all the specific RuntimeWarnings
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="invalid value encountered in scalar subtract",
                                module="emcee.moves.red_blue")
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="overflow encountered in square")
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="overflow encountered in multiply")
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="overflow encountered in power")
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="divide by zero encountered in log")

        # Run the sampler and return its result
        sampler.run_mcmc(pos, n_steps, progress=emcee_progress)

    return sampler


def run_mcmc_with_sigma_clipping(sampler, pos, n_steps, sigma_clip=3.0, repeats=1, emcee_progress=True):
    """
    Run MCMC with sigma clipping to help convergence.

    After each run, walkers outside the sigma_clip range in parameter space are replaced
    with new walkers drawn from within the sigma_clip range of the current distribution.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        MCMC sampler object
    pos : numpy.ndarray
        Initial positions for walkers
    n_steps : int
        Number of steps for each MCMC run
    sigma_clip : float, default=3.0
        Number of sigma to clip walkers at between runs
    repeats : int, default=1
        Number of times to repeat the MCMC process with sigma clipping
    emcee_progress : bool, default=True
        Whether to display progress bar

    Returns
    -------
    sampler : emcee.EnsembleSampler
        Updated sampler with chain from final run
    """
    n_walkers, n_dim = pos.shape

    # First run
    sampler = no_warnings(sampler, pos, n_steps, emcee_progress=emcee_progress)

    # Repeat the process if requested
    for i in range(repeats - 1):
        print(f"Starting MCMC run {i + 2} of {repeats}...")

        # Get the last positions
        last_pos = sampler.chain[:, -1, :]

        # Check for invalid values in the last position
        invalid_mask = np.any(~np.isfinite(last_pos), axis=1)
        if np.any(invalid_mask):
            print(f"Found {np.sum(invalid_mask)} walkers with invalid positions. Replacing them...")
            # Replace invalid walkers with valid ones
            valid_indices = np.where(~invalid_mask)[0]

            # Replace invalid walkers with valid ones plus some noise
            for idx in np.where(invalid_mask)[0]:
                valid_idx = np.random.choice(valid_indices)
                last_pos[idx] = last_pos[valid_idx] + np.random.normal(0, 1e-4, n_dim)

        # Calculate the median and std for each parameter
        medians = np.median(last_pos, axis=0)
        stds = np.std(last_pos, axis=0)

        # Handle cases where std is zero or very small to avoid division by zero
        stds = np.maximum(stds, 1e-10)

        # Identify walkers outside the sigma_clip range
        valid_walkers = np.all(np.abs(last_pos - medians) < sigma_clip * stds, axis=1)
        valid_indices = np.where(valid_walkers)[0]

        # If there are walkers outside the range, replace them
        if len(valid_indices) < n_walkers:
            print(f"Found {n_walkers - len(valid_indices)} walkers outside {sigma_clip}-sigma range.")
            print("Replacing with new walkers drawn from within the clipped distribution...")

            # Create new positions for invalid walkers by sampling from valid ones
            new_pos = np.copy(last_pos)
            invalid_indices = np.where(~valid_walkers)[0]

            for idx in invalid_indices:
                # Sample a valid walker
                valid_idx = np.random.choice(valid_indices)
                # Copy position but add some noise within the acceptable range
                for dim in range(n_dim):
                    new_pos[idx, dim] = last_pos[valid_idx, dim] + \
                                      np.random.normal(0, stds[dim] / sigma_clip)

            # Run the next iteration with the new positions
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning,
                                        message="invalid value encountered in scalar subtract",
                                        module="emcee.moves.red_blue")
                sampler = no_warnings(sampler, new_pos, n_steps, emcee_progress=emcee_progress)
        else:
            print("All walkers are within the specified sigma range.")
            # Still run the next iteration with the last positions
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning,
                                        message="invalid value encountered in scalar subtract",
                                        module="emcee.moves.red_blue")
                sampler = no_warnings(sampler, last_pos, n_steps, emcee_progress=emcee_progress)

    return sampler


def fit_data(input_table, phase_min=-200, phase_max=75, n_walkers=50, n_steps=50, n_cores=1,
             model='full', late_phase=40, default_err=0.1, default_decline_g=0.55,
             default_decline_r=0.37, burn_in=0.75, sigma_clip=2, repeats=2,
             save_trace=False, save_lcplot=False, use_median=False, object_name=None,
             plot_dir='plots', n_sigma_limit=3, emcee_progress=True):
    """"
    Fit the photometry using either a single expoential model, a double linear-exponential model,
    or a full model that includes temperature evolution.

    Parameters
    ----------
    input_table : astropy.table.Table
        Input data table containing photometry
    phase_min : float, default=-np.inf
        Minimum phase (in days) to consider for fitting
    phase_max : float, default=np.inf
        Maximum phase (in days) to consider for fitting
    n_walkers : int, default=50
        Number of walkers for the MCMC fitting
    n_steps : int, default=1000
        Number of steps for the MCMC fitting
    n_cores : int, default=1
        Number of CPU cores to use for parallel processing
    model : str, default='full'
        Model to use for fitting. Options are 'single', 'double', or 'full'.
    late_phase : float, default=40
        Phase (in days) to calculate the color at late times
    default_err : float, default=0.1
        Default error value to use if the input data has NaN or zero errors
    default_decline_g : float, default=0.55
        Default decline rate for the g-band light curve
    default_decline_r : float, default=0.37
        Default decline rate for the r-band light curve
    burn_in : float, default=0.75
        Fraction of the MCMC chain to discard as burn-in
    sigma_clip : float, default 3
        Discard walkers further than this many sigma from the mean
    repeats : float, default 1
        Repeat the emcee process this many times
    save_trace : bool, default=False
        Whether to save the trace plots of the MCMC fitting
    save_lcplot : bool, default=False
        Plot the model on an output light curve?
    use_median : bool, default=True
        Use the median values as opposed to the values
        with the highest likelihood
    object_name : str, default=None
        Name of the object being fitted
    plot_dir : str, default='plots'
        Directory to save the plots
    n_sigma_limit : int, default 3
        Default upper limit sigma to calculate the error on the limit
    emcee_progress : bool, default=True
        Whether to display progress bar

    Returns
    -------
    parameters : dict
        Dictionary containing the best-fit parameters for the model
    color_peak : float
        Color at peak brightness (g - r)
    late_color : float
        Color at the specified late phase (g - r)
    late_color10 : float
        Color at 10 days after peak (g - r)
    late_color20 : float
        Color at 20 days after peak (g - r)
    late_color40 : float
        Color at 40 days after peak (g - r)
    late_color60 : float
        Color at 60 days after peak (g - r)
    first_to_peak_r : float
        Time from first observation to peak in the r-band
    first_to_peak_g : float
        Time from first observation to peak in the g-band
    peak_to_last_r : float
        Time from peak to last observation in the r-band
    peak_to_last_g : float
        Time from peak to last observation in the g-band
    bright_mjd : float
        MJD of the brightest observation
    first_mjd : float
        MJD of the first observation
    brightest_mag : float
        Brightest magnitude (minimum value of the light curve)
    green_brightest : float
        Brightest magnitude in the g-band
    red_brightest : float
        Brightest magnitude in the r-band
    chi2 : float
        Reduced chi squared value of the fit
    chain : tuple
        One or two chains depending on the model
    Plus all the input parameters used for the fit
    """

    # Process data
    np.random.seed(42)
    output_table, bright_mjd, first_mjd = format_data(input_table, default_err=default_err,
                                                      n_sigma_limit=n_sigma_limit,
                                                      phase_max=phase_max,
                                                      phase_min=phase_min)

    # Define empty variables for the model parameters
    parameters = {'': None}
    color_peak = None
    late_color = None
    late_color10 = None
    late_color20 = None
    late_color40 = None
    late_color60 = None
    first_to_peak_r = None
    first_to_peak_g = None
    peak_to_last_r = None
    peak_to_last_g = None
    brightest_mag = None
    green_brightest = None
    red_brightest = None
    chi2 = None
    chains = [None]

    # Check if we have any data to work with and
    # calculate the brightest MJD and first MJD
    if output_table is None or len(output_table) == 0:
        print("No valid data points found after filtering.")
        return (parameters, color_peak, late_color, late_color10, late_color20, late_color40, late_color60,
                first_to_peak_r, first_to_peak_g, peak_to_last_r, peak_to_last_g, bright_mjd,
                first_mjd, brightest_mag, green_brightest, red_brightest, chi2, chains, output_table)

    # Get r and g-band data
    green_det = output_table[(output_table['Filter'] == 'g') & (output_table['Phase_boom'] < phase_max) &
                             (output_table['Phase_boom'] > phase_min) & (output_table['UL'] == 'False')
                             & (output_table['Ignore'] == 'False')]
    red_det = output_table[(output_table['Filter'] == 'r') & (output_table['Phase_boom'] < phase_max) &
                           (output_table['Phase_boom'] > phase_min) & (output_table['UL'] == 'False')
                           & (output_table['Ignore'] == 'False')]
    output_det = output_table[(output_table['Phase_boom'] < phase_max) &
                              (output_table['Phase_boom'] > phase_min) & (output_table['UL'] == 'False')
                              & (output_table['Ignore'] == 'False')]

    # Calculate the brightest magnitude in g and r bands, if these exist
    if len(green_det) > 0:
        green_brightest = green_det['Mag'][np.nanargmin(green_det['Mag'])]
    if len(red_det) > 0:
        red_brightest = red_det['Mag'][np.nanargmin(red_det['Mag'])]

    # If using the old model that splits up bands into g and r
    if (model == 'single') or (model == 'double'):
        # Define parameters for the model
        parameters = {'lc_width_r': None,
                      'lc_decline_r': None,
                      'phase_offset_r': None,
                      'mag_offset_r': None,
                      'lc_width_g': None,
                      'lc_decline_g': None,
                      'phase_offset_g': None,
                      'mag_offset_g': None}

        # Find the brightest magnitude
        try:
            brightest_mag = np.nanmin(np.append(green_det['Mag'], red_det['Mag']))
        except ValueError:
            pass

        # Check if we have enough data points for fitting
        if len(green_det) < 2 or len(red_det) < 2:
            print("Not enough data points in g or r band for fitting.")
        else:
            # Split into g and r bands and select only data within the date range
            green = output_table[(output_table['Filter'] == 'g') & (output_table['Phase_boom'] < phase_max) &
                                 (output_table['Phase_boom'] > phase_min)]
            red = output_table[(output_table['Filter'] == 'r') & (output_table['Phase_boom'] < phase_max) &
                               (output_table['Phase_boom'] > phase_min)]

            # Make sure there's at least two data points in each band
            if len(green_det) < 2 or len(red_det) < 2:
                print("Not enough data points in g or r band for fitting.")
            else:
                # Create the initial positions of the walkers
                def create_prior():
                    lc_width = np.random.uniform(-0.4, 0.0, n_walkers)
                    lc_decline = np.random.uniform(0.01, 1.0, n_walkers)
                    phase_offset = np.random.uniform(-20, 10, n_walkers)
                    mag_offset = np.random.uniform(brightest_mag-0.3, brightest_mag+0.3, n_walkers)

                    pos = np.array([lc_width, lc_decline, phase_offset, mag_offset]).T
                    return pos

                # Create array of proper length
                if brightest_mag <= priors['mag_offset'][0]:
                    prior_mag = priors['mag_offset'][0]
                    print(f'Warning: Brightest magnitude {brightest_mag} is less than the prior {prior_mag} minimum.')
                pos_in = create_prior()
                pos_out = pos_in[0:1]

                while len(pos_out) < n_walkers:
                    pos = pos_in[[np.isfinite(lnprior_double(i)) for i in pos_in]]
                    pos_out = np.append(pos_out, pos, axis=0)

                # Crop to correct length
                if len(pos_out) != n_walkers:
                    pos = pos_out[1:n_walkers+1]
                else:
                    pos = pos_out

                # If not fitting for a decline, remove that parameter
                if model == 'single':
                    pos = pos[:, [0, 2, 3]]

                # Number of parameters being fit
                n_dim = pos.shape[1]

                # Extract the data for g and r bands
                x_g = green['Phase_peak']
                y_g = green['Mag']
                z_g = green['MagErr']
                l_g = np.array([s == 'True' for s in green['UL']])
                x_r = red['Phase_peak']
                y_r = red['Mag']
                z_r = red['MagErr']
                l_r = np.array([s == 'True' for s in red['UL']])

                # Setup the MCMC sampler
                if model == 'double':
                    sampler_red = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_double,
                                                        args=(x_r, y_r, z_r, l_r), threads=n_cores)
                    sampler_green = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_double,
                                                          args=(x_g, y_g, z_g, l_g), threads=n_cores)
                elif model == 'single':
                    sampler_red = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_single,
                                                        args=(x_r, y_r, z_r, l_r, default_decline_r), threads=n_cores)
                    sampler_green = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_single,
                                                          args=(x_g, y_g, z_g, l_g, default_decline_g), threads=n_cores)

                # Run the MCMC with sigma clipping
                print("\nRunning r-MCMC ...")
                sampler_red = run_mcmc_with_sigma_clipping(sampler_red, pos, n_steps,
                                                           sigma_clip=sigma_clip,
                                                           repeats=repeats,
                                                           emcee_progress=emcee_progress)

                print("\nRunning g-MCMC ...")
                sampler_green = run_mcmc_with_sigma_clipping(sampler_green, pos, n_steps,
                                                             sigma_clip=sigma_clip,
                                                             repeats=repeats,
                                                             emcee_progress=emcee_progress)

                # Only consider the last quarter of the chain for parameter estimation
                samples_r_crop = sampler_red.chain[:, -int(n_steps*(1-burn_in)):, :].reshape((-1, n_dim))
                samples_g_crop = sampler_green.chain[:, -int(n_steps*(1-burn_in)):, :].reshape((-1, n_dim))

                # Get the log probabilities
                log_prob_red = sampler_red.lnprobability[:, -1]
                log_prob_green = sampler_green.lnprobability[:, -1]
                last_samples_red = sampler_red.chain[:, -1, :]
                last_samples_green = sampler_green.chain[:, -1, :]

                # Find the index of the maximum likelihood
                max_params_red = last_samples_red[np.argmax(log_prob_red)]
                max_params_green = last_samples_green[np.argmax(log_prob_green)]

                # Obtain the parametrs of the best fit
                if model == 'double':
                    lc_width_mcmc_r, lc_decline_mcmc_r, phase_offset_mcmc_r, mag_offset_mcmc_r = \
                        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples_r_crop, [15.87, 50, 84.13], axis=0)))
                    lc_width_mcmc_g, lc_decline_mcmc_g, phase_offset_mcmc_g, mag_offset_mcmc_g = \
                        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples_g_crop, [15.87, 50, 84.13], axis=0)))
                elif model == 'single':
                    lc_width_mcmc_r, phase_offset_mcmc_r, mag_offset_mcmc_r = \
                        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples_r_crop, [15.87, 50, 84.13], axis=0)))
                    lc_width_mcmc_g, phase_offset_mcmc_g, mag_offset_mcmc_g = \
                        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples_g_crop, [15.87, 50, 84.13], axis=0)))
                    lc_decline_mcmc_r = (default_decline_r, default_decline_r, default_decline_r)
                    lc_decline_mcmc_g = (default_decline_g, default_decline_g, default_decline_g)

                # Get best fit parameters
                if use_median:
                    parameters['lc_width_r'] = lc_width_mcmc_r[0]
                    parameters['lc_decline_r'] = lc_decline_mcmc_r[0]
                    parameters['phase_offset_r'] = phase_offset_mcmc_r[0]
                    parameters['mag_offset_r'] = mag_offset_mcmc_r[0]
                    parameters['lc_width_g'] = lc_width_mcmc_g[0]
                    parameters['lc_decline_g'] = lc_decline_mcmc_g[0]
                    parameters['phase_offset_g'] = phase_offset_mcmc_g[0]
                    parameters['mag_offset_g'] = mag_offset_mcmc_g[0]
                else:
                    if model == 'double':
                        parameters['lc_width_r'] = max_params_red[0]
                        parameters['lc_decline_r'] = max_params_red[1]
                        parameters['phase_offset_r'] = max_params_red[2]
                        parameters['mag_offset_r'] = max_params_red[3]
                        parameters['lc_width_g'] = max_params_green[0]
                        parameters['lc_decline_g'] = max_params_green[1]
                        parameters['phase_offset_g'] = max_params_green[2]
                        parameters['mag_offset_g'] = max_params_green[3]
                    elif model == 'single':
                        parameters['lc_width_r'] = max_params_red[0]
                        parameters['lc_decline_r'] = default_decline_r
                        parameters['phase_offset_r'] = max_params_red[1]
                        parameters['mag_offset_r'] = max_params_red[2]
                        parameters['lc_width_g'] = max_params_green[0]
                        parameters['lc_decline_g'] = default_decline_g
                        parameters['phase_offset_g'] = max_params_green[1]
                        parameters['mag_offset_g'] = max_params_green[2]

                # Band specfic
                g_best = (parameters['lc_width_g'], parameters['lc_decline_g'],
                          parameters['phase_offset_g'], parameters['mag_offset_g'])
                r_best = (parameters['lc_width_r'], parameters['lc_decline_r'],
                          parameters['phase_offset_r'], parameters['mag_offset_r'])

                # Calculate color at different phases
                def get_color(phase, g_best, r_best):
                    g_mag = linex(phase, *g_best)
                    r_mag = linex(phase, *r_best)
                    return g_mag - r_mag

                # Get color during peak
                color_peak = get_color(0, g_best, r_best)
                # Get color at the pre-specified late phase
                late_color = get_color(late_phase, g_best, r_best)
                # Get color at common late phases
                late_color10 = get_color(10, g_best, r_best)
                late_color20 = get_color(20, g_best, r_best)
                late_color40 = get_color(40, g_best, r_best)
                late_color60 = get_color(60, g_best, r_best)

                # Calculate the model time of peak
                peak_model_r = d_linex(*r_best[:3])
                peak_model_g = d_linex(*g_best[:3])
                # Calculate the time from peak to first detection
                first_to_peak_r = peak_model_r - np.min(red_det['Phase_peak'])
                first_to_peak_g = peak_model_g - np.min(green_det['Phase_peak'])
                # Calculate the time from peak to last detection
                peak_to_last_r = np.max(red_det['Phase_peak']) - peak_model_r
                peak_to_last_g = np.max(green_det['Phase_peak']) - peak_model_g

                # Calculate chi squared
                model_r = linex(x_r, *r_best)
                model_g = linex(x_g, *g_best)
                # Append green and red bands
                ydata = np.append(y_r, y_g)
                ymod = np.append(model_r, model_g)
                sigma = np.append(z_r, z_g)
                limits = np.append(l_r, l_g)
                if model == 'double':
                    chi2 = calc_chi2(ydata, ymod, 4, sigma, limits)
                elif model == 'single':
                    chi2 = calc_chi2(ydata, ymod, 3, sigma, limits)

                # Plot model
                if save_lcplot:
                    last_samples_g = sampler_green.chain[:, -1, :]
                    last_samples_r = sampler_red.chain[:, -1, :]
                    plot_model(input_table, [last_samples_g, last_samples_r], [g_best, r_best], model, object_name,
                               plot_dir=plot_dir, default_err=default_err, n_sigma_limit=n_sigma_limit)

                if save_trace:
                    # Plot the traces
                    output_dir = 'traces'

                    # Define parameters for each model type
                    if model == 'double':
                        params = ['lc_width', 'lc_decline', 'phase_offset', 'mag_offset']
                    else:  # model == 'single'
                        params = ['lc_width', 'phase_offset', 'mag_offset']

                    # Define mapping of parameters to their values for each band
                    samplers = {'r': sampler_red, 'g': sampler_green}
                    param_values = {
                        'r': {
                            'lc_width': lc_width_mcmc_r,
                            'lc_decline': lc_decline_mcmc_r,
                            'phase_offset': phase_offset_mcmc_r,
                            'mag_offset': mag_offset_mcmc_r
                        },
                        'g': {
                            'lc_width': lc_width_mcmc_g,
                            'lc_decline': lc_decline_mcmc_g,
                            'phase_offset': phase_offset_mcmc_g,
                            'mag_offset': mag_offset_mcmc_g
                        }
                    }

                    # Generate trace plots for both bands
                    for band in param_values:
                        sampler = samplers[band]

                        for i, param in enumerate(params):
                            plot_trace(
                                sampler.chain[:, :, i],
                                param_values[band][param],
                                param_values[band][param],
                                priors[param][0],
                                priors[param][1],
                                f"{param}_{band}",
                                f"{param}_{band}",
                                False, n_steps, burn_in, output_dir, repeats, object_name
                            )

                # Output chains
                chains = [sampler_green.chain, sampler_red.chain]

    elif model == 'full':
        # Define parameters for the model
        parameters = {'lc_width': None,
                      'lc_decline': None,
                      'phase_offset': None,
                      'mag_offset': None,
                      'initial_temp': None,
                      'cooling_rate': None}

        # How many individual filters are there?
        n_filters = len(np.unique(output_det['Filter']))

        if (n_filters < 2) or (len(output_det) < 5):
            try:
                # Calculate the brightest magnitude
                brightest_mag = np.nanmin(output_det['Mag'])
            except ValueError:
                brightest_mag = None
                pass
            print("Not enough data points or bands for fitting.")
        else:
            # Calculate the brightest magnitude
            brightest_mag = np.nanmin(output_det['Mag'])

            # Create the prior
            def create_prior():
                lc_width = np.random.uniform(-0.4, 0.0, n_walkers)
                lc_decline = np.random.uniform(0.01, 1.0, n_walkers)
                phase_offset = np.random.uniform(-20, 10, n_walkers)
                use_brightest_mag = np.max([priors['mag_offset'][0], brightest_mag])
                mag_offset = np.random.uniform(use_brightest_mag-0.3, use_brightest_mag+0.3, n_walkers)
                initial_temp = np.random.uniform(3000.0, 7000.0, n_walkers)
                cooling_rate = np.random.uniform(10, 1000.0, n_walkers)

                pos = np.array([lc_width, lc_decline, phase_offset, mag_offset,
                                initial_temp, cooling_rate]).T
                return pos

            # Create array of proper length
            pos_in = create_prior()
            pos_out = pos_in[0:1]
            while len(pos_out) < n_walkers:
                pos = pos_in[[np.isfinite(lnprior_full(i)) for i in pos_in]]
                pos_out = np.append(pos_out, pos, axis=0)

            # Crop to correct length
            if len(pos_out) != n_walkers:
                pos = pos_out[1:n_walkers+1]
            else:
                pos = pos_out

            # Number of parameters being fit
            n_dim = pos.shape[1]

            # Extract the data for g and r bands
            phases = output_det['Phase_peak']
            wavelengths = output_det['Cenwave']
            obs_mags = output_det['Mag']
            err_mags = output_det['MagErr']
            obs_limits = np.array([s == 'True' for s in output_det['UL']])

            # Setup the MCMC sampler
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_full,
                                            args=(phases, wavelengths, obs_mags, err_mags, obs_limits), threads=n_cores)

            # Run the MCMC
            print("Running MCMC ...")
            sampler = run_mcmc_with_sigma_clipping(sampler, pos, n_steps,
                                                   sigma_clip=sigma_clip,
                                                   repeats=repeats,
                                                   emcee_progress=emcee_progress)

            # Only consider the last quarter of the chain for parameter estimation
            samples_crop = sampler.chain[:, -int(n_steps*(1-burn_in)):, :].reshape((-1, n_dim))

            # Get the log probabilities
            log_prob = sampler.lnprobability[:, -1]
            last_samples = sampler.chain[:, -1, :]

            # Find the index of the maximum likelihood
            max_idx = np.argmax(log_prob)
            max_params = last_samples[max_idx]

            # Obtain the parametrs of the best fit
            lc_width_mcmc, lc_decline_mcmc, phase_offset_mcmc, mag_offset_mcmc, \
                initial_temp_mcmc, cooling_rate_mcmc = \
                map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                    zip(*np.percentile(samples_crop, [15.87, 50, 84.13], axis=0)))

            # Get best-fit parameters
            median_params = np.array([
                lc_width_mcmc[0],
                lc_decline_mcmc[0],
                phase_offset_mcmc[0],
                mag_offset_mcmc[0],
                initial_temp_mcmc[0],
                cooling_rate_mcmc[0],
            ])

            if use_median:
                best_params = median_params
            else:
                best_params = max_params

            for key, value in zip(parameters.keys(), best_params):
                parameters[key] = value

            # Calculate color at different phases using representative wavelengths
            # from the actual light curve. This keeps ZTF-only, Rubin-only, and
            # mixed ZTF+Rubin colors internally consistent with the fitted data.
            color_filters = np.array(['g', 'r'])
            color_wavelengths = dict(zip(color_filters, get_filter_wavelengths(color_filters, output_table)))

            def get_color(phase, best_params):
                g_mag = model_mag(phase, color_wavelengths['g'], *best_params)
                r_mag = model_mag(phase, color_wavelengths['r'], *best_params)
                return g_mag - r_mag

            # Get color during peak
            color_peak = get_color(0, best_params)
            # Get color at the pre-specified late phase
            late_color = get_color(late_phase, best_params)
            # Get color at common late phases
            late_color10 = get_color(10, best_params)
            late_color20 = get_color(20, best_params)
            late_color40 = get_color(40, best_params)
            late_color60 = get_color(60, best_params)

            # Calculate the time of model peak
            peak_model = d_linex(*best_params[:3])
            # Calculate the time from peak to first detection
            first_to_peak = peak_model - np.min(output_det['Phase_peak'])
            # Calculate the time from peak to last detection
            peak_to_last = np.max(output_det['Phase_peak']) - peak_model
            # For the time being, assign this to both g and r bands
            first_to_peak_r = first_to_peak
            first_to_peak_g = first_to_peak
            # Calculate the time from peak to last detection
            peak_to_last_r = peak_to_last
            peak_to_last_g = peak_to_last

            # Calculate chi squared
            model_mags = model_mag(phases, wavelengths, *best_params)
            chi2 = calc_chi2(obs_mags, model_mags, n_dim, err_mags, obs_limits)

            # Plot model
            if save_lcplot:
                plot_model(input_table, last_samples, best_params, model, object_name,
                           plot_dir=plot_dir, default_err=default_err, n_sigma_limit=n_sigma_limit)

            # Plot trace
            if save_trace:
                output_dir = 'traces'

                # Define parameters and their values
                params = [
                    ('lc_width', lc_width_mcmc),
                    ('lc_decline', lc_decline_mcmc),
                    ('phase_offset', phase_offset_mcmc),
                    ('mag_offset', mag_offset_mcmc),
                    ('initial_temp', initial_temp_mcmc),
                    ('cooling_rate', cooling_rate_mcmc)
                ]

                # Generate trace plots for all parameters
                for i, (param_name, param_value) in enumerate(params):
                    plot_trace(
                        sampler.chain[:, :, i],
                        param_value,
                        param_value,
                        priors[param_name][0],
                        priors[param_name][1],
                        param_name,
                        param_name,
                        False,
                        n_steps,
                        burn_in,
                        output_dir,
                        repeats,
                        object_name
                    )

            # Output chains
            chains = [sampler.chain]

    return (parameters, color_peak, late_color, late_color10, late_color20, late_color40, late_color60,
            first_to_peak_r, first_to_peak_g, peak_to_last_r, peak_to_last_g, bright_mjd,
            first_mjd, brightest_mag, green_brightest, red_brightest, chi2, chains, output_table)
