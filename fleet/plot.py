from .model import linex, model_mag, get_filter_wavelengths
from .catalog import sdss_refs, psst_refs, calc_separations
import glob
from astropy import table
from matplotlib.pyplot import cm
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import matplotlib.pyplot as plt
import matplotlib.dates as md
from astropy.time import Time
from bs4 import BeautifulSoup
from PyAstronomy import pyasl
from io import BytesIO
from PIL import Image
import numpy as np
import datetime
import requests
import urllib
import ephem
import os
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'font.family': 'serif'})


def calc_absmag(obsmag, redshift, k_correct=True):
    """
    Calculate the absolute magnitude for a given redshift.

    Parameters
    ----------
    obsmag : float
        Observed magnitude
    redshift : float
        Redshift of the object
    k_correct : bool, default True
        If True, apply a k-correction to the magnitude

    Returns
    -------
    absmag : float
        Absolute magnitude
    """

    # Calculate distance modulus
    DL = cosmo.luminosity_distance(redshift).to(u.pc).value
    DM = 5 * np.log10(DL / 10)

    # Apply k-correction
    if k_correct:
        kcorr = 2.5 * np.log10(1 + redshift)
    else:
        kcorr = 0

    # Calculate absolute magnitude
    absmag = obsmag - DM + kcorr

    return absmag


def plot_colors(bands):
    """
    Assign matplotlib colors to a given band or list of bands. If a filter
    is not known, the color will be black.

    Parameters
    ----------
    bands : str or list of str
        Name of filter(s).

    Returns
    -------
    colors : str or list of str
        Matplotlib color(s) for the given band(s).
    """

    # Assign matplotlib colors to Swift bands
    colors_UVOT = cm.rainbow(np.linspace(0, 1, 7))

    # Create a dictionary mapping bands to colors
    band_color_map = {
        "u": 'navy', "u'": 'navy', "U": 'navy',
        "g": 'g', "g'": 'g',
        'r': 'r', "r'": 'r', 'R': 'r', 'Rs': 'r',
        'i': 'maroon', "i'": 'maroon', 'I': 'maroon',
        "z": 'saddlebrown', "z'": 'saddlebrown',
        'V': 'lawngreen',
        'B': 'darkcyan',
        'C': 'c',
        'w': 'goldenrod',
        'G': 'orange',
        'W1': 'deeppink',
        'W2': 'tomato',
        'orange': 'gold',
        'cyan': 'blue',
        'Clear': 'magenta',
        'UVM2': colors_UVOT[0],
        'UVW1': colors_UVOT[1],
        'UVW2': colors_UVOT[2],
        'F475W': 'lightsteelblue',
        'F625W': 'slategray',
        'F775W': 'tan',
        'F850LP': 'gray',
        'H': 'hotpink',
        'J': 'mediumvioletred',
        'K': 'palevioletred', "Ks": 'palevioletred',
        'Y': 'indigo', "y": 'indigo',
        'v': 'aquamarine',
        'F062': '#003366',
        'F087': '#0077BB',
        'F106': '#99DDFF',
        'F129': '#44BB99',
        'F158': '#EEDD88',
        'F184': '#EE8866',
        'F213': '#B704C2',
        'F146': '#994455'
    }

    # If input is a single band, return its color
    if isinstance(bands, str):
        return band_color_map.get(bands, 'k')

    # If input is a list of bands, return a list of colors
    return np.array([band_color_map.get(band, 'k') for band in bands])


def photometry_source_indices(sources, telescopes=None):
    """Group photometry rows into source categories used by the light-curve plot."""
    sources = np.char.lower(np.char.strip(np.asarray(sources, dtype=str)))
    if telescopes is None:
        telescopes = np.array([''] * len(sources), dtype=str)
    else:
        telescopes = np.char.lower(np.char.strip(np.asarray(telescopes, dtype=str)))

    is_rubin = np.isin(telescopes, ['rubin', 'lsst']) | np.isin(sources, ['rubin', 'lsst'])
    is_ztf = (np.isin(telescopes, ['ztf']) | np.isin(sources, ['ztf', 'alerce'])) & ~is_rubin
    is_local = np.isin(sources, ['local', 'flwo'])
    is_other = ~(is_rubin | is_ztf | is_local)

    return {
        'ztf': np.flatnonzero(is_ztf),
        'rubin': np.flatnonzero(is_rubin),
        'local': np.flatnonzero(is_local),
        'other': np.flatnonzero(is_other),
    }


def plot_1a(time, y_0=0.92540, m=0.0182386, t_0=3.95539, g_0=-0.884576, sigma_0=11.0422, tau=-17.5522, theta_0=18.7745):
    """
    Generate a type 1a supernova light curve from an analytical function:
    Slope + Gaussian + Exponential Rise
    As described in https://arxiv.org/pdf/1707.07614.pdf

    g_model = plot_1a(timex, 2.02715, 0.0132083, -1.24344, -1.582490, 10.1705, -21.8477, 15.3465)
    r_model = plot_1a(timex, 0.92540, 0.0182386,  3.95539, -0.884576, 11.0422, -17.5522, 18.7745)
    i_model = plot_1a(timex, 0.79161, 0.0179976,  7.63109, -0.723820, 11.3517, -16.4830, 18.0652)

    Parameters
    ----------
    time : array-like
        Time array for the light curve
    y_0 : float, default 0.92540
        Magnitude offset
    m : float, default 0.0182386
        Slope of late time decay
    t_0 : float, default 3.95539
        Time of maximum for gaussian
    g_0 : float, default -0.884576
        Height of gaussian
    sigma_0 : float, default 11.0422
        Width of gaussian
    tau : float, default -17.5522
        Explosion time
    theta_0 : float, default 18.7745
        Rate of exponential rise

    Returns
    -------
    time : array-like
        Time array adjusted to the light curve
    magnitude : array-like
        Magnitude array of the light curve
    """

    # Components
    slope = y_0 + m * (time - t_0)
    gaussian = g_0 * np.exp(-(time - t_0) ** 2 / (2 * sigma_0 ** 2))
    exponential = 1 - np.exp((tau - time) / theta_0)

    # Result
    magnitude = (slope + gaussian) / exponential
    return time - time[np.argmin(magnitude)], magnitude - magnitude[np.argmin(magnitude)]


# Model Type Ia
time_1a_model_r = np.linspace(-15, 120, 500)
phase_1a_model_r, magnitude_1a_model_r = plot_1a(time_1a_model_r)


def query_PS1_image(ra_deg, dec_deg, image_color, wcs_size=90, autoscale=75):
    """
    Query the 3pi website and download an image in the
    specified filter. If no image was found return '--'.
    If the image_color is not found, return it in 'r' band.

    Parameters
    ----------
    ra_deg : float
        Coordinates of the object in degrees
    dec_deg : float
        Coordinates of the object in degrees
    image_color : str
        Filter to search for, either 'g', 'r', 'i', 'z', or 'y'
    wcs_size : float
        Image size in arcsec
    autoscale : float
        Scaling of the image

    Returns
    -------
    img : PIL Image object
        Image of the object in the specified filter
        or None if no image was found.
    """

    # Set image size
    image_size = wcs_size * 4

    # Format RA and dec
    if dec_deg > 0:
        ra_deg_str = str(np.around(ra_deg, decimals=6)) + '+' + str(np.around(dec_deg, decimals=6))
    else:
        ra_deg_str = str(np.around(ra_deg, decimals=6)) + str(np.around(dec_deg, decimals=6))

    # Make sure filter is valid
    if image_color not in ['g', 'r', 'i', 'z', 'y']:
        print("Invalid filter. Using 'r' instead.")
        image_color = 'r'

    # Request data from PS1 website
    requests_link = ("http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos=" + ra_deg_str + "&filter=" + image_color + "&size=" +
                     str(image_size) + "&autoscale=" + str(autoscale))

    try:
        # Extract image
        image_req = requests.get(requests_link)
        image_data = image_req.text
        soup = BeautifulSoup(image_data, features="html.parser")
        URL = 'http:' + soup.find_all('img')[1].get('src')
        file = BytesIO(urllib.request.urlopen(URL).read())
        img = Image.open(file)
        return img
    except Exception as e:
        print("Error: ", e)
        print("No image found for %s %s in %s band" % (ra_deg, dec_deg, image_color))
        return None


def calc_airmass(RA, DEC, Telescope, do_observability=True):
    """
    Calculate the Airmass of a source from a specified telescope

    Parameters
    ----------
    RA : str
        Right Ascension of the star in 00:00:00 format
    DEC : str
        Declination of the star in 00:00:00 format
    Telescope : str
        Name of the telescope to calculate the airmass for
        Options are: MMT, Magellan, FLWO, Kitt Peak, McDonald,
        CTIO, Harvard, Las Campanas, AAT, HWT, Gemini_North,
        Gemini_South
    do_observability : bool
        If True, calculate the airmass and sun elevation

    Returns
    -------
    Dates : array
        Array of dates to calculate the airmass
    Airmass : array
        Airmass value at each date
    SunElevation : array
        Sun's elevation at each date
    """

    # Specify the coordinates based on the observatory
    if Telescope == 'MMT':
        Lat, Lon, Alt = 31.6883, -110.885, 2608
    if Telescope == 'Magellan':
        Lat, Lon, Alt = -29.0182, -70.6915, 2516
    if Telescope == 'FLWO':
        Lat, Lon, Alt = 31.6816, -110.876, 2344
    if Telescope == 'Kitt Peak':
        Lat, Lon, Alt = 31.9633, -111.600, 2120
    if Telescope == 'McDonald':
        Lat, Lon, Alt = 30.6716, -104.021, 2075
    if Telescope == 'CTIO':
        Lat, Lon, Alt = -30.1650, -70.8150, 2215
    if Telescope == 'Harvard':
        Lat, Lon, Alt = 42.3770, -71.1167,   54
    if Telescope == 'Las Campanas':
        Lat, Lon, Alt = -29.0182, -70.6915, 2380
    if Telescope == 'AAT':
        Lat, Lon, Alt = -31.2754, 149.0672, 1164
    if Telescope == 'HWT':
        Lat, Lon, Alt = 28.7610, -17.8820, 2332
    if Telescope == 'Gemini_North':
        Lat, Lon, Alt = 19.8200, -155.468, 4213
    if Telescope == 'Gemini_South':
        Lat, Lon, Alt = -30.2280, -70.7230, 2725

    # Create the observatory
    Observatory = ephem.Observer()
    Observatory.lat = str(int(Lat)) + ":" + str(np.around((Lat - int(Lat))*60, 1))
    Observatory.lon = str(int(Lon)) + ":" + str(np.around((Lon - int(Lon))*60, 1))
    Observatory.elevation = Alt

    # Choose start of date counter as right now with a stepsize in minutes
    # But set the start hour to 4PM
    Date0 = datetime.datetime.now()
    Date0 = Date0.replace(hour=16)
    Stepsize = 30

    # Empty Variables to modify
    Dates = np.array([])

    # Define the start time of the observations, and the subsequent timesteps
    # For which to do the calculations.
    Step = datetime.timedelta(0, 60*Stepsize)
    Hours = np.arange(24*60/Stepsize)
    for i in range(len(Hours)):
        Dates = np.append(Dates, Date0 + Step * i)

    # Don't actually run the function if requested
    if not do_observability:
        return Dates, np.nan * np.ones(len(Dates)), np.nan * np.ones(len(Dates))

    # Define the observer's location
    location = EarthLocation(lat=Lat * u.deg, lon=Lon * u.deg, height=Alt * u.m)

    SunElevation = np.zeros(len(Dates))
    for i in range(len(Dates)):
        # Convert the date to an Astropy Time object
        time = Time(Dates[i])
        # Calculate the Sun's position
        sun_altaz = get_sun(time).transform_to(AltAz(obstime=time, location=location))
        # Extract the Sun's elevation
        SunElevation[i] = sun_altaz.alt.deg

    # Create the star
    Star = ephem.readdb("Star,f|M|A0,%s,%s,0,2000" % (RA, DEC))

    # Calculate the airmass of the star for the selected dates
    ZenithAngle = np.zeros(len(Dates))
    Airmass = np.zeros(len(Dates))

    for i in range(len(ZenithAngle)):
        Observatory.date = Dates[i]
        Star.compute(Observatory)
        ZenithAngle[i] = 90.0 - (Star.alt * 180 / np.pi)
        Airmass[i] = pyasl.airmassPP(ZenithAngle[i])

    Airmass[Airmass < 0] = 'Nan'

    return Dates, Airmass, SunElevation


def is_it_observable(Airmass_telescope, SunElevation_telescope, SunElevation=-18.0, max_airmass=2.0):
    """
    Determine if at any point an object is observable
    given its airmass and the Sun's elevation.

    Parameters
    ----------
    Airmass_telescope : array
        Array of airmasses
    SunElevation_telescope : array
        Array of Sun's elevation
    SunElevation : float
        Minimum Sun's elevation for astronomical twilight
    max_airmass : float
        Maximum acceptable airmass for the object to be observable

    Returns
    -------
    bool
        True if the object is observable, False otherwise
    """

    # Is the airmass a real number
    good_telescope = np.isfinite(Airmass_telescope)
    # Does it satisfy the conditions
    observable_telescope = np.where((SunElevation_telescope[good_telescope] <= SunElevation) & (Airmass_telescope[good_telescope] < max_airmass))[0]

    if len(observable_telescope) > 0:
        return True
    else:
        return False


def calculate_observability(ra_deg, dec_deg, do_observability=True, SunElevation=-18.0, max_airmass=2.0):
    """
    Calculate the airmass for MMT, Magellan, and McDonald for an object
    given its RA and DEC.

    Parameters
    ----------
    ra_deg : float
        Right Ascension of the object in degrees
    dec_deg : float
        Declination of the object in degrees
    do_observability : bool, default True
        If True, calculate if the object is observable
    SunElevation : float, default -18.0
        Minimum Sun's elevation for astronomical twilight
    max_airmass : float, default 2.0
        Maximum acceptable airmass for the object to be observable

    Returns
    -------
    telescope_arrays : list
        List of arrays containing the names of the telescopes
        (MMT, Magellan, McDonald).
    dates_arrays : list
        List of arrays containing the time arrays
        for all the telescopes.
    airmasses_arrays : list
        List of arrays containing the airmass arrays
        for all the telescopes.
    sun_elevations_arrays : list
        List of arrays containing the sun elevation arrays
        for all the telescopes.
    observable_arrays : list
        List of booleans indicating if the object is observable
        from each telescope.
    """

    # Convert coordinates to string format
    coord = SkyCoord(ra_deg, dec_deg, unit="deg")
    coord_str = coord.to_string('hmsdms', sep=':')
    RA_str = coord_str[:coord_str.find(' ')]
    DEC_str = coord_str[coord_str.find(' ')+1:]

    # Calculate airmass for MMT and Magellan
    Dates_MMT, Airmass_MMT, SunElevation_MMT = calc_airmass(RA_str, DEC_str, 'MMT', do_observability)
    Dates_Magellan, Airmass_Magellan, SunElevation_Magellan = calc_airmass(RA_str, DEC_str, 'Magellan', do_observability)
    Dates_McDonald, Airmass_McDonald, SunElevation_McDonald = calc_airmass(RA_str, DEC_str, 'McDonald', do_observability)

    # Calculate if the source will be observable from MMT and Magellan
    if do_observability:
        MMT_observable = is_it_observable(Airmass_MMT, SunElevation_MMT, SunElevation, max_airmass)
        Magellan_observable = is_it_observable(Airmass_Magellan, SunElevation_Magellan, SunElevation, max_airmass)
        McDonald_observable = is_it_observable(Airmass_McDonald, SunElevation_McDonald, SunElevation, max_airmass)
    else:
        MMT_observable = None
        Magellan_observable = None
        McDonald_observable = None

    # Create and return the arrays
    telescope_arrays = ['MMT', 'Magellan', 'McDonald']
    dates_arrays = [Dates_MMT, Dates_Magellan, Dates_McDonald]
    airmasses_arrays = [Airmass_MMT, Airmass_Magellan, Airmass_McDonald]
    sun_elevations_arrays = [SunElevation_MMT, SunElevation_Magellan, SunElevation_McDonald]
    observable_arrays = [MMT_observable, Magellan_observable, McDonald_observable]

    return telescope_arrays, dates_arrays, airmasses_arrays, sun_elevations_arrays, observable_arrays


def plot_nature_mag_distance(sub_y, sub_x, sub_n, data_catalog, info_table):
    """
    Plot the host magnitude as a function of distance from transient
    And make the size equal to the probability of being a galaxy

    Parameters
    ----------
    sub_y : int
        Number of rows in the plot
    sub_x : int
        Number of columns in the plot
    sub_n : int
        Plot number
    data_catalog : astropy.table.Table
        Astropy table with catalog data
    info_table : astropy.table.Table
        Table with all the output FLEET information
    """

    # Peak Transient magnitude
    brightest_mag = info_table['brightest_mag'][0]

    # Get host information
    host_separation = info_table['host_separation'][0]
    host_magnitude = info_table['host_magnitude'][0]
    host_nature = info_table['host_nature'][0]

    # Define x limit from search radius and zoom in if possible
    search_radius = info_table['search_radius'][0]
    half_radius = (search_radius * 60) / 2
    if host_separation < half_radius:
        search_radius_use = half_radius
    else:
        search_radius_use = search_radius * 60

    # Get separation and host magnitude
    separations = data_catalog['separation']
    magnitudes = data_catalog['effective_magnitude']
    nature = data_catalog['object_nature']

    # Get the ones for the closest host
    closest_separation = info_table['closest_separation'][0]
    closest_nature = info_table['closest_nature'][0]
    closest_magnitude = info_table['closest_magnitude'][0]

    # Plot data
    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_yaxis()
    plt.title("Probability of being a Galaxy")
    plt.scatter(separations, magnitudes, s=(nature + 0.1)*1000, c=(nature + 0.1)*1000, vmin=0, vmax=1000, alpha=0.75)
    plt.scatter(host_separation, host_magnitude, marker='+', alpha=1.0, color='b', s=1000)
    plt.axhline(brightest_mag, color='b', linestyle='--', linewidth=1, label="Brightest Mag=%s" % np.around(brightest_mag, decimals=2))
    plt.axhline(host_magnitude, color='k', linestyle='-.', linewidth=1)
    if host_separation < search_radius_use + 2:
        plt.annotate(str(np.around(host_nature, decimals=2)), xy=(host_separation, host_magnitude))
        if host_separation != closest_separation:
            plt.annotate(str(np.around(closest_nature, decimals=2)), xy=(closest_separation, closest_magnitude))
    plt.legend(loc='upper right', fancybox=True)
    plt.xlim(0, search_radius_use + 2)
    plt.xlabel("Distance from Transient [Arcsec]")
    plt.ylabel("Effective Magnitude")


def plot_host_mag_distance(sub_y, sub_x, sub_n, data_catalog, info_table):
    """
    Plot the host magnitude as a function of distance from transient
    And make the size equal to the probability of being the host

    Parameters
    ----------
    sub_y : int
        Number of rows in the plot
    sub_x : int
        Number of columns in the plot
    sub_n : int
        Plot number
    data_catalog : astropy.table.Table
        Astropy table with catalog data
    info_table : astropy.table.Table
        Table with all the output FLEET information
    """

    # Get host nature
    object_nature = data_catalog['object_nature']
    star_cut = info_table['star_cut'][0]

    # Separate galaxies and stars
    galaxies = np.where(object_nature > star_cut)[0]
    stars = np.where(object_nature <= star_cut)[0]

    # Get catalog information
    separations = data_catalog['separation']
    magnitudes = data_catalog['effective_magnitude']
    chance_coincidence = data_catalog['chance_coincidence']
    brightest_mag = info_table['brightest_mag'][0]

    # Get host information
    host_separation = info_table['host_separation'][0]
    host_magnitude = info_table['host_magnitude'][0]
    host_Pcc = info_table['host_Pcc'][0]
    closest_Pcc = info_table['closest_Pcc'][0]

    # Get the ones for the closest host
    closest_separation = info_table['closest_separation'][0]
    closest_magnitude = info_table['closest_magnitude'][0]

    # Define x limit from search radius and zoom in if possible
    search_radius = info_table['search_radius'][0]
    half_radius = (search_radius * 60) / 2
    if host_separation < half_radius:
        search_radius_use = half_radius
    else:
        search_radius_use = search_radius * 60

    average_probability = 1 - chance_coincidence
    host_probability = 1 - host_Pcc
    closest_probability = 1 - closest_Pcc
    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_yaxis()
    plt.title("Probability of being the Host")
    plt.scatter(separations[galaxies], magnitudes[galaxies], s=(average_probability[galaxies] + 0.1) * 1000,
                c=(average_probability[galaxies] + 0.1) * 1000, vmin=0, vmax=1000, alpha=0.75)
    plt.scatter(separations[stars], magnitudes[stars], s=100, c='r', marker='x', alpha=0.75, label='Stars')
    plt.scatter(host_separation, host_magnitude, marker='+', alpha=1.0, color='b', s=1000)
    plt.axhline(float(brightest_mag), color='b', linestyle='--', linewidth=1)
    plt.axhline(float(host_magnitude), color='k', linestyle='-.', linewidth=1)
    if host_separation < search_radius_use + 2:
        if host_separation != closest_separation:
            plt.annotate(str(np.around(closest_probability, decimals=2)), xy=(closest_separation, closest_magnitude))
        plt.annotate(str(np.around(host_probability, decimals=2)), xy=(host_separation, host_magnitude))
    plt.legend(loc='upper right', fancybox=True)
    plt.xlim(0, search_radius_use + 2)
    plt.xlabel("Distance from Transient [Arcsec]")
    plt.ylabel("Effective Magnitude")


def extract_probabilities(info_table):
    """
    Extract and organize classification probabilities from the info_table.

    Parameters
    ----------
    info_table : astropy.table.Table
        Table with all the output FLEET information

    Returns
    -------
    prob_data : dict
        Dictionary with classification probabilities for late, rapid_slsn, and rapid_tde
    """
    # Define classifier types and output classes
    classifiers = ['late', 'rapid_slsn', 'rapid_tde']
    classes = ['AGN', 'SLSNI', 'SLSNII', 'SNII', 'SNI', 'Star', 'TDE']

    # Initialize results dictionary
    results = {classifier: {'probs': [], 'stds': []} for classifier in classifiers}

    # Process each classifier type
    for classifier in classifiers:
        # Map for combined classes
        combined_classes = {
            'SNII': ['SNII', 'SNIIb', 'SNIIn'],
            'SNI': ['SNIa', 'SNIbc']
        }

        # Process each class
        for class_name in classes:
            # Direct probability extraction
            if class_name in ['AGN', 'SLSNI', 'SLSNII', 'Star', 'TDE']:
                col_name = f"P_{classifier}_{class_name}"
                prob = info_table[col_name][0]
                std = info_table[f"{col_name}_std"][0]

            # Combined probability calculation
            else:  # SNI or SNII
                base_classes = combined_classes[class_name]
                # Calculate combined probability
                prob = sum(info_table[f"P_{classifier}_{bc}"][0] for bc in base_classes)

                # Calculate combined standard deviation using error propagation
                std = np.sqrt(sum(info_table[f"P_{classifier}_{bc}_std"][0]**2
                                  for bc in base_classes))

            results[classifier]['probs'].append(prob)
            results[classifier]['stds'].append(std)

    return {
        'late': (results['late']['probs'], results['late']['stds']),
        'slsn': (results['rapid_slsn']['probs'], results['rapid_slsn']['stds']),
        'tde': (results['rapid_tde']['probs'], results['rapid_tde']['stds'])
    }


def plot_host_information(sub_y, sub_x, sub_n, info_table):
    """
    Make a plot with information about the transient and its host galaxy.

    Parameters
    ----------
    sub_y : int
        Number of rows in the subplot grid.
    sub_x : int
        Number of columns in the subplot grid.
    sub_n : int
        Index of the current subplot.
    """
    # Empty array values to check against
    emptys = ['', ' ', 'None', '--', '-', b'', b' ', b'None', b'--', b'-',
              None, np.nan, 'nan', b'nan', '0', np.inf, 'inf']

    # Format Names
    object_name = info_table['object_name'][0]
    ZTF_name = info_table['ztf_name'][0] if info_table['ztf_name'][0] not in emptys else ''
    Rubin_name = ''
    if 'rubin_name' in info_table.colnames and info_table['rubin_name'][0] not in emptys:
        Rubin_name = info_table['rubin_name'][0]
    TNS_name = info_table['tns_name'][0] if info_table['tns_name'][0] not in emptys else ''

    # Don't duplicate names
    if object_name == ZTF_name:
        ZTF_name = ''
    if object_name == Rubin_name:
        Rubin_name = ''
    if object_name == TNS_name:
        TNS_name = ''

    # Get classification
    object_class = info_table['object_class'][0]

    # Coordinates Info - with fixed precision
    ra_deg = info_table['ra_deg'][0]
    dec_deg = info_table['dec_deg'][0]
    coords = SkyCoord(ra_deg, dec_deg, unit='deg')
    coord_str = coords.to_string('hmsdms', sep=':')
    RA_hms = coord_str[:coord_str.find(' ')][:12]
    DEC_dms = coord_str[coord_str.find(' ')+1:][:12]
    l_deg = coords.galactic.l.value
    b_deg = coords.galactic.b.value

    # Redshift Info
    redshift = info_table['redshift_use'][0]
    redshift_label = info_table['redshift_label'][0]

    # Check for photoz information
    photoz = None
    photoz_err = None
    if redshift_label != 'Photoz':
        if info_table['photoz'][0] not in emptys:
            photoz = float(info_table['photoz'][0])
        if info_table['photoz_err'][0] not in emptys:
            photoz_err = float(info_table['photoz_err'][0])

    # Get classifier probabilities with proper error handling
    try:
        # Extract all probabilities with a single function call
        prob_data = extract_probabilities(info_table)

        # Unpack probability data
        late_probs, late_stds = prob_data['late']
        slsn_probs, slsn_stds = prob_data['slsn']
        tde_probs, tde_stds = prob_data['tde']

        has_probabilities = True
    except Exception:
        # Handle any errors in probability extraction
        has_probabilities = False

    # Create the plot
    plt.subplot(sub_y, sub_x, sub_n)

    # Center-align the object names with proper spacing
    title = f"{object_name}"
    if TNS_name:
        title += f"     {TNS_name}"
    if ZTF_name:
        title += f"     {ZTF_name}"
    if Rubin_name:
        title += f"     {Rubin_name}"
    plt.title(title)

    # Build the text content
    content = []

    # Coordinates section - with fixed precision and aligned columns
    content.append(f"RA = {RA_hms:<14}    DEC = {DEC_dms}")
    content.append(f"   = {ra_deg:<14.6f}        = {dec_deg:.6f}")
    content.append(f"l  = {l_deg:<14.6f}      b = {b_deg:.6f}")
    content.append("")

    # Classification and magnitude info
    if object_class not in emptys:
        content.append(f"{object_class}")

    hostless_value = info_table['hostless'][0]
    if isinstance(hostless_value, str):
        is_hostless = hostless_value.strip().lower() == 'true'
    else:
        is_hostless = bool(hostless_value)

    if is_hostless:
        content.append("Hostless!")

    # Magnitude information
    if is_hostless:
        host_magnitude_r = info_table['host_limit_r'][0]
    else:
        host_magnitude_r = info_table['host_magnitude_r'][0]
    brightest_mag = info_table['brightest_mag'][0]
    red_brightest = info_table['red_brightest'][0]
    if red_brightest not in emptys:
        content.append(f"ΔM = {host_magnitude_r:.2f} - {red_brightest:.2f} = {(host_magnitude_r - red_brightest):.2f}")
    else:
        content.append(f"ΔM = {host_magnitude_r:.2f} - {brightest_mag:.2f} = {(host_magnitude_r - brightest_mag):.2f}")

    # Size and separation information
    host_radius = info_table['host_radius'][0]
    host_separation = info_table['host_separation'][0]
    content.append(f"Size = {host_radius:.2f}\"")
    content.append(f"Separation = {host_separation:.2f}\"")
    if np.isfinite(host_radius) and host_radius > 0:
        content.append(f"Offset = {(host_separation/host_radius):.2f} Re")
    else:
        content.append("Offset = -- Re")
    content.append("")

    # Light curve duration information
    first_to_peak_r = info_table['first_to_peak_r'][0]
    first_to_peak_g = info_table['first_to_peak_g'][0]
    peak_to_last_r = info_table['peak_to_last_r'][0]
    peak_to_last_g = info_table['peak_to_last_g'][0]

    if first_to_peak_r > 80:
        content.append(f"Warning, first to peak [r] = {first_to_peak_r:.2f} days")
    if peak_to_last_r < -30:
        content.append(f"Warning, peak to last [r] = {peak_to_last_r:.2f} days")
    if first_to_peak_g > 120:
        content.append(f"Warning, first to peak [g] = {first_to_peak_g:.2f} days")
    if peak_to_last_g < -30:
        content.append(f"Warning, peak to last [g] = {peak_to_last_g:.2f} days")
    content.append("")

    # Redshift information
    absolute_magnitude = '--'
    if redshift not in emptys:
        redshift_float = float(redshift)
        if np.isfinite(redshift_float):
            content.append(f"z = {redshift_float:.3f} ({redshift_label})")
            if redshift_float > 0:
                if red_brightest not in emptys:
                    absolute_magnitude = calc_absmag(red_brightest, redshift_float)
                else:
                    absolute_magnitude = calc_absmag(brightest_mag, redshift_float)

    # Add photometric redshift if available
    if photoz is not None and np.isfinite(photoz):
        if photoz_err is not None:
            content.append(f"z = {photoz:.3f}±{photoz_err:.3f} (photoz)")
        else:
            content.append(f"z = {photoz:.3f} (photoz)")

    # Add absolute magnitude if available
    if str(absolute_magnitude) not in emptys:
        content.append(f"Abs. Mag = {float(absolute_magnitude):.2f}")
    content.append("")

    # Format probability table if available
    if has_probabilities:
        # Create a more compact header that matches the data rows
        header = f"{'Class':4s} {'AGN':^4s} {'SLSNI':^4s} {'SLSNII':^4s} {'SNII':^4s} {'SNI':^5s} {'Star':^5s} {'TDE':^5s}"
        content.append(header)
        content.append("-" * len(header))

        # Format the probability rows with tight spacing
        def format_prob_row(name, probs, stds):
            row = f"{name:2s}"
            for p, s in zip(probs, stds):
                if np.isfinite(p) and np.isfinite(s):
                    row += f"{p:2.0f}±{s:<2.0f} "
                elif np.isfinite(p):
                    row += f"{p:2.0f}   "
                else:
                    row += "--    "
            return row

        # Add each probability row to content
        content.append(format_prob_row("Late ", late_probs, late_stds))
        content.append(format_prob_row("SLSN ", slsn_probs, slsn_stds))
        content.append(format_prob_row("TDE  ", tde_probs, tde_stds))

    # Remove axes and ticks
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
                    labelleft=False, labeltop=False, labelright=False, labelbottom=False)

    # Join all content with newlines and display as annotation
    full_text = '\n'.join(content)
    plt.annotate(full_text, xy=(0.02, 0.96), fontsize=10, va='top', family='monospace')

    # Set plot limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def calc_HET_observability(Dates, info_table, pupil_fraction=0.3):
    """
    Determine which dates are observable by HET.

    Parameters
    ----------
    Dates : array
        Array of dates to check for observability.
    info_table : astropy.table.Table
        Table with all the output FLEET information.
    pupil_fraction : float, default 0.3
        Fraction of the full pupil area that is considered observable.

    Returns
    -------
    het_mask : array
        Boolean array indicating which dates are observable by HET.
    """
    import pyHETobs
    import contextlib

    # Get object coordinates
    ra_deg = info_table['ra_deg'][0]
    dec_deg = info_table['dec_deg'][0]

    # Set up target object
    target = SkyCoord(ra_deg, dec_deg, unit=(u.deg, u.deg), frame='icrs')

    # McDonald Observatory location
    Obslocation = EarthLocation.of_site(u'mcdonald')
    mcdonald_longitude = -104.02166669 * u.deg  # Longitude of McDonald Observatory
    full_pupil_area = 78.8083117443839  # Full area of HET's pupil

    # Pick a time near the middle of the night at UT = 10
    mid_date = Time(Dates[-1].replace(hour=10, minute=0, second=0, microsecond=0))

    # Calculate the time of the zenith crossing
    HA_toStar = mid_date.sidereal_time('apparent', longitude=mcdonald_longitude) - target.ra
    ZenithCrossTime = mid_date - HA_toStar.hour * u.hour

    # Calculate the optimal azimuth to park the telescope
    optimal_azimuth_east, _, _ = pyHETobs.HET_Tracker.find_HET_optimal_azimuth(target, ZenithCrossTime, True)
    optimal_azimuth_west, _, _ = pyHETobs.HET_Tracker.find_HET_optimal_azimuth(target, ZenithCrossTime, False)

    # Suppress any output from pyHETobs
    print('Calculating HET observability ...')
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        # Setup the HET telescope
        HET_east = pyHETobs.TrackPupilSize.HET_Telescope(park_azimuth=optimal_azimuth_east)
        HET_west = pyHETobs.TrackPupilSize.HET_Telescope(park_azimuth=optimal_azimuth_west)

    # Empty variable
    max_area = np.zeros(len(Dates))

    for i in range(len(Dates)):
        obs_time = Dates[i]
        # Calcualte the object's altitude and azimuth
        ObjAltAz = target.transform_to(AltAz(obstime=obs_time, location=Obslocation))

        # Calculate the effective pupil area
        east_pupil = HET_east.get_effective_pupil(ObjAltAz)
        west_pupil = HET_west.get_effective_pupil(ObjAltAz)
        east_pupil_area = east_pupil.area
        west_pupil_area = west_pupil.area

        max_area[i] = np.max([east_pupil_area, west_pupil_area])

    # Create mask
    het_mask = np.zeros(len(Dates), dtype=bool)
    het_mask[max_area > full_pupil_area * pupil_fraction] = True
    return het_mask


def plot_observability(sub_y, sub_x, sub_n, info_table, telescope_arrays, dates_arrays, airmasses_arrays, sun_elevations_arrays,
                       include_het=False, pupil_fraction=0.3):
    """
    Plot airmass graphs of a given object for MMT and Magellan

    Parameters
    ----------
    sub_y : int
        Number of rows in the subplot
    sub_x : int
        Number of columns in the subplot
    sub_n : int
        Subplot number
    info_table : astropy.table.Table
        Table with all the output FLEET information
    telescope_arrays : list
        List of telescope names
    dates_arrays : list
        List of dates for each telescope
    airmasses_arrays : list
        List of airmasses for each telescope
    sun_elevations_arrays : list
        List of sun elevations for each telescope
    include_het : bool, default False
        If True, calculate observability specifically for HET
    """

    # Airmass plots
    axes = plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_yaxis()

    # Default colors for the plots
    colors = ['g', 'r', 'b', 'k', 'm', 'c', 'y']

    for i in range(len(telescope_arrays)):
        Telescope = telescope_arrays[i]
        Dates = dates_arrays[i]
        Airmass = airmasses_arrays[i]
        SunElevation = sun_elevations_arrays[i]

        # Plot the airmass
        if (Telescope != 'McDonald') or (not include_het):
            plt.plot(Dates, Airmass, color=colors[i], linewidth=4, label='%s Airmass' % telescope_arrays[i])
            plt.fill_between(Dates, 0, 3.0, where=SunElevation <= -18.0, facecolor=colors[i], alpha=0.25)
        elif Telescope == 'McDonald':
            het_mask = calc_HET_observability(Dates, info_table, pupil_fraction)
            airmass_true = np.copy(Airmass)
            airmass_true[~het_mask] = np.nan
            plt.plot(Dates, Airmass, color=colors[i], linewidth=1, linestyle='--')
            plt.plot(Dates, airmass_true, color=colors[i], linewidth=4, label='%s Airmass' % telescope_arrays[i])
            plt.fill_between(Dates, 0, 3.0, where=SunElevation <= -18.0, facecolor=colors[i], alpha=0.25)

    plt.fill_between([], [], facecolor='gray', alpha=0.25, label='Night')
    plt.legend(loc='upper left')
    plt.ylabel("Airmass")
    plt.title("UT=%s" % datetime.datetime.now().strftime('%Y-%m-%d') + "     MJD=%s" % np.around(Time(datetime.datetime.now()).mjd, decimals=1))
    plt.ylim(3, 0.5)
    # Calculate the minimum and maximum dates
    min_date = Dates[0].replace(hour=17, minute=0, second=0, microsecond=0)
    max_date = min_date + datetime.timedelta(hours=23)
    plt.xlim(min_date, max_date)

    # Change Axes Labels
    plt.xticks(rotation=0)
    axes.set_xticks([min_date + datetime.timedelta(hours=i) for i in range(0, 24, 2)])
    xfmt = md.DateFormatter('%H')
    axes.xaxis.set_major_formatter(xfmt)
    plt.xlabel('UT')


def plot_ra_dec_size(sub_y, sub_x, sub_n, data_catalog, info_table):
    """
    Plot the field objects in RA and DEC, where the size
    is the size of the objects

    Parameters
    ----------
    sub_y : int
        Number of rows in the plot
    sub_x : int
        Number of columns in the plot
    sub_n : int
        Plot number
    data_catalog : astropy.table.Table
        Astropy table with catalog data
    info_table : astropy.table.Table
        Table with all the output FLEET information
    """

    # Calculate the transient luminosity
    brightest_mag = info_table['brightest_mag'][0]
    Lum_Transient = 10 ** 9 * (10 ** (-0.4 * float(brightest_mag)))

    # Get catalog information
    objects_ra = data_catalog['ra_matched']
    objects_dec = data_catalog['dec_matched']
    separation = data_catalog['separation']
    output_nature = data_catalog['object_nature']
    halflight_radius = data_catalog['halflight_radius']

    # Get host information
    search_radius = info_table['search_radius'][0]
    ra_deg = info_table['ra_deg'][0]
    dec_deg = info_table['dec_deg'][0]
    host_ra = info_table['host_ra'][0]
    host_dec = info_table['host_dec'][0]

    # Within radius
    within = (separation < search_radius * 60)
    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_xaxis()
    plt.title("Probability of being a Galaxy (Size)")
    plt.scatter(objects_ra[within], objects_dec[within], s=(output_nature[within] + 0.1)*1000, c=(output_nature[within] + 0.1)*1000,
                vmin=0, vmax=1000, alpha=0.75)
    plt.scatter(ra_deg, dec_deg, marker='*', color='r', s=Lum_Transient)
    plt.scatter(host_ra, host_dec, marker='+', alpha=1.0, color='b', s=1000)
    plt.xlabel("RA")
    plt.ylabel("DEC")
    plt.ticklabel_format(useOffset=False)
    plt.xticks(np.round(plt.xticks()[0], 2), rotation=0)
    plt.yticks(np.round(plt.yticks()[0], 2))
    correction = np.cos(np.radians(dec_deg))
    plt.xlim(ra_deg + search_radius / 60 / correction, ra_deg - search_radius / 60 / correction)
    plt.ylim(dec_deg - search_radius / 60, dec_deg + search_radius / 60)
    for j in range(len(objects_ra[within])):
        if np.isfinite(halflight_radius[within][j]):
            plt.annotate(str(np.around(halflight_radius[within][j], decimals=2)), xy=(objects_ra[within][j], objects_dec[within][j]))


def plot_ra_dec_magnitude(sub_y, sub_x, sub_n, data_catalog, info_table):
    """
    Plot the field objects in RA and DEC, where the size
    is the magnitude of the objects

    Parameters
    ----------
    sub_y : int
        Number of rows in the plot
    sub_x : int
        Number of columns in the plot
    sub_n : int
        Plot number
    data_catalog : astropy.table.Table
        Astropy table with catalog data
    info_table : astropy.table.Table
        Table with all the output FLEET information
    """

    # Get catalog information
    objects_ra = data_catalog['ra_matched']
    objects_dec = data_catalog['dec_matched']
    hosts_magnitudes = data_catalog['effective_magnitude']

    # Get host information
    ra_deg = info_table['ra_deg'][0]
    dec_deg = info_table['dec_deg'][0]
    host_ra = info_table['host_ra'][0]
    host_dec = info_table['host_dec'][0]
    search_radius = info_table['search_radius'][0]
    brightest_mag = info_table['brightest_mag'][0]

    Lum_Transient = 10 ** 9 * (10 ** (-0.4 * float(brightest_mag)))
    Lum = 10 ** 9 * (10 ** (-0.4 * hosts_magnitudes))

    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_xaxis()
    plt.title("Magnitude")
    plt.scatter(objects_ra, objects_dec, s=40 * Lum, c=Lum, alpha=0.75)
    plt.scatter(ra_deg, dec_deg, marker='*', color='r', s=Lum_Transient)
    plt.scatter(host_ra, host_dec, marker='+', alpha=1.0, color='b', s=1000, label='Best Host')
    plt.legend(loc='upper right')
    plt.xlabel("RA")
    plt.xticks(np.round(plt.xticks()[0], 2), rotation=0)
    plt.yticks(np.round(plt.yticks()[0], 2))
    correction = np.cos(np.radians(dec_deg))
    plt.xlim(ra_deg + search_radius / 60 / correction, ra_deg - search_radius / 60 / correction)
    plt.ylim(dec_deg - search_radius / 60, dec_deg + search_radius / 60)
    plt.ticklabel_format(useOffset=False)


def plot_field_image(sub_y, sub_x, sub_n, info_table, image_color='r', autoscale=75,
                     images_dir='images'):
    """
    Plot the field image at the given RA and DEC from 3PI

    Parameters
    ----------
    sub_y : int
        Number of rows in the subplot
    sub_x : int
        Number of columns in the subplot
    sub_n : int
        Plot number in the subplot
    info_table : astropy.table.Table
        Table containing the transient info
    image_color : str, default 'r'
        Filter name for the image
    autoscale : int, default 75
        PS1 scaling (lower is more contrast)
    """

    # Create folder to store images, if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    # Get transient information
    object_name = info_table['object_name'][0]
    ra_deg = info_table['ra_deg'][0]
    dec_deg = info_table['dec_deg'][0]
    search_radius = info_table['search_radius'][0]

    # If the image exists, open it
    wcs_size = int(search_radius * 2 * 60)
    image_name = f'{images_dir}/{object_name}.jpeg'
    if not os.path.exists(image_name):
        img = query_PS1_image(ra_deg, dec_deg, image_color, wcs_size, autoscale)
        try:
            img.save(image_name, 'jpeg')
        except Exception:
            pass
    else:
        img = Image.open(image_name)

    # Plot the image
    plt.subplot(sub_y, sub_x, sub_n)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    if img == '--':
        plt.annotate('Outside PS1 footprint', xy=(0, 0))
        plt.set_xlim(-0.5, 1)
        plt.set_ylim(-1, 1)
    else:
        plt.imshow(np.array(img), cmap='viridis')

    # Plot center
    plt.scatter(wcs_size*2, wcs_size*2, marker='+', color='r')


def plot_lightcurve(sub_y, sub_x, sub_n, input_table, info_table, subtract_phase=0, add_phase=0,
                    plot_model=True, full_range=False, plot_comparison=True, plot_today=False):
    """
    Plot the light curve and model for a given transient

    Parameters
    ----------
    sub_y : int
        Number of rows in the subplot
    sub_x : int
        Number of columns in the subplot
    sub_n : int
        Subplot number
    input_table : astropy.table.Table
        Data with photometry
    info_table : astropy.table.Table
        Data with information about the transient
    subtract_phase : float, default 0
        Days to subtract from minimum x limit
    add_phase : float, default 0
        Days to add to maximum x limit
    plot_model : bool, default True
        If True, plot the model
    full_range : bool, default False
        If True, plot the entire photometry (True), or just
        the relevant portion (False)
    plot_comparison : bool, default True
        If True, plot a comparison Ia light curve
    plot_today : bool, default False
        If True, plot the current date in the light curve

    Returns
    -------
    used_filters : list
        List of filters used in the plot
    used_sources : list
        List of source categories used in the plot
    """

    # Get transient information
    first_mjd = info_table['first_mjd'][0]
    bright_mjd = info_table['bright_mjd'][0]

    # Extract light curve data
    all_magnitudes = input_table['Mag']
    all_times = input_table['MJD']
    all_sigmas = input_table['MagErr']
    all_filters = input_table['Filter']
    all_sources = input_table['Source'] if 'Source' in input_table.colnames else np.array([''] * len(input_table), dtype=str)
    all_telescopes = input_table['Telescope'] if 'Telescope' in input_table.colnames else np.array([''] * len(input_table), dtype=str)
    all_upperlimits = input_table['UL']
    all_ignores = input_table['Ignore']
    # If plotting the full range, use MJD as phase
    if full_range:
        all_phases = all_times
    else:
        all_phases = all_times - bright_mjd

    # Which are Upper limits and ignored data
    upper_limit = np.where((all_upperlimits == 'True') & (all_ignores == 'False'))[0]
    detection = np.where((all_upperlimits == 'False') & (all_ignores == 'False'))[0]
    # Ignored
    upper_ignore = np.where((all_upperlimits == 'True') & (all_ignores == 'True'))[0]
    detect_ignore = np.where((all_upperlimits == 'False') & (all_ignores == 'True'))[0]

    # Select Plot Colors
    all_colors = plot_colors(all_filters)

    # Select groups of data. Source='Alerce' is ambiguous now because it can
    # represent either ZTF or Rubin; use Telescope to separate them.
    det_groups = photometry_source_indices(all_sources[detection], all_telescopes[detection])
    is_det_ztf = det_groups['ztf']
    is_det_rubin = det_groups['rubin']
    is_det_local = det_groups['local']
    is_det_other = det_groups['other']

    # Set plot limits to ± 0.5 the magnitude limits
    if full_range:
        real_magnitudes = all_magnitudes[np.isfinite(all_magnitudes)]
    else:
        real_magnitudes = all_magnitudes[detection][np.isfinite(all_magnitudes[detection])]

    # Set Phase Limits
    # Minimum
    brightest_phase = bright_mjd - first_mjd
    if brightest_phase > 100:
        MJD_minimum = bright_mjd - 100
    else:
        MJD_minimum = first_mjd - 5
    phase_minimum = MJD_minimum - bright_mjd

    # Maximum
    latest_phase = np.nanmax(all_phases[detection])
    latest_mjd = np.nanmax(all_times[detection])
    if latest_phase >= 30:
        MJD_maximum = latest_mjd + 5
    else:
        MJD_maximum = latest_mjd + 35
    phase_maximum = MJD_maximum - bright_mjd

    # Set up plot
    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_yaxis()
    plt.ylim(np.nanmax(real_magnitudes) + 0.5, np.nanmin(real_magnitudes) - 0.5)

    # Include offsets if specified
    if full_range:
        mino, maxo = np.nanmin(all_times), np.nanmax(all_times)
    else:
        mino, maxo = phase_minimum - subtract_phase, phase_maximum + add_phase

    # Making sure the min and max are not the same, otherwise who cares
    if not plot_today:
        if mino != maxo:
            plt.xlim(mino, maxo)

    # Plot detections
    plt.errorbar(all_phases[detection][is_det_ztf], all_magnitudes[detection][is_det_ztf], all_sigmas[detection][is_det_ztf],
                 ecolor=all_colors[detection][is_det_ztf], fmt='d', alpha=0.8, ms=0)
    plt.errorbar(all_phases[detection][is_det_rubin], all_magnitudes[detection][is_det_rubin], all_sigmas[detection][is_det_rubin],
                 ecolor=all_colors[detection][is_det_rubin], fmt='s', alpha=0.8, ms=0)
    plt.errorbar(all_phases[detection][is_det_local], all_magnitudes[detection][is_det_local], all_sigmas[detection][is_det_local],
                 ecolor=all_colors[detection][is_det_local], fmt='*', alpha=0.8, ms=0)
    plt.errorbar(all_phases[detection][is_det_other], all_magnitudes[detection][is_det_other], all_sigmas[detection][is_det_other],
                 ecolor=all_colors[detection][is_det_other], fmt='.', alpha=0.8, ms=0)
    plt.scatter(all_phases[detection][is_det_ztf], all_magnitudes[detection][is_det_ztf],
                color=all_colors[detection][is_det_ztf], marker='d', alpha=0.8, s=90)
    plt.scatter(all_phases[detection][is_det_rubin], all_magnitudes[detection][is_det_rubin],
                color=all_colors[detection][is_det_rubin], marker='s', alpha=0.8, s=90)
    plt.scatter(all_phases[detection][is_det_local], all_magnitudes[detection][is_det_local],
                color=all_colors[detection][is_det_local], marker='*', alpha=0.8, s=90)
    plt.scatter(all_phases[detection][is_det_other], all_magnitudes[detection][is_det_other],
                color=all_colors[detection][is_det_other], marker='.', alpha=0.8, s=90)

    # Plot upper limits
    ul_alpha = 0.5
    plt.scatter(all_phases[upper_limit], all_magnitudes[upper_limit],
                color=all_colors[upper_limit], marker='v', alpha=ul_alpha, s=90)

    # Plot Ignored Data
    ignore_alpha = 0.15
    # Select groups of ignored detections
    ignored_groups = photometry_source_indices(all_sources[detect_ignore], all_telescopes[detect_ignore])
    was_det_ztf = ignored_groups['ztf']
    was_det_rubin = ignored_groups['rubin']
    was_det_local = ignored_groups['local']
    was_det_other = ignored_groups['other']

    # Plot ignored detections
    plt.errorbar(all_phases[detect_ignore][was_det_ztf], all_magnitudes[detect_ignore][was_det_ztf], all_sigmas[detect_ignore][was_det_ztf],
                 ecolor=all_colors[detect_ignore][was_det_ztf], fmt='d', alpha=ignore_alpha, ms=0)
    plt.errorbar(all_phases[detect_ignore][was_det_rubin], all_magnitudes[detect_ignore][was_det_rubin], all_sigmas[detect_ignore][was_det_rubin],
                 ecolor=all_colors[detect_ignore][was_det_rubin], fmt='s', alpha=ignore_alpha, ms=0)
    plt.errorbar(all_phases[detect_ignore][was_det_local], all_magnitudes[detect_ignore][was_det_local], all_sigmas[detect_ignore][was_det_local],
                 ecolor=all_colors[detect_ignore][was_det_local], fmt='*', alpha=ignore_alpha, ms=0)
    plt.errorbar(all_phases[detect_ignore][was_det_other], all_magnitudes[detect_ignore][was_det_other], all_sigmas[detect_ignore][was_det_other],
                 ecolor=all_colors[detect_ignore][was_det_other], fmt='.', alpha=ignore_alpha, ms=0)
    plt.scatter(all_phases[detect_ignore][was_det_ztf], all_magnitudes[detect_ignore][was_det_ztf],
                color=all_colors[detect_ignore][was_det_ztf], marker='d', alpha=ignore_alpha, s=90)
    plt.scatter(all_phases[detect_ignore][was_det_rubin], all_magnitudes[detect_ignore][was_det_rubin],
                color=all_colors[detect_ignore][was_det_rubin], marker='s', alpha=ignore_alpha, s=90)
    plt.scatter(all_phases[detect_ignore][was_det_local], all_magnitudes[detect_ignore][was_det_local],
                color=all_colors[detect_ignore][was_det_local], marker='*', alpha=ignore_alpha, s=90)
    plt.scatter(all_phases[detect_ignore][was_det_other], all_magnitudes[detect_ignore][was_det_other],
                color=all_colors[detect_ignore][was_det_other], marker='.', alpha=ignore_alpha, s=90)

    # Plot ignored upper limits
    plt.scatter(all_phases[upper_ignore], all_magnitudes[upper_ignore],
                color=all_colors[upper_ignore], marker='v', alpha=ignore_alpha, s=90)

    # Set Axis Labels
    if full_range:
        plt.tick_params(axis='both', top=False, bottom=True, labeltop=False, labelbottom=True)
        plt.xlabel('MJD')
    else:
        plt.tick_params(axis='both', top=False, bottom=True, labeltop=False, labelbottom=True)
        plt.xlabel('Days Since Brightest Point')

    plt.ylabel('Magnitude')

    # Get filter information and representative wavelengths from the actual data.
    used_filters = np.unique(all_filters)
    all_groups = photometry_source_indices(all_sources, all_telescopes)
    used_sources = []
    if len(all_groups['ztf']) > 0:
        used_sources.append('ZTF')
    if len(all_groups['rubin']) > 0:
        used_sources.append('Rubin')
    if len(all_groups['local']) > 0:
        used_sources.append('Local')
    if len(all_groups['other']) > 0:
        used_sources.append('Other')

    wavelengths = get_filter_wavelengths(used_filters, input_table)

    # Plot fits to the data
    if plot_model and not full_range:
        # Get model
        model = info_table['model'][0]

        if (model == 'double') or (model == 'single'):
            # Get model parameters
            lc_width_r = info_table['lc_width_r'][0]
            lc_decline_r = info_table['lc_decline_r'][0]
            phase_offset_r = info_table['phase_offset_r'][0]
            mag_offset_r = info_table['mag_offset_r'][0]
            lc_width_g = info_table['lc_width_g'][0]
            lc_decline_g = info_table['lc_decline_g'][0]
            phase_offset_g = info_table['phase_offset_g'][0]
            mag_offset_g = info_table['mag_offset_g'][0]

            # Create model
            model_time = np.linspace(-100, 200, 1000)
            if lc_width_r:
                model_red = linex(model_time, lc_width_r, lc_decline_r, phase_offset_r, mag_offset_r)
                plt.plot(model_time, model_red, color='r', linestyle=':', linewidth=0.5)
            if lc_width_g:
                model_green = linex(model_time, lc_width_g, lc_decline_g, phase_offset_g, mag_offset_g)
                plt.plot(model_time, model_green, color='g', linestyle=':', linewidth=0.5)

        elif model == 'full':
            # Get model parameters
            lc_width = info_table['lc_width'][0]
            lc_decline = info_table['lc_decline'][0]
            phase_offset = info_table['phase_offset'][0]
            mag_offset = info_table['mag_offset'][0]
            initial_temp = info_table['initial_temp'][0]
            cooling_rate = info_table['cooling_rate'][0]

            # Plot the light curve for each filter
            model_time = np.linspace(-100, 200, 1000)
            if lc_width:
                for filter_name, filter_wave in zip(used_filters, wavelengths):
                    # Plot the model
                    model_data = model_mag(model_time, filter_wave, lc_width, lc_decline,
                                           phase_offset, mag_offset, initial_temp, cooling_rate)
                    plt.plot(model_time, model_data, color=plot_colors(str(filter_name)), linestyle=':', linewidth=0.5)

        # Add chi^2 legend
        chi2 = info_table['chi2'][0]
        if isinstance(chi2, (int, float)) and np.isfinite(chi2):
            plt.plot([], [], color='k', linestyle=':', linewidth=0.5, label=r'$\chi^2 = %s$' % round(chi2, 2))
            plt.legend(loc='upper right')

    # Plot comparison SN Ia
    brightest_mag = info_table['brightest_mag'][0]
    if plot_comparison and not full_range and brightest_mag:
        plt.plot(phase_1a_model_r, magnitude_1a_model_r + brightest_mag, color='m', linestyle='--', linewidth=1)

    # Plot today's MJD date
    if plot_today:
        today = Time(datetime.datetime.now()).mjd
        # Days since latest observation
        latest_phase = today - latest_mjd
        plt.axvline(today, color='k', linestyle='--', linewidth=1.0, label=f'{latest_phase:.0f} days since latest')
        plt.legend(loc='upper right')

    if not full_range:
        return used_filters, used_sources


def plot_host_sed(sub_y, sub_x, sub_n, data_catalog, info_table):
    """
    Plot the SED of the best host galaxy.

    Parameters
    ----------
    sub_y : int
        Number of rows in the subplot
    sub_x : int
        Number of columns in the subplot
    sub_n : int
        Index of the subplot
    data_catalog : astropy.table.Table
        Catalog of objects in field
    info_table : astropy.table.Table
        Table with information about the best host
    """

    # Get best host index
    best_host = info_table['best_host'][0]

    # Flux conversion factor
    Jy = 10 ** (8.9 / 2.5) * u.Jy
    plt.subplot(sub_y, sub_x, sub_n)

    # If there is 3pi data, plot it
    if 'gKronMag_3pi' in data_catalog.colnames:
        # Get the best host index
        best_host = info_table['best_host'][0]

        # Get the data for the best host
        magnitudes = np.array([data_catalog[best_host]['gKronMag_3pi'],
                              data_catalog[best_host]['rKronMag_3pi'],
                              data_catalog[best_host]['iKronMag_3pi'],
                              data_catalog[best_host]['zKronMag_3pi'],
                              data_catalog[best_host]['yKronMag_3pi']])
        errors = np.array([data_catalog[best_host]['gKronMagErr_3pi'],
                           data_catalog[best_host]['rKronMagErr_3pi'],
                           data_catalog[best_host]['iKronMagErr_3pi'],
                           data_catalog[best_host]['zKronMagErr_3pi'],
                           data_catalog[best_host]['yKronMagErr_3pi']])
        wavelengths = np.array(list(psst_refs.values()))
        flux = 10 ** (-0.4 * magnitudes) * Jy.value * 1000
        flux_err = flux * (10 ** (0.4 * errors) - 1)

        plt.errorbar(wavelengths, flux, flux_err, fmt='o', color='c', alpha=0.7, label='3PI')
    if 'modelMag_g_sdss' in data_catalog.colnames:
        # Get the data for the best host
        magnitudes = np.array([data_catalog[best_host]['modelMag_u_sdss'],
                               data_catalog[best_host]['modelMag_g_sdss'],
                               data_catalog[best_host]['modelMag_r_sdss'],
                               data_catalog[best_host]['modelMag_i_sdss'],
                               data_catalog[best_host]['modelMag_z_sdss']])
        errors = np.array([data_catalog[best_host]['modelMagErr_u_sdss'],
                           data_catalog[best_host]['modelMagErr_g_sdss'],
                           data_catalog[best_host]['modelMagErr_r_sdss'],
                           data_catalog[best_host]['modelMagErr_i_sdss'],
                           data_catalog[best_host]['modelMagErr_z_sdss']])
        wavelengths = np.array(list(sdss_refs.values()))
        flux = 10 ** (-0.4 * magnitudes) * Jy.value * 1000
        flux_err = flux * (10 ** (0.4 * errors) - 1)

        plt.errorbar(wavelengths, flux, flux_err, fmt='o', color='orange', alpha=0.7, label='SDSS')

    # Setup plot
    plt.legend(loc='upper right')
    plt.title('Best Host SED')
    plt.xlim(2500, 13000)
    plt.ylim(ymin=0)
    plt.xlabel('Wavelength [\u212b]')
    plt.ylabel('Flux [mJy]')


def plot_coordinates(sub_y, sub_x, sub_n, data_catalog, info_table, acceptance_boxsize=1.5):
    """
    Plot the closest area and the coordinates of all the catalogs of the closest object.

    Parameters
    ----------
    sub_y : int
        Number of rows in the subplot
    sub_x : int
        Number of columns in the subplot
    sub_n : int
        Index of the subplot
    data_catalog : astropy.table.Table
        Catalog of objects in field
    info_table : astropy.table.Table
        Table with information about the best host
    acceptance_boxsize : float
        Objects inside this box were matched in catalog (arcsec)
    """

    # Get RA and DEC from the host
    closest_ra = info_table['closest_ra'][0]
    closest_dec = info_table['closest_dec'][0]

    # Get RA and DEC from the transient
    ra_deg = info_table['ra_deg'][0]
    dec_deg = info_table['dec_deg'][0]

    # Calculate separate separations
    delta_ra, delta_dec = calc_separations(ra_deg, dec_deg, closest_ra, closest_dec, separate=True)

    # Plot data
    plt.subplot(sub_y, sub_x, sub_n)
    # Offset
    o = acceptance_boxsize / 3600 / 2
    # plot square region
    x_box = np.array([closest_ra - o, closest_ra + o, closest_ra + o, closest_ra - o, closest_ra - o])
    y_box = np.array([closest_dec - o, closest_dec - o, closest_dec + o, closest_dec + o, closest_dec - o])
    plt.plot((x_box - closest_ra) * 3600, (y_box - closest_dec) * 3600, color='b', linewidth=1, alpha=0.5)

    plt.title('Closest Host')
    plt.scatter(0, 0, marker='+', color='b', s=90, alpha=0.5, zorder=10, label='Joint')
    plt.scatter(delta_ra, delta_dec, marker='*', color='b', s=200, alpha=0.7, label='Target')
    # Plot individual catalogs
    if 'raStack_3pi' in data_catalog.colnames:
        delta_ra_3pi, delta_dec_3pi = calc_separations(data_catalog['raStack_3pi'], data_catalog['decStack_3pi'], closest_ra, closest_dec, separate=True)
        plt.scatter(delta_ra_3pi, delta_dec_3pi, marker='o', color='g', alpha=0.5, label='3PI')
    if 'ra_sdss' in data_catalog.colnames:
        delta_ra_sdss, delta_dec_sdss = calc_separations(data_catalog['ra_sdss'], data_catalog['dec_sdss'], closest_ra, closest_dec, separate=True)
        plt.scatter(delta_ra_sdss, delta_dec_sdss, marker='o', color='r', alpha=0.5, label='SDSS')

    # Set limits
    plt.legend(loc='best')
    plt.xlim(-1, +1)
    plt.ylim(-1, +1)
    plt.gca().invert_xaxis()
    plt.xlabel('RA [arcsec]')
    plt.ylabel('DEC [arcsec]')


def plot_legend(sub_y, sub_x, sub_n, used_filters, used_sources):
    """
    Include a legend with the light curve information
    in the plot.

    Parameters
    ----------
    sub_y : int
        Number of rows in the subplot
    sub_x : int
        Number of columns in the subplot
    sub_n : int
        Subplot number
    used_filters : list
        List of filters used in the plot
    used_sources : list
        List of sources used in the plot
    """

    # Select Plot Colors
    all_colors = plot_colors(used_filters)

    # Select groups of data source categories
    used_sources = np.asarray(used_sources, dtype=str)
    is_det_ztf = np.flatnonzero(np.isin(used_sources, ['ZTF', 'Alerce']))
    is_det_rubin = np.flatnonzero(np.isin(used_sources, ['Rubin', 'LSST']))
    is_det_local = np.flatnonzero(np.isin(used_sources, ['Local', 'FLWO']))
    is_det_other = np.where(~np.isin(used_sources, ['ZTF', 'Alerce', 'Rubin', 'LSST', 'Local', 'FLWO']))[0]

    plt.subplot(sub_y, sub_x, sub_n)
    if len(is_det_ztf) > 0:
        plt.scatter([], [], marker='d', alpha=1.0, s=90, color='k', label='ZTF')
    if len(is_det_rubin) > 0:
        plt.scatter([], [], marker='s', alpha=1.0, s=90, color='k', label='Rubin')
    if len(is_det_local) > 0:
        plt.scatter([], [], marker='*', alpha=1.0, s=90, color='k', label='Local')
    if len(is_det_other) > 0:
        plt.scatter([], [], marker='o', alpha=1.0, s=90, color='k', label='Other')

    for i in range(len(used_filters)):
        plt.scatter([], [], marker='o', alpha=1.0, s=90, color=all_colors[i], label=used_filters[i])

    plt.legend(loc='center', frameon=False)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False,
                    labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    plt.axis('off')


def make_plot(input_table, output_table, data_catalog, info_table, telescope_arrays, dates_arrays,
              airmasses_arrays, sun_elevations_arrays, plot_dir='plots', do_observability=True,
              include_het=False, pupil_fraction=0.3):
    """
    Create a diagnosis plot with the light curve, field image, and all other information about
    a transient.

    Parameters
    ----------
    input_table : astropy.table.Table
        Data with photometry
    output_table : astropy.table.Table
        Data with information about the transient
    data_catalog : astropy.table.Table
        Astropy table with catalog data
    info_table : astropy.table.Table
        Table with all the output FLEET information
    telescope_arrays : list
        List of names for all telescopes for which
        to plot the airmass curves
    dates_arrays : list
        List of date arrays for each telescope
    airmasses_arrays : list
        List of airmass arrays for each telescope
    sun_elevations_arrays : list
        List of sun elevation arrays for each telescope
    plot_dir : str
        Directory to save the plots
    do_observability : bool
        Whether to plot observability information
    include_het : bool
        Whether to include visibility from HET
    pupil_fraction : float
        Fraction of the pupil to use for HET observability
    """

    # Get object name
    object_name = info_table['object_name'][0]

    # Set up plot
    plt.close('all')
    plt.figure(figsize=(24, 18))
    plt.subplots_adjust(hspace=0.25)

    # Plot magnitude as a function of distance, coded by galaxyness
    plot_nature_mag_distance(3, 4, 1, data_catalog, info_table)
    # Plot magnitude as a function of distance, coded by probability of being the host
    plot_host_mag_distance(3, 4, 2, data_catalog, info_table)
    # Plot host information
    plot_host_information(3, 4, 3, info_table)
    # Plot observability
    if do_observability:
        plot_observability(3, 4, 4, info_table, telescope_arrays, dates_arrays, airmasses_arrays, sun_elevations_arrays, include_het,
                           pupil_fraction=pupil_fraction)
    # Plot RA and DEC coded by size
    plot_ra_dec_size(3, 4, 5, data_catalog, info_table)
    # Plot RA ande DEC coded by magnitude
    plot_ra_dec_magnitude(3, 4, 6, data_catalog, info_table)
    # Plot image of the field
    plot_field_image(3, 4, 7, info_table)
    # Plot zoomed in light curve
    used_filters, used_sources = plot_lightcurve(3, 4, 8, output_table, info_table, full_range=False)
    # Plot host galaxy SED
    plot_host_sed(3, 4, 9, data_catalog, info_table)
    # Plot closest zoom in to the transient
    plot_coordinates(3, 4, 10, data_catalog, info_table)
    # Plot the legend with light curve information
    plot_legend(3, 4, 11, used_filters, used_sources)
    # Plot zoomed out light curve
    plot_lightcurve(3, 4, 12, input_table, info_table, full_range=True)

    plt.savefig(f'{plot_dir}/{object_name}_output.pdf', bbox_inches='tight')
    plt.clf()
    plt.close('all')


def quick_plot(input_table, info_table, plot_dir='plots'):
    """
    Create a quick diagnosis plot with just the light curve of a transient.

    Parameters
    ----------
    input_table : astropy.table.Table
        Data with photometry
    info_table : astropy.table.Table
        Table with all the output FLEET information
    plot_dir : str
        Directory to save the plots
    """

    # Create folder to store images, if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Get object information
    object_name = info_table['object_name'][0]
    object_class = info_table['object_class'][0]

    # Set up plot
    plt.figure(figsize=(8, 7))
    plt.subplots_adjust(hspace=0.2)

    # Plot Lightcurve
    plot_lightcurve(2, 1, 1, input_table, info_table, full_range=False)
    plt.title(f'{object_name} - {object_class}')
    plt.xlim(xmax=100)

    # Plot full range light curve
    plot_lightcurve(2, 1, 2, input_table, info_table, full_range=True)
    plt.savefig(f'{plot_dir}/{object_name}_quick.pdf', bbox_inches='tight')
    plt.clf()
    plt.close('all')


def calculate_purity_completeness(val_table, class_column='true_class_name', target_class='SLSN-I',
                                  threshold_steps=100):
    """
    Calculate both the purity and completeness as a function of probability threshold for
    a given validation table and object class.

    Parameters
    ----------
    val_table : astropy.table.Table
        The input table with probability and class columns
    class_column : str
        Column name for the class labels
    target_class : str
        The class name we're calculating metrics for
    threshold_steps : int
        Number of threshold steps to evaluate

    Returns
    -------
    thresholds : numpy.ndarray
        Array of threshold values
    purities : numpy.ndarray
        Corresponding purity values for each threshold
    completeness : numpy.ndarray
        Corresponding completeness values for each threshold
    """

    # Get the probabilities and true classes
    probs = val_table[target_class]
    true_classes = val_table[class_column]

    # Count total number of target class instances
    total_target_class = np.sum(true_classes == target_class)

    # Create array of threshold values
    thresholds = np.linspace(0, 1, threshold_steps)
    purities = np.zeros(threshold_steps)
    completeness = np.zeros(threshold_steps)

    # Calculate metrics for each threshold
    for i, threshold in enumerate(thresholds):
        # Find samples above threshold
        above_threshold = probs > threshold

        # Count samples above threshold
        n_samples_above = np.sum(above_threshold)

        # Count true positives (correct predictions above threshold)
        true_positives = np.sum((probs > threshold) & (true_classes == target_class))

        # Calculate purity (precision)
        purities[i] = true_positives / n_samples_above if n_samples_above > 0 else 0

        # Calculate completeness (recall)
        completeness[i] = true_positives / total_target_class if total_target_class > 0 else 0

    return thresholds, purities, completeness


def plot_metric_curves(training_days, testing_days, grouping, n_estimators, max_depth, features, model, clean,
                       chunk_size, metric='both', output_dir='validations', class_column='true_class_name',
                       target_class='SLSN-I', threshold_steps=100, file_pattern=None):
    """
    Plot the mean purity or completeness curve with 1-sigma confidence interval.

    Parameters
    ----------
    training_days : float
        The number of days of photometry to use for training.
    testing_days : float
        The number of days of photometry to use for testing.
    grouping : str
        The grouping of transient types to use for training.
    n_estimators : int
        The number of trees in the random forest.
    max_depth : int
        The maximum depth of the trees in the random forest.
    features : int
        The index of the feature list to use for training.
    model : str, either 'single' or 'double'
        The model to use for training.
    clean : bool
        Whether to require objects to have a host galaxy.
    chunk_size : int
        Number of items to remove at a time during leave-out. Default is 1.
    metric : str
        Which metric to plot ('purity', 'completeness', or 'both')
    output_dir : str
        Directory containing validation tables and for output plots
    class_column : str
        Column name for the class labels
    target_class : str
        The class name we're calculating metrics for
    threshold_steps : int
        Number of threshold steps to evaluate
    file_pattern : str
        Optional pattern to filter files in the output directory
    """

    # Find all validation tables
    if file_pattern is None:
        file_pattern = (
            f'{output_dir}/{training_days}_{testing_days}_{grouping}_*_{n_estimators}_'
            f'{max_depth}_*_*_{features}_{model}_{clean}_{chunk_size}.txt'
        )
    table_files = glob.glob(file_pattern)

    # Read all tables
    tables = []
    for file_path in table_files:
        val_table = table.Table.read(file_path, format='ascii')
        tables.append(val_table)
        print(f"Loaded {file_path}, shape: {len(val_table)}")

    # Format title
    title = (
        f"Train={training_days} Test={testing_days} Group={grouping} Estimators={n_estimators} Depth={max_depth}\n"
        f"Features={features} Model={model} Clean={clean} Chunk={chunk_size}"
    )

    # Use a common set of threshold values for all tables
    common_thresholds = np.linspace(0, 1, threshold_steps)
    all_purities = []
    all_completeness = []

    # Calculate metrics for each table
    for val_table in tables:
        thresholds, purities, completeness = calculate_purity_completeness(
            val_table,
            class_column=class_column,
            target_class=target_class,
            threshold_steps=threshold_steps
        )
        all_purities.append(purities)
        all_completeness.append(completeness)

    # Convert to numpy arrays for calculations
    all_purities = np.array(all_purities)
    all_completeness = np.array(all_completeness)

    # Calculate mean and standard deviation
    mean_purity = np.mean(all_purities, axis=0)
    std_purity = np.std(all_purities, axis=0)
    mean_completeness = np.mean(all_completeness, axis=0)
    std_completeness = np.std(all_completeness, axis=0)

    # Plot based on specified metric
    if metric.lower() in ['purity']:
        # Plot mean purity line
        plt.figure(figsize=(8, 6))
        plt.plot(common_thresholds, mean_purity, 'g', label='Purity')

        # Plot 1-sigma confidence interval
        plt.fill_between(
            common_thresholds,
            np.maximum(0, mean_purity - std_purity),
            np.minimum(1, mean_purity + std_purity),
            color='g', alpha=0.2, linewidth=0
        )

        # Add labels and settings
        plt.xlabel(f'P({target_class})')
        plt.ylabel('Purity')
        plt.ylim(0, 1.0)
        plt.xlim(0, 1.0)
        plt.title(title)

        # Save purity plot
        output_file = (
            f'{output_dir}/purity_{target_class}_{training_days}_{testing_days}_{grouping}_'
            f'{n_estimators}_{max_depth}_{features}_{model}_{clean}_{chunk_size}.pdf'
        )
        plt.savefig(output_file, bbox_inches='tight')
        plt.clf()

    if metric.lower() in ['completeness']:

        # Plot mean completeness line
        plt.figure(figsize=(8, 6))
        plt.plot(common_thresholds, mean_completeness, 'b', label='Completeness')

        # Plot 1-sigma confidence interval
        plt.fill_between(
            common_thresholds,
            np.maximum(0, mean_completeness - std_completeness),
            np.minimum(1, mean_completeness + std_completeness),
            color='b', alpha=0.2, linewidth=0
        )

        # Add labels and settings
        plt.xlabel(f'P({target_class})')
        plt.ylabel('Completeness')
        plt.ylim(0, 1.0)
        plt.xlim(0, 1.0)
        plt.title(title)

        # Save completeness plot
        output_file = (f'{output_dir}/completeness_{target_class}_{training_days}_{testing_days}_{grouping}_{n_estimators}_'
                       f'{max_depth}_{features}_{model}_{clean}_{chunk_size}.pdf')

        plt.savefig(output_file, bbox_inches='tight')
        plt.clf()

    # If both metrics are requested, create a combined plot
    if metric.lower() == 'both':

        # Plot both metrics
        plt.figure(figsize=(8, 6))
        plt.plot(common_thresholds, mean_purity, 'g', label='Purity')
        plt.plot(common_thresholds, mean_completeness, 'b', label='Completeness')

        # Plot 1-sigma confidence intervals
        plt.fill_between(
            common_thresholds,
            np.maximum(0, mean_purity - std_purity),
            np.minimum(1, mean_purity + std_purity),
            color='g', alpha=0.1, linewidth=0
        )
        plt.fill_between(
            common_thresholds,
            np.maximum(0, mean_completeness - std_completeness),
            np.minimum(1, mean_completeness + std_completeness),
            color='b', alpha=0.1, linewidth=0
        )

        # Add labels and settings
        plt.xlabel(f'P({target_class})')
        plt.ylabel('Purity / Completeness')
        plt.ylim(0, 1.0)
        plt.xlim(0, 1.0)
        plt.title(title)
        plt.legend(loc='lower center')

        # Save combined plot
        output_file = (
            f'{output_dir}/combined_{target_class}_{training_days}_{testing_days}_{grouping}_'
            f'{n_estimators}_{max_depth}_{features}_{model}_{clean}_{chunk_size}.pdf'
        )
        plt.savefig(output_file, bbox_inches='tight')

    plt.close('all')


def plot_confusion_matrix(training_days, testing_days, classes_names, grouping, n_estimators, max_depth, features, model,
                          clean, chunk_size, output_dir='validations', norm=True, prob_threshold=0.5, figsize=(10, 8),
                          file_pattern=None):
    """
    Plot a confusion matrix showing the mean values and standard deviations across multiple validation tables.

    Parameters
    ----------
    training_days : float
        The number of days of photometry used for training.
    testing_days : float
        The number of days of photometry used for testing.
    grouping : str
        The grouping of transient types used for training.
    n_estimators : int
        The number of trees in the random forest.
    max_depth : int
        The maximum depth of the trees in the random forest.
    features : int
        The index of the feature list used for training.
    model : str, either 'single' or 'double'
        The model used for training.
    clean : bool
        Whether objects were required to have a host galaxy.
    chunk_size : int
        Number of items removed at a time during leave-out.
    output_dir : str
        Directory containing validation tables and for output plots
    norm : bool, default=True
        Whether to normalize the confusion matrix by row (true classes)
    prob_threshold : float, default=0.5
        Probability threshold for class assignment
    figsize : tuple, default=(10, 8)
        Figure size for the confusion matrix plot
    file_pattern : str, optional
        Optional pattern to filter files in the output directory
    """

    # Find all validation tables
    if file_pattern is None:
        file_pattern = (
            f'{output_dir}/{training_days}_{testing_days}_{grouping}_*_{n_estimators}_'
            f'{max_depth}_*_*_{features}_{model}_{clean}_{chunk_size}.txt'
        )
    table_files = glob.glob(file_pattern)

    # Read all tables
    tables = []
    for file_path in table_files:
        val_table = table.Table.read(file_path, format='ascii')
        tables.append(val_table)
        print(f"Loaded {file_path}, shape: {len(val_table)}")

    class_indices = list(classes_names.values())
    labels = list(classes_names.keys())
    n_classes = len(class_indices)

    # Initialize matrices for all tables
    confusion_matrices = []
    raw_counts_matrices = []

    # Calculate confusion matrix for each table
    for val_table in tables:
        # Initialize confusion matrix for this table
        cm = np.zeros((n_classes, n_classes))

        # Calculate confusion matrix
        for true_class_idx in class_indices:
            true_class_mask = val_table['true_class'] == true_class_idx
            true_class_samples = val_table[true_class_mask]
            true_class_samples_values = true_class_samples[labels]

            if len(true_class_samples) == 0:
                continue

            for pred_class_idx in class_indices:
                pred_count = 0
                for row in true_class_samples_values:
                    probs = np.array(row.as_void().tolist())
                    max_prob_idx = np.argmax(probs)
                    if probs[max_prob_idx] > prob_threshold and classes_names[labels[max_prob_idx]] == pred_class_idx:
                        pred_count += 1

                cm[class_indices.index(true_class_idx)][class_indices.index(pred_class_idx)] = pred_count

        raw_counts_matrices.append(cm.copy())

        # Normalize if requested
        if norm:
            cm_type = 'Completeness'
            row_sums = cm.sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
                cm_norm = cm / row_sums[:, np.newaxis]
            cm_norm[np.isnan(cm_norm)] = 0
            confusion_matrices.append(cm_norm)
        else:
            cm_type = 'Purity'
            row_sums = cm.sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
                cm_norm = cm / row_sums[np.newaxis, :]
            cm_norm[np.isnan(cm_norm)] = 0
            confusion_matrices.append(cm_norm)

    # Convert list of matrices to 3D array
    confusion_matrices = np.array(confusion_matrices)
    raw_counts_matrices = np.array(raw_counts_matrices)

    # Calculate mean and standard deviation across all matrices
    mean_cm = np.mean(confusion_matrices, axis=0)
    std_cm = np.std(confusion_matrices, axis=0)
    mean_raw_counts = np.mean(raw_counts_matrices, axis=0)

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot the mean values with a color scale
    plt.imshow(mean_cm, cmap='Blues', vmin=0, vmax=1 if norm else None)

    # Set ticks and labels
    plt.xticks(np.arange(n_classes), labels, rotation=45, ha='right')
    plt.yticks(np.arange(n_classes), labels)

    # Add title
    plt.title(f"{cm_type} Confusion Matrix (p > {prob_threshold})\nTrain={training_days} Test={testing_days} Group={grouping} "
              f"Estimators={n_estimators} Depth={max_depth}\n"
              f"Features={features} Model={model} Clean={clean} Chunk={chunk_size}")

    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    # Add text annotations for mean and std values
    for i in range(n_classes):
        for j in range(n_classes):
            # Display as percentage with error
            percentage = mean_cm[i, j] * 100
            if confusion_matrices.shape[0] > 1:  # Only show std if we have multiple tables
                std_percentage = std_cm[i, j] * 100
                text = f"{percentage:.1f}\n±{std_percentage:.1f}%"
            else:
                text = f"{percentage:.1f}%"

            # Add raw count in parentheses
            text += f"\n({int(mean_raw_counts[i, j])})"

            # Choose text color based on the cell's value (white for dark cells)
            text_color = 'white' if mean_cm[i, j] > 0.5 else 'black'
            plt.text(j, i, text, ha='center', va='center', color=text_color, fontsize=9)

    # Tight layout to ensure all elements are visible
    plt.tight_layout()

    # Save the plot
    output_file = (f'{output_dir}/{cm_type}_matrix_{training_days}_{testing_days}_{grouping}_'
                   f'{n_estimators}_{max_depth}_{features}_{model}_{clean}_{chunk_size}_{norm}.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def plot_purity_grid(training_days_arr, testing_days_arr, grouping, n_estimators, max_depth, features, model, clean,
                     chunk_size, metric='Purity', output_dir='validations', class_column='true_class_name',
                     target_class='SLSN-I', threshold_steps=101, prob_threshold=0.5):
    """
    Compare the purity for a training set using different testing sets,
    not necessarily the same one.

    Parameters
    ----------
    training_days_arr : array
        Array of the number of days of photometry to use for training.
    testing_days_arr : array
        Array of the number of days of photometry to use for testing.
    grouping : str
        The grouping of transient types to use for training.
    n_estimators : int
        The number of trees in the random forest.
    max_depth : int
        The maximum depth of the trees in the random forest.
    features : int
        The index of the feature list to use for training.
    model : str, either 'single' or 'double'
        The model to use for training.
    clean : bool
        Whether to require objects to have a host galaxy.
    chunk_size : int
        Number of items to remove at a time during leave-out. Default is 1.
    metric : str
        Which metric to plot ('purity', 'completeness', or 'both')
    output_dir : str
        Directory containing validation tables and for output plots
    class_column : str
        Column name for the class labels
    target_class : str
        The class name we're calculating metrics for
    threshold_steps : int
        Number of threshold steps to evaluate
    prob_threshold : float
        Probability threshold
    """

    final_grid = np.zeros((len(training_days_arr), len(testing_days_arr))) * np.nan

    for i in range(len(training_days_arr)):
        for j in range(len(testing_days_arr)):
            training_days = training_days_arr[i]
            testing_days = testing_days_arr[j]
            # Find all validation tables
            file_pattern = (
                f'{output_dir}/{training_days}_{testing_days}_{grouping}_*_{n_estimators}_'
                f'{max_depth}_*_*_{features}_{model}_{clean}_{chunk_size}.txt'
            )
            table_files = glob.glob(file_pattern)

            # Read all tables
            tables = []
            for file_path in table_files:
                val_table = table.Table.read(file_path, format='ascii')
                tables.append(val_table)
                print(f"Loaded {file_path}, shape: {len(val_table)}")

            # Use a common set of threshold values for all tables
            all_purities = []
            all_completeness = []

            # Calculate metrics for each table
            for val_table in tables:
                thresholds, purities, completeness = calculate_purity_completeness(
                    val_table,
                    class_column=class_column,
                    target_class=target_class,
                    threshold_steps=threshold_steps
                )
                all_purities.append(purities)
                all_completeness.append(completeness)

            # Convert to numpy arrays for calculations
            all_purities = np.array(all_purities)
            all_completeness = np.array(all_completeness)

            # Calculate mean and standard deviation
            mean_purity = np.mean(all_purities, axis=0)
            mean_completeness = np.mean(all_completeness, axis=0)

            common_thresholds = np.linspace(0, 1, threshold_steps)
            if metric.lower() == 'purity':
                if np.iterable(mean_purity):
                    final_grid[i, j] = mean_purity[common_thresholds == prob_threshold][0]
            elif metric.lower() == 'completeness':
                if np.iterable(mean_completeness):
                    final_grid[i, j] = mean_completeness[common_thresholds == prob_threshold][0]

    # Create the plot
    plt.figure(figsize=(8, 8))

    # Plot the mean values with a color scale
    plt.imshow(final_grid, cmap='Greens', vmin=0, vmax=1)

    # # Set ticks and labels
    plt.xticks(range(len(testing_days_arr)), testing_days_arr)
    plt.yticks(range(len(training_days_arr)), training_days_arr)

    # # Add title
    plt.title(f"{metric} Comparison p({target_class}) > {prob_threshold}\nGroup={grouping} "
              f"Estimators={n_estimators} Depth={max_depth}\n"
              f"Features={features} Model={model} Clean={clean} Chunk={chunk_size}")

    plt.xlabel('Testing Days')
    plt.ylabel('Training Days')

    # Add text annotations for mean and std values
    for i in range(len(training_days_arr)):
        for j in range(len(testing_days_arr)):
            # Display as percentage with error
            text = f"{final_grid[i, j] * 100:.1f}%"

            # Choose text color based on the cell's value (white for dark cells)
            text_color = 'white' if final_grid[i, j] > 0.5 else 'black'
            plt.text(j, i, text, ha='center', va='center', color=text_color, fontsize=9)

    # Tight layout to ensure all elements are visible
    plt.tight_layout()

    # Save the plot
    output_file = (f'{output_dir}/{metric}_compare_{grouping}_'
                   f'{n_estimators}_{max_depth}_{features}_{model}_{clean}_{chunk_size}_{target_class}.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()
    plt.close('all')
