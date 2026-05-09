import zipfile
from .model import get_table_cenwaves
from dustmaps.sfd import SFDQuery
from collections import OrderedDict
import json
import pathlib
import requests
import os
from alerce.core import Alerce
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import table
from xml.etree import ElementTree
from dust_extinction.parameter_averages import G23

try:
    fleet_data = os.environ['fleet_data']
except KeyError:
    fleet_data = os.path.join(os.path.dirname(__file__), 'data')

# Define possible empty values
empties = ['', ' ', 'None', '--', '-', b'', b' ', b'None', b'--', b'-', None, np.nan, 'nan', b'nan', '0', 0]


def calc_separations(ra_array, dec_array, ra_center, dec_center,
                     separate=False):
    """
    Calculate the separation in arcseconds between arrays of
    coordinates and a central position using astropy.

    Parameters
    ----------
    ra_array : np.array
        Array of Right Ascension values in degrees
    dec_array : np.array
        Array of Declination values in degrees
    ra_center : float
        Central Right Ascension in degrees
    dec_center : float
        Central Declination in degrees
    separate : bool, optional
        If True, return separate arrays for RA and Dec
        (Only valid for small angles)

    Returns
    -------
    sep : np.array
        Array of separations in arcseconds
    """

    if separate:
        # Calculate simple differences
        delta_dec = (dec_array - dec_center) * 3600

        # For RA, we need to account for the cos(dec) factor
        # We use the mean declination for the correction as we're assuming small distances
        cos_dec = np.cos(np.radians(dec_center))
        delta_ra = ((ra_array - ra_center) * cos_dec) * 3600

        return delta_ra, delta_dec
    else:
        # Create SkyCoord objects for comparison
        c1 = SkyCoord(ra_array*u.deg, dec_array*u.deg)
        c2 = SkyCoord(ra_center*u.deg, dec_center*u.deg)

        # Calculate the separation using astropy
        sep = c1.separation(c2).arcsec

        return sep


def convert_coords(ra_in, dec_in):
    """
    Convert RA and DEC from either sexagesimal string format
    or decimal degrees to decimal degrees.

    Parameters
    ----------
    ra_in : str, float, int, list, or numpy.ndarray
        Right ascension in either:
        - Sexagesimal string format ('hh:mm:ss', 'hh mm ss', etc.)
        - Decimal degrees (float or integer)
        - Single-element list or numpy array

    dec_in : str, float, int, list, or numpy.ndarray
        Declination in either:
        - Sexagesimal string format ('+dd:mm:ss', '-dd mm ss', etc.)
        - Decimal degrees (float or integer)
        - Single-element list or numpy array

    Returns
    -------
    ra_deg : float
        Right ascension in decimal degrees

    dec_deg : float
        Declination in decimal degrees
    """

    # Extract values from numpy arrays or lists if needed
    if isinstance(ra_in, np.ndarray):
        if len(ra_in) == 1:
            ra_in = ra_in.item()  # Convert single-element array to scalar
        else:
            raise ValueError("RA input must be a single value, not an array with multiple elements")

    if isinstance(ra_in, list):
        if len(ra_in) == 1:
            ra_in = ra_in[0]  # Extract the single value from the list
        else:
            raise ValueError("RA input must be a single value, not a list with multiple elements")

    if isinstance(dec_in, np.ndarray):
        if len(dec_in) == 1:
            dec_in = dec_in.item()  # Convert single-element array to scalar
        else:
            raise ValueError("DEC input must be a single value, not an array with multiple elements")

    if isinstance(dec_in, list):
        if len(dec_in) == 1:
            dec_in = dec_in[0]  # Extract the single value from the list
        else:
            raise ValueError("DEC input must be a single value, not a list with multiple elements")

    # Check that RA and DEC have compatible types
    if not (isinstance(ra_in, (str, float, int)) and isinstance(dec_in, (str, float, int))):
        raise TypeError("RA and DEC must be strings, floats, or integers")

    # Allow float/int combination
    if (isinstance(ra_in, (float, int)) and isinstance(dec_in, (float, int))):
        # Both are numeric, assume they are already in degrees
        return float(ra_in), float(dec_in)

    elif isinstance(ra_in, str) and isinstance(dec_in, str):
        # Both are strings, try to determine if they're in degrees or sexagesimal
        try:
            # First try: both might be numeric strings representing degrees
            ra_deg = float(ra_in)
            dec_deg = float(dec_in)
            return ra_deg, dec_deg
        except ValueError:
            # Second try: they might be sexagesimal strings
            try:
                coord = SkyCoord(ra_in, dec_in, unit=(u.hourangle, u.deg))
                return coord.ra.deg, coord.dec.deg
            except Exception as e:
                raise ValueError(f"Could not parse coordinates: {ra_in}, {dec_in}. Error: {e}")

    else:
        # Types are different and not both numeric
        raise TypeError("RA and DEC must be of the same type (both strings or both numeric)")


def transient_origin(object_name_in):
    '''
    Determine the origin of a transient based on its name and standardize the name format.

    Parameters
    ----------
    object_name_in : str
        Original transient name (e.g. 'AT2019abc', 'ZTF19abcdef', 'SN2020xyz')

    Returns
    -------
    transient_source : str
        Origin of the transient: 'TNS', 'ZTF', 'Rubin', or 'other'
    object_name : str
        Standardized transient name with prefixes removed and separators cleaned
    '''

    # Clean input
    name = str(object_name_in).strip()

    # Define patterns for identification
    tns_prefixes = ('AT', 'SN', 'TDE')
    ztf_prefix = 'ZTF'
    year_prefixes = tuple(('19', '20'))

    # Pattern matching logic for source identification
    if name.startswith(tns_prefixes):
        transient_source = 'TNS'
    elif name.startswith(ztf_prefix):
        transient_source = 'ZTF'
    # If it looks like a ZTF object ID without the leading "ZTF", assume ZTF
    elif len(name) >= 9 and not name[:4].isdigit():
        transient_source = 'ZTF'
    # If the name is 18 characters long and they're all digits, assume Rubin
    elif len(name) == 18 and name.isdigit():
        transient_source = 'Rubin'
    # If name starts with 19 or 20, assume TNS
    elif name.startswith(year_prefixes):
        transient_source = 'TNS'
    else:
        transient_source = 'other'

    # Standardize the name based on the transient_source
    if transient_source == 'TNS':
        # Remove separators and TNS prefixes
        object_name = name
        for prefix in ('SN2', 'AT2', 'SN1', 'AT1'):
            if object_name.startswith(prefix):
                object_name = prefix[-1] + object_name[len(prefix):]
        object_name = object_name.replace(' ', '').replace('_', '').replace('-', '')
    elif transient_source == 'ZTF':
        # Keep the ZTF prefix if present, and remove separators
        object_name = name.replace(' ', '').replace('_', '').replace('-', '')
    elif transient_source == 'Rubin':
        # Rubin names are already standardized as 18-digit numbers, so we can keep them as is
        object_name = name
    else:
        object_name = name

    return transient_source, object_name


def get_ztf_name(ra_deg, dec_deg, acceptance_radius=3):
    """
    Query the Alerce database to find the ZTF name of an object at given coordinates.

    Parameters
    -----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    acceptance_radius : float, optional
        Search radius in arcseconds (default: 3)

    Returns
    --------
    ztf_name : str or None
        The ZTF object name if found, None if no object is found at the given coordinates.
    """
    try:
        # Initialize Alerce client
        client = Alerce()

        # Query for objects at the given coordinates
        objects = client.query_objects(ra=ra_deg, dec=dec_deg, radius=acceptance_radius,
                                       survey='ztf')

        # Return the ZTF name if an object was found, None otherwise
        if len(objects) > 0:
            # Calculate the distance to each object
            # Objects is a pandas DataFrame with keys meanra and meandec
            ztf_ra = objects['meanra'].values
            ztf_dec = objects['meandec'].values
            separations = calc_separations(ztf_ra, ztf_dec, ra_deg, dec_deg)
            best_index = int(np.argmin(separations))
            ztf_name = str(objects['oid'].iloc[best_index])
        else:
            ztf_name = None

        return ztf_name

    except Exception as e:
        print(f"Error querying Alerce: {str(e)}")
        return None


def get_ztf_coords(ztf_name):
    """
    Query the Alerce database to get the coordinates of a ZTF object.

    Parameters
    -----------
    ztf_name : str
        The ZTF object name to query

    Returns
    --------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    """

    try:
        # Initialize Alerce client
        client = Alerce()

        # Query for the object by name
        objects = client.query_object(ztf_name, survey='ztf')

        # Extract RA and DEC from the first object found
        ra_deg = objects['meanra']
        dec_deg = objects['meandec']

        return ra_deg, dec_deg

    except Exception as e:
        print(f"Error querying Alerce: {str(e)}")
        return None, None


def get_ztf_lightcurve(object_name, ztf_name=None, save_ztf=True, ztf_dir='ztf', download_ztf=True):
    """
    Query the Alerce database to get the light curve of a ZTF object.

    Parameters
    ----------
    object_name : str
        The name of the object to save the output file
    ztf_name : str
        The ZTF object name to query
    save_ztf : bool, optional
        Whether to save the light curve data to a file (default: True)
    ztf_dir : str, optional
        Directory to save the light curve data (default: 'ztf')
    download_ztf : bool, optional
        Whether to re-download the ZTF data (default: True)
        If False, it will read the data from a local file.

    Returns
    --------
    ztf_data : astropy.table.Table
        The light curve data in an Astropy table format
    """

    # Empty default table
    ztf_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'],
                           dtype=['float64','float64','float64','str','str','str','str','float64','float64'])

    # Setup name and output file
    output_ztf_file = os.path.join(ztf_dir, f'{object_name}.txt')
    # Read existing data if it exists and download_ztf is False
    if os.path.exists(output_ztf_file) and not download_ztf:
        print('\nReading existing ZTF data ...')
        ztf_data = table.Table.read(output_ztf_file, format='ascii')
        if 'comments' in ztf_data.meta:
            header = {}
            for item in ztf_data.meta['comments']:
                if '=' in item:
                    key, value = item.split('=', 1)
                    header[key.strip()] = value.strip()

            ztf_name = header.get('ztf_name', ztf_name)

        return ztf_data, ztf_name
    elif not download_ztf:
        return ztf_data, ztf_name

    try:
        # Initialize Alerce client
        client = Alerce()

        # Query detections for the object
        if ztf_name:
            print(f"Querying Alerce for light curve of {object_name} = {ztf_name}...")
            lightcurve = client.query_lightcurve(ztf_name, format='pandas')
        else:
            print(f"No ZTF name found for {object_name}.")
            return ztf_data, ztf_name

    except Exception as e:
        print(f"Error querying light curve: {str(e)}")
        return ztf_data, ztf_name

    # If we have detections, convert to astropy table and get coordinates
    if ('detections' in lightcurve) and (not lightcurve.empty):
        # Convert to Astropy table
        det = table.Table(lightcurve['detections'][0])['mjd', 'magpsf', 'sigmapsf', 'fid', 'ra', 'dec']
        if 'non_detections' in lightcurve:
            if len(lightcurve['non_detections'][0]) > 0:
                non_det = table.Table(lightcurve['non_detections'][0])['mjd', 'diffmaglim', 'fid']
            else:
                non_det = table.Table(names=['MJD', 'Raw', 'Filter'], dtype=['str'] * 3)
        else:
            non_det = table.Table(names=['MJD', 'Raw', 'Filter'], dtype=['str'] * 3)

        # Convert to a uniform format
        detections = table.Table(det, names=['MJD', 'Raw', 'MagErr', 'Filter', 'RA', 'DEC'], dtype=['str'] * 6)
        non_detections = table.Table(non_det, names=['MJD', 'Raw', 'Filter'], dtype=['str'] * 3)

        # Rename filters to something useful
        detections['Filter'][detections['Filter'] == '1'] = 'g'
        detections['Filter'][detections['Filter'] == '2'] = 'r'
        detections['Filter'][detections['Filter'] == '3'] = 'i'
        non_detections['Filter'][non_detections['Filter'] == '1'] = 'g'
        non_detections['Filter'][non_detections['Filter'] == '2'] = 'r'
        non_detections['Filter'][non_detections['Filter'] == '3'] = 'i'

        # Set the error to -1.0 for non-detections
        ZTF_UL_det = table.Column(['False'] * len(detections), name='UL')
        ZTF_UL_nondet = table.Column(['True'] * len(non_detections), name='UL')
        ZTF_err_nondet = table.Column(['-1.0'] * len(non_detections), name='MagErr')

        # Set the RA and DEC columns for non-detections to Nan
        non_detections['RA'] = table.Column([np.nan] * len(non_detections), name='RA', dtype='str')
        non_detections['DEC'] = table.Column([np.nan] * len(non_detections), name='DEC', dtype='str')

        # Create final table
        detections.add_column(ZTF_UL_det)
        non_detections.add_column(ZTF_UL_nondet)
        non_detections.add_column(ZTF_err_nondet)

        # Join tables if non-detections exist
        if len(non_detections) > 0:
            ztf_data = table.vstack([detections, non_detections])
        else:
            ztf_data = detections

        # Add telescope and source columns
        ztf_data['Telescope'] = 'ZTF'
        ztf_data['Source'] = 'Alerce'

        # Sort data my MJD
        ztf_data.sort('MJD')

        # Append name to header
        if 'comments' not in ztf_data.meta:
            ztf_data.meta['comments'] = []
        ztf_data.meta['comments'].append(f"ztf_name = {ztf_name}")

    # Save data to file if requested
    if save_ztf:
        # Create directory if it doesn't exist
        os.makedirs(ztf_dir, exist_ok=True)
        # Save the data to a file
        ztf_data.write(output_ztf_file, format='ascii', overwrite=True)

    return ztf_data, ztf_name


def get_rubin_name(ra_deg, dec_deg, acceptance_radius=3):
    """
    Query the Alerce database to find the LSST Rubin name of an
    object at the given coordinates.

    Parameters
    -----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    acceptance_radius : float, optional
        Search radius in arcseconds (default: 3)
    
        
    Returns
    --------
    rubin_name : str or None
        The LSST Rubin object name if found, None if no object is found at the given coordinates.
    """

    try:
        # Initialize Alerce client
        client = Alerce()

        # Query for objects at the given coordinates
        objects = client.query_objects(ra=ra_deg, dec=dec_deg, radius=acceptance_radius,
                                       survey='lsst')

        # Return the LSST Rubin name if an object was found, None otherwise
        if len(objects) > 0:
            # Calculate the distance to each object
            rubin_ra = objects['meanra'].values
            rubin_dec = objects['meandec'].values
            separations = calc_separations(rubin_ra, rubin_dec, ra_deg, dec_deg)
            best_index = int(np.argmin(separations))
            rubin_name = str(objects['oid'].iloc[best_index])
        else:
            rubin_name = None

        return rubin_name
    except Exception as e:
        print(f"Error querying Alerce: {str(e)}")
        return None


def get_rubin_coords(rubin_name):
    """
    Query the Alerce database to get the coordinates of a LSST Rubin object.

    Parameters
    -----------
    rubin_name : str
        The LSST Rubin object name to query

    Returns
    --------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    """

    try:
        # Initialize Alerce client
        client = Alerce()

        # Query for the object by name
        objects = client.query_object(rubin_name, survey='lsst')

        # Extract RA and DEC from the first object found
        ra_deg = objects['meanra']
        dec_deg = objects['meandec']

        return ra_deg, dec_deg

    except Exception as e:
        print(f"Error querying Alerce: {str(e)}")
        return None, None


def get_rubin_lightcurve(object_name, rubin_name=None, save_rubin=True, rubin_dir="rubin",
                         download_rubin=True, nsigma_ul=3.0):
    """
    Query ALeRCE to get the LSST light curve of an object. The output will
    be an Astropy Table with the following columns:
        MJD, Raw, MagErr, Telescope, Filter, Source, UL, RA, DEC

    Detections are converted from flux to AB mags using:
        mag_AB = -2.5 * log10(flux) + 31.4

    Upper limits are estimated as:
        lim_mag = -2.5 * log10(nsigma * error) + 31.4

    WARNING: Upper limits are not included right now, they were incorrectly being
             read from the detections table, will need to read them from forced_photometry
             table later.

    Parameters
    ----------
    object_name : str
        Name of the object to use for the saved output file.
    rubin_name : str, optional
        Rubin/LSST object identifier to query, required if 
        there isn't data already in rubin_dir
    save_rubin : bool, optional
        Whether to save the light curve data to a file.
    rubin_dir : str, optional
        Directory to save the light curve data.
    download_rubin : bool, optional
        Whether to re-download the Rubin data.
        If False, read from a local file if available.
    nsigma_ul : float, optional, default 3.0
        Sigma threshold used to estimate upper limits from psfFluxErr.

    Returns
    -------
    rubin_data : astropy.table.Table
        Light curve data

    rubin_name : str
        Rubin/LSST object name
    """

    # Output table format
    output_names = ["MJD","Raw","MagErr","Telescope","Filter","Source","UL","RA","DEC"]
    output_dtype = ["float64","float64","float64","str","str","str","str","float64","float64"]
    rubin_data = table.Table(names=output_names, dtype=output_dtype)

    # Read output file if it exists
    output_rubin_file = os.path.join(rubin_dir, f"{object_name}.txt")
    if os.path.exists(output_rubin_file) and not download_rubin:
        print("\nReading existing Rubin data ...")
        rubin_data = table.Table.read(output_rubin_file, format="ascii")

        if "comments" in rubin_data.meta:
            header = {}
            for item in rubin_data.meta["comments"]:
                if "=" in item:
                    key, value = item.split("=", 1)
                    header[key.strip()] = value.strip()

            if "rubin_name" in header:
                rubin_name = header["rubin_name"]

        return rubin_data, rubin_name

    # If no file exists and user does not want data, well don't give it to them
    elif not download_rubin:
        return rubin_data, rubin_name

    # If no rubin_name was given, the rest of the script will not work
    if rubin_name is None:
        print(f"No Rubin name found for {object_name}.")
        return rubin_data, rubin_name

    # Query the lsst version of Alerce
    try:
        client = Alerce()
        print(f"Querying ALeRCE for Rubin light curve of {object_name} = {rubin_name}...")
        input_df = client.query_lightcurve(rubin_name, format="pandas", survey="lsst")

    except Exception as e:
        print(f"Error querying light curve: {str(e)}")
        return rubin_data, rubin_name

    if input_df is None:
        return rubin_data, rubin_name

    # The latest version of Alerce 2.3.0 should return a pandas DataFrame with nested columns
    if hasattr(input_df, "columns") and "detections" in input_df.columns:
        lightcurve = input_df["detections"]
    else:
        # Fallback in case ALeRCE returns a DataFrame-like object directly.
        lightcurve = input_df

    if lightcurve is None or len(lightcurve) == 0:
        return rubin_data, rubin_name

    # Convert the pandas dataframe to an Astropy Table
    try:
        if hasattr(lightcurve, "iloc") and len(lightcurve) == 1:
            first_cell = lightcurve.iloc[0]
            if isinstance(first_cell, (list, tuple, dict)):
                det_tab = table.Table(first_cell)
            else:
                det_tab = table.Table.from_pandas(lightcurve)
        elif hasattr(lightcurve, "columns"):
            det_tab = table.Table.from_pandas(lightcurve)
        else:
            det_tab = table.Table(lightcurve)
    except Exception:
        det_tab = table.Table(lightcurve)

    if len(det_tab) == 0:
        return rubin_data, rubin_name

    # Get filter names from table
    if "band_name" in det_tab.colnames:
        filter_values = np.array(det_tab["band_name"]).astype(str)
    else:
        filter_values = np.array(["unknown"] * len(det_tab)).astype(str)

    # And coordinates
    if "ra" in det_tab.colnames:
        ra_values = np.array(det_tab["ra"], dtype=float)
    else:
        ra_values = np.full(len(det_tab), np.nan, dtype=float)

    if "dec" in det_tab.colnames:
        dec_values = np.array(det_tab["dec"], dtype=float)
    else:
        dec_values = np.full(len(det_tab), np.nan, dtype=float)

    # etc...
    mjd = np.array(det_tab["mjd"], dtype=float)
    flux = np.array(det_tab["psfFlux"], dtype=float)
    flux_err = np.array(det_tab["psfFluxErr"], dtype=float)

    # Only use detections that are positive with positive errors
    finite_flux = np.isfinite(flux)
    finite_flux_err = np.isfinite(flux_err)
    positive_flux = flux > 0.0
    positive_flux_err = flux_err > 0.0

    # Use the LSST flux flags to reject bad data
    good_flags = np.ones(len(det_tab), dtype=bool)

    for flag_col in ["psfFlux_flag", # Failure to derive linear least-squares fit of psf model
                     "psfFlux_flag_edge", # Object was too close to the edge of the image to use the full PSF model
                     "psfFlux_flag_noGoodPixels", # Not enough non-rejected pixels in data to attempt the fit
                     ]:
        if flag_col in det_tab.colnames:
            flag_values = np.array(det_tab[flag_col]).astype(bool)
            good_flags &= ~flag_values

    # Return only valid measurements
    valid_measurement = finite_flux_err & positive_flux_err & good_flags

    # Return only positve, unflagged values
    is_detection = (valid_measurement & finite_flux & positive_flux)

    # Everything else with a usable flux error becomes an estimated upper limit
    is_upper_limit = np.zeros(len(det_tab), dtype=bool)

    # Convert detections from flux to magnitude
    det_mag = -2.5 * np.log10(flux[is_detection]) + 31.4
    det_mag_err = (2.5 / np.log(10.0)) * flux_err[is_detection] / flux[is_detection]

    # Create final detections table
    if np.sum(is_detection) > 0:
        detections = table.Table(data=[
                mjd[is_detection],
                det_mag,
                det_mag_err,
                np.array(["Rubin"] * np.sum(is_detection)),
                filter_values[is_detection],
                np.array(["Alerce"] * np.sum(is_detection)),
                np.array(["False"] * np.sum(is_detection)),
                ra_values[is_detection],
                dec_values[is_detection],
                ], names=output_names, dtype=output_dtype)
    else:
        detections = table.Table(names=output_names, dtype=output_dtype)

    # Calculate magnitude of upper limits from flux error
    if np.sum(is_upper_limit) > 0:
        lim_flux = nsigma_ul * flux_err[is_upper_limit]
        lim_mag = -2.5 * np.log10(lim_flux) + 31.4

        upper_limits = table.Table(data=[
                mjd[is_upper_limit],
                lim_mag,
                np.full(np.sum(is_upper_limit), -1.0, dtype=float),
                np.array(["Rubin"] * np.sum(is_upper_limit)),
                filter_values[is_upper_limit],
                np.array(["Alerce"] * np.sum(is_upper_limit)),
                np.array(["True"] * np.sum(is_upper_limit)),
                ra_values[is_upper_limit],
                dec_values[is_upper_limit],
            ], names=output_names, dtype=output_dtype)
    else:
        upper_limits = table.Table(names=output_names, dtype=output_dtype)

    # Stack tables
    if len(detections) > 0 and len(upper_limits) > 0:
        rubin_data = table.vstack([detections, upper_limits])
    elif len(detections) > 0:
        rubin_data = detections
    elif len(upper_limits) > 0:
        rubin_data = upper_limits

    if len(rubin_data) > 0:
        rubin_data.sort("MJD")
        if "comments" not in rubin_data.meta:
            rubin_data.meta["comments"] = []
        rubin_data.meta["comments"].append(f"rubin_name = {rubin_name}")

    # Save output table if requested
    if save_rubin:
        os.makedirs(rubin_dir, exist_ok=True)
        rubin_data.write(output_rubin_file, format="ascii", overwrite=True)

    return rubin_data, rubin_name


def get_tns_credentials():
    """
    Retrieve TNS credentials from environment variables if available,
    otherwise fall back to reading the tns_key.txt file. The tns_key.txt
    file must be formatted with three lines:

    1. API key
    2. TNS ID
    3. Username
    """
    # Check for environment variables
    api_key = os.environ.get("TNS_API_KEY")
    tns_id = os.environ.get("TNS_ID")
    username = os.environ.get("TNS_USERNAME")

    if api_key and tns_id and username:
        return api_key, tns_id, username

    # Fall back to the key file in the user's home directory.
    if os.path.exists(pathlib.Path.home() / 'tns_key.txt'):
        key_path = pathlib.Path.home() / 'tns_key.txt'
    else:
        key_path = os.path.join(fleet_data, 'tns_key.txt')
    try:
        with open(key_path, 'r') as key_file:
            lines = [line.strip() for line in key_file if line.strip()]
        if len(lines) < 3:
            raise ValueError("TNS key file is incomplete. It must contain API key, TNS ID, and username.")
        return lines[0], lines[1], lines[2]
    except Exception as e:
        raise Exception("Error retrieving TNS credentials: " + str(e))


def download_tns_public_objects(filename="tns_public_objects.csv"):
    """
    Downloads the CSV file containing all public TNS objects from the TNS API,
    uncompresses the downloaded ZIP file, and saves the CSV file with the specified filename.

    Parameters:
    -----------
    filename : str
        The desired name for the CSV file (without the .csv extension).
    """

    # Check if the filename ends with .csv and remove it
    if filename.endswith('.csv'):
        filename = filename[:-4]

    # Retrieve TNS credentials
    api_key, tns_id, username = get_tns_credentials()

    # Build the URL for the TNS public objects download endpoint
    url = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"

    # Prepare the headers with the user-agent using the credentials
    headers = {
        'User-Agent': f'tns_marker{{"tns_id":{tns_id}, "type":"bot", "name":"{username}"}}'
    }

    # Prepare the data payload with the API key
    payload = {
        'api_key': api_key
    }

    try:
        # Perform the POST request to download the file
        print(f"Downloading public TNS objects to '{filename}.csv'...")
        response = requests.post(url, data=payload, headers=headers)

        # Check if the request was successful
        response.raise_for_status()

        # Write the content of the response to a ZIP file
        zip_filename = 'tns_public_objects.csv.zip'
        with open(zip_filename, 'wb') as f:
            f.write(response.content)

        print(f"Download completed successfully and saved as '{zip_filename}'.")

        # Uncompress the ZIP file and save the CSV file with the specified filename
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Extract the first file in the ZIP archive (assuming there's only one file)
            csv_filename = zip_ref.namelist()[0]
            zip_ref.extract(csv_filename)

            # Rename the extracted CSV file to the specified filename
            os.rename(csv_filename, f"{filename}.csv")

        print(f"CSV file saved as '{filename}.csv'.")

        # Delete the ZIP file after extraction
        os.remove(zip_filename)
        print(f"Deleted the ZIP file '{zip_filename}'.")

    except requests.RequestException as req_err:
        print("HTTP Request error while downloading TNS public objects:", req_err)
    except zipfile.BadZipFile as zip_err:
        print("Error while extracting the ZIP file:", zip_err)
    except Exception as e:
        print("An unexpected error occurred:", e)


def get_tns_coords_class(tns_name):
    """
    Retrieve the Right Ascension (RA) and Declination (DEC) in degrees for a transient
    from the Transient Name Server (TNS) based on its IAU name, along with the transient type.

    This function requires a TNS API key file located in the user's home directory named 'tns_key.txt'.
    The file should contain three lines:
      1. API key
      2. TNS ID
      3. Username

    Parameters
    -----------
    tns_name : str
        The name of the transient (e.g. "2018hyz"). If the name starts with "AT" or "AT_",
        those prefixes will be removed.

    Returns
    --------
    ra_deg : float
        Right Ascension in degrees, or None if not found.
    dec_deg : float
        Declination in degrees, or None if not found.
    object_class : str
        The type of the transient, or None if not found.
    redshift : float
        The redshift of the transient, or None if not found.
    """

    # Empty variables
    ra_deg = None
    dec_deg = None
    object_class = None
    redshift = None

    # Normalize tns_name: remove leading "AT", "AT_", "AT-", "AT ", "SN", or "TDE" if present.
    tns_name = tns_name.strip()
    tns_name = tns_name.replace(' ', '').replace('_', '').replace('-', '')
    for prefix in ("AT", "SN", "TDE"):
        if tns_name.startswith(prefix):
            tns_name = tns_name[len(prefix):]
            break

    # Locate and read the TNS key file
    try:
        api_key, tns_id, username = get_tns_credentials()
    except Exception as e:
        print("Error retrieving TNS credentials:", e)
        return ra_deg, dec_deg, object_class, redshift

    # Build the URL and the query payload
    base_url = "https://www.wis-tns.org/api/get"
    object_endpoint = f"{base_url}/object"
    query_data = OrderedDict([
        ("objname", tns_name),
        ("photometry", "0"),
        ("tns_id", tns_id),
        ("type", "user"),
        ("name", username)
    ])
    payload = {
        'api_key': (None, api_key),
        'data': (None, json.dumps(query_data))
    }
    headers = {
        'User-Agent': f'tns_marker{{"tns_id":{tns_id}, "type":"bot", "name":"{username}"}}'
    }

    try:
        print(f"Querying TNS for coordinates of '{tns_name}'...")
        response = requests.post(object_endpoint, files=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()

        # Check if the response contains valid data
        if 'data' not in response_json:
            error_message = response_json.get('id_message', 'Unknown error from TNS')
            print("TNS error:", error_message)
            return ra_deg, dec_deg, object_class, redshift

        data = response_json['data']
        ra_deg = data.get('radeg')
        dec_deg = data.get('decdeg')
        object_class = data.get('object_type', {}).get('name')
        if object_class in empties:
            object_class = None
        redshift = data.get('redshift', None)
        if redshift in empties:
            redshift = None

        if ra_deg is None or dec_deg is None:
            print(f"Coordinates not found in TNS for '{tns_name}'.")
            return ra_deg, dec_deg, object_class, redshift

    except requests.RequestException as req_err:
        print("HTTP Request error while querying TNS:", req_err)
    except Exception as e:
        print("An unexpected error occurred while querying TNS:", e)

    return ra_deg, dec_deg, object_class, redshift


def get_tns_name(ra_deg, dec_deg, acceptance_radius=3):
    """
    Query the TNS for the name of a transient at given coordinates.

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    acceptance_radius : float
        Search radius in arcseconds
        (default: 3)

    Returns
    -------
    tns_name : str
        The name of the transient if found, None otherwise.
    """

    # Locate and read the TNS key file
    try:
        api_key, tns_id, username = get_tns_credentials()
    except Exception as e:
        print("Error retrieving TNS credentials:", e)
        return None

    # Build the URL and the query payload
    base_url = "https://www.wis-tns.org/api/get"
    search_url = base_url + "/search"
    tns_marker = 'tns_marker{"tns_id": "' + str(tns_id) + '", "type": "bot", "name": "' + username + '"}'
    headers = {'User-Agent': tns_marker}
    json_file = OrderedDict([("ra", str(ra_deg)), ("dec", str(dec_deg)), ("radius", str(acceptance_radius)), ("units", "arcsec")])
    search_data = {'api_key': api_key, 'data': json.dumps(json_file)}
    try:
        response = requests.post(search_url, headers=headers, data=search_data)
        data = response.json()['data']

        if len(data) > 0:
            # Extract the name of the first object
            tns_name = data[0]['objname']
            return tns_name
        else:
            return None

    except requests.RequestException as req_err:
        print("HTTP Request error while querying TNS:", req_err)
    except Exception as e:
        print("An unexpected error occurred while querying TNS:", e)

    return None


def get_osc_lightcurve(object_name, ra_deg=None, dec_deg=None, save_osc=True, osc_dir='osc', download_osc=True):
    """
    Query the OSC database to get the light curve of a transient object.

    Parameters
    ----------
    object_name : str
        The name of the object to query
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    save_osc : bool, optional
        Whether to save the light curve data to a file (default: True)
    osc_dir : str, optional
        Directory to save the light curve data (default: 'osc')
    download_osc : bool, optional
        Whether to re-download the OSC data (default: True)
        If False, it will read the data from a local file.

    Returns
    -------
    osc_data : astropy.table.Table
        The light curve data in an Astropy table format
    """

    # Empty default table
    osc_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'],
                           dtype=['float64','float64','float64','str','str','str','str','float64','float64'])

    # Setup name and output file
    output_osc_file = os.path.join(osc_dir, f'{object_name}.txt')
    # Read existing data if it exists and download_osc is False
    if not download_osc:
        if os.path.exists(output_osc_file):
            print('\nReading existing OSC data ...')
            osc_data = table.Table.read(output_osc_file, format='ascii')
        return osc_data

    # Create osc link for query
    osc_link = f'https://astrocats.space/api/{object_name}/photometry/time+magnitude+e_magnitude+band+upperlimit+telescope'

    # Request Data
    print('Querying OSC ...')
    osc_request = requests.get(osc_link).json()

    # Get the data and remove any lines with no time
    raw_data = np.array(osc_request[object_name]['photometry'])

    # Make empties nans or '--'
    OSC_MJD = np.array([np.nan if i in empties else float(i) for i in raw_data.T[0]])
    OSC_Mag = np.array([np.nan if i in empties else float(i) for i in raw_data.T[1]])
    OSC_MagErr = np.array([np.nan if i in empties else float(i) for i in raw_data.T[2]])
    OSC_filter = np.array(['--' if i in empties else i for i in raw_data.T[3]])
    OSC_UL = np.array(['--' if i in empties else i for i in raw_data.T[4]])
    OSC_telescope = np.array(['--' if i in empties else i for i in raw_data.T[5]])

    # Combine into an Astropy Table
    osc_data = table.Table(
        [OSC_MJD, OSC_Mag, OSC_MagErr, OSC_filter, OSC_UL, OSC_telescope],
        names=['MJD', 'Raw', 'MagErr', 'Filter', 'UL', 'Telescope']
    )

    # Add RA, DEC, and Source columns
    osc_data['RA'] = table.Column([ra_deg] * len(osc_data), name='RA', dtype='float64')
    osc_data['DEC'] = table.Column([dec_deg] * len(osc_data), name='DEC', dtype='float64')
    osc_data['Source'] = table.Column(['OSC'] * len(osc_data), name='Source', dtype='str')

    # Sort data by MJD
    osc_data.sort('MJD')

    # Save data to file if requested
    if save_osc:
        # Create directory if it doesn't exist
        os.makedirs(osc_dir, exist_ok=True)
        # Save the data to a file
        osc_data.write(output_osc_file, format='ascii', overwrite=True)

    return osc_data


def get_local_lightcurve(object_name, local_dir='photometry', read_local=True):
    """
    Read light curve data from a local file.

    Parameters
    ----------
    object_name : str
        The name of the object to read
    local_dir : str
        Directory to read the light curve data from (default: 'photometry')
    read_local : bool
        Whether to read local data (default: True)
        If False, it will not read the data.

    Returns
    -------
    my_data : astropy.table.Table
        The light curve data in an Astropy table format
    """

    local_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'],
                             dtype=['float64','float64','float64','str','str','str','str','float64','float64'])

    # Setup name and output file
    local_file = os.path.join(local_dir, f'{object_name}.txt')

    if read_local and os.path.exists(local_file):
        print('Reading local data ...')
        local_data = table.Table.read(local_file, format='ascii')

        # If RA and DEC are not in the data, set them to NaN
        if 'RA' not in local_data.colnames:
            local_data['RA'] = table.Column([np.nan] * len(local_data), name='RA', dtype='float64')
        if 'DEC' not in local_data.colnames:
            local_data['DEC'] = table.Column([np.nan] * len(local_data), name='DEC', dtype='float64')

        # If Source is not in the data, set it to 'Local'
        if 'Source' not in local_data.colnames:
            local_data['Source'] = table.Column(['Local'] * len(local_data), name='Source', dtype='str')

    return local_data


def get_transient_info(object_name_in=None, ra_in=None, dec_in=None, object_class_in=None, redshift_in=None,
                       acceptance_radius=3, save_ztf=True, save_rubin=True, download_ztf=True, download_rubin=True,
                       download_osc=False, read_local=True, query_tns=True, ztf_dir='ztf', rubin_dir='rubin',
                       lc_dir='lightcurves', osc_dir='osc', local_dir='photometry'):
    '''
    Get the coordinates and name for a transient. Either the coordinates
    and/or the name must be specified. The function will search for missing
    cross-matches in TNS, ZTF, and Rubin/LSST. Photometry from ZTF, Rubin/LSST,
    OSC, and local files will also be gathered if available.

    Parameters
    ----------
    object_name_in : str
        Name of the object.
    ra_in : float
        Right ascension of the object in degrees.
    dec_in : float
        Declination of the object in degrees.
    object_class_in : str
        Transient type, to overwrite any existing classes.
    redshift_in : float
        Redshift of the object, to overwrite any existing redshifts.
    acceptance_radius : float
        Search radius in arcseconds.
    save_ztf : bool
        Save ZTF data to a file? If False, it will not save the data.
    save_rubin : bool
        Save Rubin data to a file? If False, it will not save the data.
    download_ztf : bool
        Re-download ZTF data from Alerce? If False, it will
        read the data from a local file.
    download_rubin : bool
        Re-download Rubin data from Alerce? If False, it will
        read the data from a local file.
    download_osc : bool
        Re-download OSC data? If False, it will read the
        data from a local file.
    read_local : bool
        Read local data from the "photometry" directory?
    query_tns : bool
        Query the TNS for missing info like object class
        and redshift?
    ztf_dir : str
        Directory to save the ZTF data to. Default is 'ztf'.
    rubin_dir : str
        Directory to save the Rubin data to. Default is 'rubin'.
    lc_dir : str
        Directory to save the light curve data to. Default is 'lightcurves'.
    osc_dir : str, default 'osc'
        Directory where OSC data is stored
    local_dir : str, default 'photometry'
        Directory where local photometry data is stored

    Returns
    -------
    ra_deg : float
        Right ascension of the object in degrees.
    dec_deg : float
        Declination of the object in degrees.
    transient_source : str
        Source origin of the transient (TNS, ZTF, Rubin, Other).
    object_name : str
        Standard name of the transient.
    ztf_data : astropy.table.Table
        Astropy Table with or without data.
    rubin_data : astropy.table.Table
        Astropy Table with or without data.
    osc_data : astropy.table.Table
        Astropy Table with or without data.
    local_data : astropy.table.Table
        Astropy Table with or without data.
    ztf_name : str
        Name of transient in ZTF, if it exists.
    rubin_name : str
        Name of transient in Rubin/LSST, if it exists.
    tns_name : str
        Name of transient in TNS, if it exists.
    object_class : str
        Class of the object.
    redshift : float
        Redshift of the object.
    '''

    # Empty variables
    ra_deg = None
    dec_deg = None
    transient_source = None
    object_name = None
    ztf_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'])
    rubin_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'])
    osc_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'])
    local_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'])
    ztf_name = None
    rubin_name = None
    tns_name = None
    object_class = None
    redshift = None

    # If downloading is disabled, only flip it back on when there is no local
    # combined/survey/local file to read. Guard against coordinate-only calls,
    # where object_name_in can be None.
    clean_object_name_in = None
    if object_name_in is not None:
        clean_object_name_in = object_name_in.strip().replace(' ', '').replace('_', '').replace('-', '')
        if clean_object_name_in.startswith('AT'):
            clean_object_name_in = clean_object_name_in[2:]
        if clean_object_name_in.startswith('SN'):
            clean_object_name_in = clean_object_name_in[2:]

    if not download_ztf and clean_object_name_in is not None:
        ztf_local_files = [
            os.path.join(lc_dir, f'{clean_object_name_in}.txt'),
            os.path.join(osc_dir, f'{clean_object_name_in}.txt'),
            os.path.join(local_dir, f'{clean_object_name_in}.txt'),
            os.path.join(ztf_dir, f'{clean_object_name_in}.txt'),
        ]
        if not any(os.path.exists(filename) for filename in ztf_local_files):
            print(f"Warning: {clean_object_name_in} not found locally. Setting download_ztf to True.")
            download_ztf = True

    if not download_rubin and clean_object_name_in is not None:
        rubin_local_files = [
            os.path.join(lc_dir, f'{clean_object_name_in}.txt'),
            os.path.join(osc_dir, f'{clean_object_name_in}.txt'),
            os.path.join(local_dir, f'{clean_object_name_in}.txt'),
            os.path.join(rubin_dir, f'{clean_object_name_in}.txt'),
        ]
        if not any(os.path.exists(filename) for filename in rubin_local_files):
            print(f"Warning: {clean_object_name_in} not found locally. Setting download_rubin to True.")
            download_rubin = True

    # If no coordinates or name were specified, raise an error
    if ra_in is None and dec_in is None and object_name_in is None:
        print("Either coordinates or a name must be specified.")
        return ra_deg, dec_deg, transient_source, object_name, ztf_data, rubin_data, osc_data, local_data, ztf_name, rubin_name, tns_name, object_class, redshift

    # If coordinates were specified, convert them to degrees
    if ra_in is not None and dec_in is not None:
        try:
            ra_deg, dec_deg = convert_coords(ra_in, dec_in)
        except ValueError as e:
            print(f"Invalid coordinates: {e}")
            return ra_deg, dec_deg, transient_source, object_name, ztf_data, rubin_data, osc_data, local_data, ztf_name, rubin_name, tns_name, object_class, redshift

        # Get names from enabled services when no object name was provided
        if object_name_in is None:
            if query_tns:
                tns_name = get_tns_name(ra_deg, dec_deg, acceptance_radius)
                if tns_name is not None:
                    _, _, object_class, redshift = get_tns_coords_class(tns_name)

            if download_ztf:
                ztf_name = get_ztf_name(ra_deg, dec_deg, acceptance_radius)

            if download_rubin:
                rubin_name = get_rubin_name(ra_deg, dec_deg, acceptance_radius)

            # Explicit user inputs should override queried class/redshift values
            if object_class_in is not None:
                object_class = object_class_in
            if redshift_in is not None:
                redshift = redshift_in

            # Determine source and standardize the name
            if tns_name is not None:
                transient_source = 'TNS'
                object_name = tns_name
            elif ztf_name is not None:
                transient_source = 'ZTF'
                object_name = ztf_name
            elif rubin_name is not None:
                transient_source = 'Rubin'
                object_name = rubin_name
            else:
                transient_source = 'other'
                object_name = f"coord_{ra_deg:.6f}_{dec_deg:.6f}".replace('-', 'm')

    # If a name was specified, clean it and determine the source
    if object_name_in is not None:
        # Determine the source and standardize the name
        transient_source, object_name = transient_origin(object_name_in)

        # If transient_source is ZTF, query Alerce for light curve data
        if transient_source == 'ZTF':
            ztf_name = object_name
            if ra_in is None or dec_in is None:
                ra_deg, dec_deg = get_ztf_coords(ztf_name)
            else:
                ra_deg, dec_deg = convert_coords(ra_in, dec_in)

        # If transient_source is Rubin, query Alerce for light curve data
        elif transient_source == 'Rubin':
            rubin_name = object_name
            if ra_in is None or dec_in is None:
                ra_deg, dec_deg = get_rubin_coords(rubin_name)
            else:
                ra_deg, dec_deg = convert_coords(ra_in, dec_in)

        # If transient_source is TNS, query TNS for coordinates and class
        elif transient_source == 'TNS':
            tns_name = object_name
            # Query TNS for coordinates and class
            if ra_in is None or dec_in is None:
                ra_deg, dec_deg, fetched_class, fetched_redshift = get_tns_coords_class(tns_name)
            else:
                ra_deg, dec_deg = convert_coords(ra_in, dec_in)
                if query_tns:
                    _, _, fetched_class, fetched_redshift = get_tns_coords_class(tns_name)
                else:
                    fetched_class = None
                    fetched_redshift = None

            # Assign class and redshift
            if object_class_in is None:
                object_class = fetched_class
            else:
                object_class = object_class_in
            if redshift_in is None:
                redshift = fetched_redshift
            else:
                redshift = redshift_in

            # Get ZTF name
            if download_ztf:
                ztf_name = get_ztf_name(ra_deg, dec_deg, acceptance_radius)
            
            # Get Rubin name
            if download_rubin:
                rubin_name = get_rubin_name(ra_deg, dec_deg, acceptance_radius)

    # Explicit user inputs should always be preserved, including for ZTF,
    # Rubin, and local/other objects.
    if object_class_in is not None:
        object_class = object_class_in
    if redshift_in is not None:
        redshift = redshift_in

    # Once coordinates are known, cross-match the other enabled survey too.
    # This lets ZTF-named objects pick up Rubin light curves and Rubin-named
    # objects pick up ZTF light curves when both are available.
    if ra_deg is not None and dec_deg is not None:
        if download_ztf and ztf_name is None:
            ztf_name = get_ztf_name(ra_deg, dec_deg, acceptance_radius)
        if download_rubin and rubin_name is None:
            rubin_name = get_rubin_name(ra_deg, dec_deg, acceptance_radius)

    # Query light curves from ZTF
    ztf_data, ztf_name = get_ztf_lightcurve(object_name, ztf_name, save_ztf=save_ztf, ztf_dir=ztf_dir, download_ztf=download_ztf)

    # Query light curves from OSC
    osc_data = get_osc_lightcurve(object_name, ra_deg, dec_deg, osc_dir=osc_dir, download_osc=download_osc)

    # Query light curves from Rubin/LSST
    rubin_data, rubin_name = get_rubin_lightcurve(object_name, rubin_name, save_rubin=save_rubin,
                                                  rubin_dir=rubin_dir, download_rubin=download_rubin)

    # Query local data
    if read_local:
        local_data = get_local_lightcurve(object_name, local_dir=local_dir)

    return (ra_deg, dec_deg, transient_source, object_name, ztf_data, rubin_data,
            osc_data, local_data, ztf_name, rubin_name, tns_name, object_class, redshift)


def ignore_data(object_name, output_table):
    '''
    Search the ./ignore folder to find photomety that needs to be ignored.
    Take the input astropy table and modify its 'Ignore' column to specify
    which datapoints will be ignored

    Parameters
    ----------
    object_name : str
        Name of the object.
    output_table : astropy.table.Table
        The output table to modify.

    Returns
    -------
    output_table : astropy.table.Table
        The modified output table with the 'Ignore' column updated.
    '''

    # Create folder called ignore if it does not exist
    os.makedirs('ignore', exist_ok=True)

    # Search for files
    ignore_file = os.path.join('ignore', f'{object_name}.txt')

    if os.path.exists(ignore_file):
        print('\nReading ignore file ...')
        # Read the ignore file
        ignore_range = np.genfromtxt(ignore_file)

        # If there's only one line
        if ignore_range.shape == (4,):
            min_MJD, max_MJD, min_mag, max_mag = ignore_range

            data_ignored = ((output_table['MJD'] > min_MJD) & (output_table['MJD'] < max_MJD) &
                            (output_table['Raw'] > min_mag) & (output_table['Raw'] < max_mag))
            output_table['Ignore'][data_ignored] = 'True'

        # If there are multiple lines
        else:
            for i in range(len(ignore_range)):
                min_MJD, max_MJD, min_mag, max_mag = ignore_range[i]

                data_ignored = ((output_table['MJD'] > min_MJD) & (output_table['MJD'] < max_MJD) &
                                (output_table['Raw'] > min_mag) & (output_table['Raw'] < max_mag))
                output_table['Ignore'][data_ignored] = 'True'

    return output_table


def query_dust(ra_deg, dec_deg, dust_map='SFD'):
    """
    Query dust maps to get reddening value. In order to use the 'SF'
    map you need to download the dust maps, which are queried locally
    by doing:

    from dustmaps.config import config
    config['data_dir'] = '/path/to/store/maps/in'

    import dustmaps.sfd
    dustmaps.sfd.fetch()

    The 'SFD' dust map uses a slower online query

    Parameters
    ----------
    ra_deg : float
        Right Ascension of the object in degrees.
    dec_deg : float
        Declination of the object in degrees.
    dust_map : str
        'SF', 'SFD', or 'none', to query Schlafy and Finkbeiner 2011
        or Schlafy, Finkbeiner and Davis 1998 set to 'none' to not
        correct for extinction.

    Returns
    -------
    ebv : float
        The reddening value in magnitudes.
    """

    if dust_map == 'none':
        return 0

    # Query using S&F
    if dust_map == 'SF':
        # Generate URL to query
        dust_url = f'https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={ra_deg}+{dec_deg}+equ+j2000'
        response = requests.get(dust_url)
        # Create xml response Tree
        tree = ElementTree.fromstring(response.content)
        # Extract mean reddening value from S&F
        for child in tree.iter('meanValueSandF'):
            reddeningSandF = child.text.replace('\n', '').replace(' ', '').replace('(mag)', '')
        ebv = float(reddeningSandF)
        return ebv
    # Query using SFD
    elif dust_map == 'SFD':
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        sfd = SFDQuery()
        ebv = sfd(coord)
        return ebv
    else:
        print("'dust_map' must be 'SF', 'SFD', or 'none'")
        return


def process_lightcurve(object_name, ra_deg=None, dec_deg=None, ztf_data=None, rubin_data=None, osc_data=None,
                       local_data=None, save_lc=True, lc_dir='lightcurves', read_existing=False,
                       clean_ignore=True, dust_map='SFD'):
    """
    Gather all the available photometry and merge it into one astropy table.

    Parameters
    ----------
    object_name : str
        Name of the object.
    ra_deg : float
        Right Ascension in degrees.
    dec_deg : float
        Declination in degrees.
    ztf_data : astropy.table.Table
        ZTF light curve data.
    rubin_data : astropy.table.Table
        Rubin light curve data.
    osc_data : astropy.table.Table
        OSC light curve data.
    local_data : astropy.table.Table
        Local light curve data.
    save_lc : bool
        Whether to save the combined light curve data to a file (default: True).
    lc_dir : str
        Directory to save the light curve data (default: 'lightcurves').
    read_existing : bool
        Whether to read existing light curve data from a file (default: False).
    clean_ignore : bool
        Whether to ignore data based on the ignore file (default: True).
    dust_map : str, default: 'SFD'
        Dust map to use for reddening correction ('SF', 'SFD', or 'none').

    Returns
    -------
    input_table : astropy.table.Table
        Combined light curve data from all sources.
    """

    # Read existing data if requested
    output_file = os.path.join(lc_dir, f'{object_name}.txt')
    # Check if the file exists
    if os.path.exists(output_file) and read_existing:
        print('\nReading existing light curve data ...')
        input_table = table.Table.read(output_file, format='ascii')

        # Ignore data if requested
        if clean_ignore:
            input_table = ignore_data(object_name, input_table)

        # Recompute central wavelengths so old cached mixed ZTF+Rubin light
        # curves are not stuck with stale or generic wavelength assignments.
        if len(input_table) > 0 and 'Filter' in input_table.colnames:
            input_table['Cenwave'] = get_table_cenwaves(input_table)

        return input_table
    else:
        print('\nProcessing light curve data ...')

    # Change table type
    if ztf_data is None:
        ztf_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'])
    if rubin_data is None:
        rubin_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'])
    if osc_data is None:
        osc_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'])
    if local_data is None:
        local_data = table.Table(names=['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC'])

    # Needed columns
    colnames = ['MJD', 'Raw', 'MagErr', 'Telescope', 'Filter', 'Source', 'UL', 'RA', 'DEC', 'Ignore']

    # Make sure all three tables have RA and DEC columns, if not add them with ra_deg and dec_deg
    for data in [ztf_data, rubin_data, osc_data, local_data]:
        if 'RA' not in data.colnames:
            data['RA'] = table.Column([ra_deg] * len(data), name='RA', dtype='float64')
        if 'DEC' not in data.colnames:
            data['DEC'] = table.Column([dec_deg] * len(data), name='DEC', dtype='float64')
        # If Ignore is not a column, add it as all False
        if 'Ignore' not in data.colnames:
            data['Ignore'] = table.Column(['False'] * len(data), name='Ignore', dtype='str')

    # Make sure all columns exist in all tables, if not add them as NaN's
    for col in colnames:
        if col not in ztf_data.colnames:
            ztf_data[col] = table.Column([np.nan] * len(ztf_data), name=col, dtype='float64')
        if col not in rubin_data.colnames:
            rubin_data[col] = table.Column([np.nan] * len(rubin_data), name=col, dtype='float64')
        if col not in osc_data.colnames:
            osc_data[col] = table.Column([np.nan] * len(osc_data), name=col, dtype='float64')
        if col not in local_data.colnames:
            local_data[col] = table.Column([np.nan] * len(local_data), name=col, dtype='float64')

    # Copy the types from ztf_data into the other two tables
    if len(ztf_data) > 0:
        rubin_data = table.Table(rubin_data, names=ztf_data.colnames, dtype=ztf_data.dtype)
        osc_data = table.Table(osc_data, names=ztf_data.colnames, dtype=ztf_data.dtype)
        local_data = table.Table(local_data, names=ztf_data.colnames, dtype=ztf_data.dtype)

    # Combine all data into one table
    input_table = table.vstack([ztf_data, rubin_data, osc_data, local_data])

    # If there is no data, return an empty table
    if len(input_table) == 0:
        print("No data found for the specified object.")
        return input_table

    # Assigning types: float, float, float, str, str, str, str, float, float, str
    input_table = table.Table(input_table[colnames], names=colnames,
                              dtype=['float64', 'float64', 'float64', 'str', 'str', 'str', 'str', 'float64', 'float64', 'str'])

    # Make sure all values in UL are either True or False, if they are in this list set it to True
    limit_list = [True, -1.0, 'True', '-1', '-1.0', '-1.', b'True', b'-1', b'-1.0', b'-1.', 'T']
    input_table['UL'] = ['True' if ul in limit_list else 'False' for ul in input_table['UL']]

    # Only keep items where MJD and Mag are real
    input_table = input_table[~np.isnan(input_table['MJD']) & ~np.isnan(input_table['Raw']) & (input_table['Raw'] > 0)]

    # If any MagErr are nan's, set them to 0.1
    missing_magerr = np.isnan(input_table['MagErr'])
    input_table['MagErr'][missing_magerr] = 0.1

    # Sort by MJD
    input_table.sort('MJD')

    # Ignore data if requested
    if clean_ignore:
        input_table = ignore_data(object_name, input_table)

    # Add central wavelengths per row so ZTF-only, Rubin-only, and mixed
    # ZTF+Rubin light curves keep the correct survey-specific filter references.
    input_table['Cenwave'] = get_table_cenwaves(input_table)

    # Correct for Extinction using Gordon 2023
    E_BV = query_dust(ra_deg, dec_deg, dust_map=dust_map)
    R_V = 3.1
    ext = G23(Rv=R_V)
    correction = -2.5 * np.log10(ext.extinguish(np.array(input_table['Cenwave']) * u.AA, Ebv=E_BV))

    # Apply correction
    input_table['Mag'] = input_table['Raw'] - correction

    # Save output
    if save_lc:
        # Create directory if it doesn't exist
        os.makedirs(lc_dir, exist_ok=True)
        # Save the data to a file
        input_table.write(output_file, format='ascii', overwrite=True)

    return input_table
