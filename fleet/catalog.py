from .transient import query_dust, calc_separations
from astroquery.vizier import Vizier
from scipy.special import gamma, gammainc
import warnings
from dust_extinction.parameter_averages import G23
import os
from astroquery.mast import Catalogs
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astroquery.sdss import SDSS
from astroquery.gaia import Gaia
from astroquery.ipac.irsa import Irsa
import time
from astropy import table
import pkg_resources

try:
    fleet_data = os.environ['fleet_data']
except KeyError:
    fleet_data = os.path.join(os.path.dirname(__file__), 'data')

# Central wavelengths for SDSS and 3PI filters
sdss_refs = {'u': 3608.04, 'g': 4671.78, 'r': 6141.12, 'i': 7457.89, 'z': 8922.78}
psst_refs = {'g': 4810.16, 'r': 6155.47, 'i': 7503.03, 'z': 8668.36, 'y': 9613.60}

# Survey limits for star/galaxy separation
survey_limits = {'gPSFMag_3pi': 23.64,
                 'rPSFMag_3pi': 23.27,
                 'iPSFMag_3pi': 22.81,
                 'zPSFMag_3pi': 22.44,
                 'yPSFMag_3pi': 22.86,
                 'psfMag_u_sdss': 23.42,
                 'psfMag_g_sdss': 23.17,
                 'psfMag_r_sdss': 22.59,
                 'psfMag_i_sdss': 22.04,
                 'psfMag_z_sdss': 21.58}

# Default limits for host galaxy mags
host_limit = {'u': 23.42,
              'g': 23.64,
              'r': 23.27,
              'i': 22.81,
              'z': 22.44,
              'y': 22.86}

# Import CFHLST data to classify objects as star/galaxy
classification_catalog_filename = pkg_resources.resource_filename(__name__, 'classification_catalog.dat')
cached_catalog = table.Table.read(classification_catalog_filename, format='ascii', guess=False)

# Import GLADE data to classify objects as star/galaxy
glade_filename = pkg_resources.resource_filename(__name__, 'GLADE_short.txt')
cached_glade = table.Table.read(glade_filename, format='ascii.fast_csv', delimiter=',', guess=False)


def clean_table(catalog, replace_value=np.nan):
    """
    Replace values in an astropy Table that are considered empty with np.nan.

    Parameters
    ----------
    catalog : astropy.table.Table
        The catalog table to process
    replace_value : float
        The value to replace empty values with (default is np.nan)

    Returns
    -------
    astropy.table.Table
        Table with empty values replaced by np.nan
    """

    # List of values considered empty
    empties = [False, '9999', '9999.0', 'False', '', '-999', '-999.0', '--', 'n', '-9999.0', 'nan', b'']

    # Replace values with empty 9999's
    catalog = catalog.filled(fill_value=9999)

    # Clean catalog
    for j in catalog.colnames:
        # If the column type is a boolean, convert to a string
        if catalog[j].dtype == 'bool':
            catalog[j] = catalog[j].astype(str)
        for i in range(len(catalog)):
            if str(catalog[i][j]) in empties:
                try:
                    catalog[i][j] = replace_value
                except Exception:
                    catalog[j] = catalog[j].astype('float')
                    catalog[i][j] = replace_value

    # Remove rows with nan coordinates
    if 'raStack_3pi' in catalog.colnames:
        mask = np.isfinite(catalog['raStack_3pi']) & np.isfinite(catalog['decStack_3pi'])
        catalog = catalog[mask]
    elif 'ra_sdss' in catalog.colnames:
        mask = np.isfinite(catalog['ra_sdss']) & np.isfinite(catalog['dec_sdss'])
        catalog = catalog[mask]

    return catalog


def merge_duplicates(catalog, ra_key, dec_key, duplicate_distance=0.1):
    """
    Merge entries in a star catalog that are within some distance of each other, defined
    by duplicate_distance in arcseconds.

    Parameters
    ----------
    catalog : astropy.table.Table
        The catalog to process
    ra_key : str
        Name of the column containing right ascension in degrees
    dec_key : str
        Name of the column containing declination in degrees
    duplicate_distance : float
        Distance in arcsec within which entries are considered duplicates

    Returns
    -------
    single_catalog : astropy.table.Table
        Catalog with duplicates merged
    """
    if catalog is None or len(catalog) <= 1:
        return catalog

    # Ensure the specified coordinate columns exist
    if ra_key not in catalog.colnames or dec_key not in catalog.colnames:
        raise ValueError(f"Coordinate columns {ra_key} and/or {dec_key} not found in catalog")

    # Create SkyCoord objects for all sources
    coords = SkyCoord(ra=catalog[ra_key], dec=catalog[dec_key], unit='degree')

    # Find pairs within the duplicate distance
    _, d2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    duplicate_mask = d2d < duplicate_distance * u.arcsec

    if not any(duplicate_mask):
        return catalog  # No duplicates found
    else:
        print(f"Found {np.sum(duplicate_mask)} duplicates in catalog.")

    # Create a new catalog with the same column structure as the original
    single_catalog = table.Table(names=catalog.colnames,
                                 dtype=[catalog[col].dtype for col in catalog.colnames])

    # Keep track of which entries have been processed
    processed = np.zeros(len(catalog), dtype=bool)

    # Process each entry
    for i in range(len(catalog)):
        if processed[i]:
            continue  # Skip already processed entries

        # Find all entries within duplicate_distance of this one
        separations = coords[i].separation(coords)
        duplicate_indices = np.where(separations < duplicate_distance * u.arcsec)[0]

        # Mark all duplicates as processed
        processed[duplicate_indices] = True

        # If there are duplicates, merge them
        if len(duplicate_indices) > 1:
            # Create a subset of the catalog with just the duplicates
            duplicates = catalog[duplicate_indices]

            # Initialize a row for the merged entry
            merged_row = []
            for col in catalog.colnames:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        merged_row.append(np.nanmean(duplicates[col]))
                except TypeError:
                    # For non-numeric columns, use the first non-empty value
                    non_empty = [val for val in duplicates[col] if val]
                    merged_row.append(non_empty[0] if non_empty else duplicates[col][0])

            # Add to merged catalog
            single_catalog.add_row(merged_row)
        else:
            # No duplicates for this entry, just add it as is
            # Extract the row as a list of values to avoid column mismatch issues
            row_values = [catalog[i][col] for col in catalog.colnames]
            single_catalog.add_row(row_values)

    return single_catalog


def query_sdss(ra_deg, dec_deg, search_radius=1.0, DR=18,
               duplicate_distance=0.1, use_old=True,
               timeout=60):
    """
    Query SDSS for objects within a search radius of given coordinates.

    Parameters
    -----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    search_radius : float
        Search radius in arcminutes
    DR : int, default 18
        SDSS Data Release
    duplicate_distance : float, default 0.1
        Distance in arcsec to consider as duplicates
    use_old : bool, default False
        Use the old version of the query that requires
        an SQL query.
    timeout : int, default 60
        Timeout for the query in seconds

    Returns
    --------
    results : astropy.table.Table or None
        Table containing:
        - ra, dec: coordinates in degrees
        - Type: object type
            - 0 : unknown
            - 1 : Cosmic ray
            - 2 : Defect
            - 3 : Galaxy
            - 4 : Ghost
            - 5 : Known Object
            - 6 : Star
            - 7 : Trail
            - 8 : Sky
            - 9 : Not a type
        - raErr, decErr: coordinate errors in degrees
        - u,g,r,i,z: PSF magnitudes
        Returns None if query fails or no objects found
    """

    if use_old:
        # Define Query
        SDSS_query = """SELECT p.objid, -- Object ID
                p.type, -- Type of object, Galaxy (3) vs. Star (6) or other
                p.clean, -- Is the photometry flagged? (1 = Clean, 0 = Dirty)
                p.ra, p.dec, -- RA and DEC
                p.raErr, p.decErr, -- RA and DEC Errors
                p.psfMag_u,p.psfMag_g,p.psfMag_r,p.psfMag_i,p.psfMag_z, -- PSF magnitudes
                p.psfMagErr_u,p.psfMagErr_g,p.psfMagErr_r,p.psfMagErr_i,p.psfMagErr_z, -- PSF magnitudes Errors
                p.petroR50_u,p.petroR50_g,p.petroR50_r,p.petroR50_i,p.petroR50_z, -- Petrosian radius contaning 50 percent of flux
                p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,  -- FWHM of the PSF
                p.modelMagErr_u,p.modelMagErr_g,p.modelMagErr_r,p.modelMagErr_i,p.modelMagErr_z, -- Model Mag Errors
                p.modelMag_u, -- Better of DeV/Exp magnitude fit (Vaucouleurs magnitude fit / Exponential fit magnitude)
                p.modelMag_g, -- Better of DeV/Exp magnitude fit
                p.modelMag_r, -- Better of DeV/Exp magnitude fit
                p.modelMag_i, -- Better of DeV/Exp magnitude fit
                p.modelMag_z, -- Better of DeV/Exp magnitude fit
                pz.z as photoz, -- Photometric redshift
                pz.zErr as photozErr, -- Error on the photometric redshift
                s.z as specz, -- Spectroscopic redshift
                s.zErr as speczErr -- Error on the Spectroscopic redshift
                FROM PhotoObj AS p , dbo.fGetNearbyObjEq(%s, %s, %s) AS n
                LEFT JOIN SpecPhotoAll s on n.objID=s.objID
                LEFT JOIN Photoz AS pz ON pz.objID = n.objID WHERE n.objID = p.objID
                """

        # Format RA and DEC with proper precision
        ra_fmt = np.around(ra_deg, decimals=5)
        dec_fmt = np.around(dec_deg, decimals=5)

        try:
            print('Querying SDSS Catalog ...')
            try:
                results = SDSS.query_sql(SDSS_query % (ra_fmt, dec_fmt, search_radius),
                                         timeout=timeout, data_release=DR)
            except Exception as e:
                print(f"Error querying SDSS for coordinates ({ra_fmt}, {dec_fmt}): {str(e)}")
                time.sleep(2)
                results = SDSS.query_sql(SDSS_query % (ra_fmt, dec_fmt, search_radius),
                                         timeout=timeout, data_release=DR)
        except Exception as e:
            print(f"Error querying SDSS for coordinates ({ra_fmt}, {dec_fmt}): {str(e)}")
            print('Trying with Data Release 10 ...')
            try:
                results = SDSS.query_sql(SDSS_query % (ra_fmt, dec_fmt, search_radius),
                                         timeout=timeout, data_release=10)
            except Exception as e2:
                print(f"DR10 query also failed: {str(e2)}")
                return None

        # Change name of objid
        if results is None or len(results) == 0:
            return None

        # Rename objid -> objID only if present
        if 'objid' in results.colnames:
            results.rename_column('objid', 'objID')
        elif 'objID' not in results.colnames:
            print(f"SDSS query returned columns: {results.colnames}")
            print("Expected 'objid' (or 'objID') but it wasn't returned; check SDSS_query SELECT list.")
            return None

    else:
        try:
            # Convert search radius to degrees
            radius = search_radius * u.arcmin

            # Create coordinate object
            coords = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)

            # Define columns to retrieve
            photoobj_fields = [
                # Basics
                'objID', 'ra', 'dec', 'raErr', 'decErr', 'type',
                # PSF magnitudes
                'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z',
                'psfMagErr_u', 'psfMagErr_g', 'psfMagErr_r', 'psfMagErr_i', 'psfMagErr_z',
                # Model magnitudes
                'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z',
                'modelMagErr_u', 'modelMagErr_g', 'modelMagErr_r', 'modelMagErr_i', 'modelMagErr_z',
                # Petrosian radius
                'petroR50_u', 'petroR50_g', 'petroR50_r', 'petroR50_i', 'petroR50_z'
            ]

            # Query SDSS
            print("Querying SDSS...")
            results = SDSS.query_region(
                coordinates=coords,
                radius=radius,
                photoobj_fields=photoobj_fields,
                data_release=DR
            )
        except Exception as e:
            print(f"Error querying SDSS: {str(e)}")
            return None

    # Add '_sdss' suffix to all column names
    for col in results.colnames:
        results.rename_column(col, f"{col}_sdss")

    # Clean the table
    results = clean_table(results)

    # Remove obvious duplicates
    results = table.unique(results, keys='objID_sdss', keep='first')

    # Clean up duplicates
    if duplicate_distance > 0:
        results = merge_duplicates(results, 'ra_sdss', 'dec_sdss', duplicate_distance)

    # Remove any objects that don't have a single magnitude u, g, r, i, z
    sdss_mags = np.array(results['psfMag_u_sdss', 'psfMag_g_sdss', 'psfMag_r_sdss',
                                 'psfMag_i_sdss', 'psfMag_z_sdss'].to_pandas()).astype(float)
    # If all values are NaN, remove the row
    mask = np.any(np.isfinite(sdss_mags), axis=1)
    results = results[mask]
    print(f'Found {len(results)} objects\n')

    return results


def query_sdss_redshift(ra_deg=None, dec_deg=None, objID=None, search_radius=3, DR=18,
                        timeout=60):
    """
    Query SDSS to get both photo-z and spec-z of a single object.
    https://skyserver.sdss.org/dr12/en/help/browser/browser.aspx#&&history=description+PhotoObjAll+U

    Parameters
    -----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    objID : int
        SDSS object ID
    search_radius : float
        Search radius in arcseconds
    DR : int, default 18
        SDSS Data Release
    timeout : int, default 60
        Timeout for the query in seconds

    Returns
    --------
    photoz : float
        Photometric redshift
    photozErr : float
        Error in photometric redshift
    specz : float
        Spectroscopic redshift
    speczErr : float
        Error in spectroscopic redshift
    """

    # Define the SQL query to get z and zErr from Photoz and SpecPhoto tables by objID
    if objID is not None:
        SDSS_query = """
        SELECT pz.z as photoz, pz.zErr as photozErr,
               s.z as specz, s.zErr as speczErr
        FROM PhotoObj AS p
        LEFT JOIN Photoz AS pz ON p.objID = pz.objID
        LEFT JOIN SpecPhotoAll AS s ON p.objID = s.objID
        WHERE p.objID = %s
        """

        # Execute the query
        try:
            print('Querying SDSS for photo-z and spec-z...')
            results = SDSS.query_sql(SDSS_query % objID, timeout=timeout, data_release=DR)
        except Exception as e:
            print(f"Error querying SDSS for objID {objID}: {str(e)}")
            time.sleep(2)
            try:
                results = SDSS.query_sql(SDSS_query % objID, timeout=timeout, data_release=DR)
            except Exception as e:
                print(f"Second attempt failed: {str(e)}")
                return None, None, None, None

        if results and len(results) > 0:
            # Extract the z and zErr values
            photoz = results['photoz'][0] if 'photoz' in results.colnames and results['photoz'][0] is not None else None
            photozErr = results['photozErr'][0] if 'photozErr' in results.colnames and results['photozErr'][0] is not None else None
            specz = results['specz'][0] if 'specz' in results.colnames and results['specz'][0] is not None else None
            speczErr = results['speczErr'][0] if 'speczErr' in results.colnames and results['speczErr'][0] is not None else None

            print('Found Redshift for objID %s' % objID)
            return photoz, photozErr, specz, speczErr
        else:
            print(f"No results found for objID {objID}")
            return None, None, None, None

    # Otherwise, use the coordinates and do a cone search
    elif ra_deg is not None and dec_deg is not None:
        # Define the SQL query for cone search with both photoz and specz
        SDSS_query = """SELECT p.objid, -- Object ID
                p.ra, p.dec, -- RA and DEC
                pz.z as photoz, -- Photometric redshift
                pz.zErr as photozErr, -- Error on the photometric redshift
                s.z as specz, -- Spectroscopic redshift
                s.zErr as speczErr -- Error on the Spectroscopic redshift
                FROM PhotoObj AS p, dbo.fGetNearbyObjEq(%s, %s, %s) AS n
                LEFT JOIN Photoz AS pz ON pz.objID = n.objID
                LEFT JOIN SpecPhotoAll s ON n.objID = s.objID
                WHERE n.objID = p.objID
                ORDER BY n.distance ASC
                """

        # Convert search radius from arcsec to arcmin for fGetNearbyObjEq
        search_radius_arcmin = search_radius / 60.0

        # Format RA and DEC with proper precision
        ra_fmt = np.around(ra_deg, decimals=5)
        dec_fmt = np.around(dec_deg, decimals=5)

        try:
            print('Querying SDSS for photo-z and spec-z...')
            try:
                results = SDSS.query_sql(SDSS_query % (ra_fmt, dec_fmt, search_radius_arcmin),
                                         timeout=timeout, data_release=DR)
            except Exception as e:
                print(f"Error querying SDSS for coordinates ({ra_fmt}, {dec_fmt}): {str(e)}")
                time.sleep(2)
                results = SDSS.query_sql(SDSS_query % (ra_fmt, dec_fmt, search_radius_arcmin),
                                         timeout=timeout, data_release=DR)
        except Exception as e:
            print(f"Error querying SDSS for coordinates ({ra_fmt}, {dec_fmt}): {str(e)}")
            print('Trying with Data Release 10 ...')
            try:
                results = SDSS.query_sql(SDSS_query % (ra_fmt, dec_fmt, search_radius_arcmin),
                                         timeout=timeout, data_release=10)
            except Exception as e2:
                print(f"DR10 query also failed: {str(e2)}")
                return None, None, None, None

        if results and len(results) > 0:
            # Extract the z and zErr values - handling masked values properly
            photoz = results['photoz'][0] if 'photoz' in results.colnames and results['photoz'][0] is not None else None
            photozErr = results['photozErr'][0] if 'photozErr' in results.colnames and results['photozErr'][0] is not None else None
            specz = results['specz'][0] if 'specz' in results.colnames and results['specz'][0] is not None else None
            speczErr = results['speczErr'][0] if 'speczErr' in results.colnames and results['speczErr'][0] is not None else None

            # Print summary of found redshifts
            print(f"Found Redshift for object at ({ra_fmt}, {dec_fmt})")
            return photoz, photozErr, specz, speczErr
        else:
            print(f"No results found for coordinates ({ra_fmt}, {dec_fmt})")
            return None, None, None, None

    else:
        print("Error: Either objID or (ra_deg, dec_deg) must be provided")
        return None, None, None, None


def query_panstarrs(ra_deg, dec_deg, search_radius=1, DR=2,
                    duplicate_distance=0.1, use_old=True):
    """
    Query PanSTARRS DR2 3π survey for objects within a search radius of given coordinates.

    Parameters
    -----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    search_radius : float
        Search radius in acminutes
    DR : int, default 2
        PanSTARRS Data Release
    duplicate_distance : float, default 0.1
        Distance in arcsec to consider as duplicates
    use_old : bool, default False
        Use the old version of the query that requires
        an API key.

    Returns
    --------
    astropy.table.Table or None
        Table containing:
        - Identifiers: objID, distance
        - Stack positions: raStack, decStack, raStackErr, decStackErr
        - Mean positions: raStack, decStack, raStackErr, decStackErr
        - Filter-specific positions: [g,r,i,z,y]ra, [g,r,i,z,y]dec and their errors
        - PSF magnitudes: [g,r,i,z,y]PSFMag and their errors
        - Kron magnitudes: [g,r,i,z,y]KronMag and their errors
        - Kron radii: [g,r,i,z,y]KronRad
        Returns None if query fails or no objects found
    """
    if use_old:
        import mastcasjobs
        import pathlib

        # Get the PS1 MAST username and password from /Users/username/3PI_key.txt
        try:
            key_location = os.path.join(pathlib.Path.home(), '3PI_key.txt')
            wsid, password = np.genfromtxt(key_location, dtype='str')
        except Exception:
            key_location = os.path.join(fleet_data, '3PI_key.txt')
            wsid, password = np.genfromtxt(key_location, dtype='str')

        # 3PI query
        # Kron Magnitude and Radius, PSF Magnitude and radius, and sersic profile
        the_query = """
        SELECT o.objID,o.objInfoFlag,o.nDetections,o.raStack,o.decStack,o.raStackErr,o.decStackErr,nb.distance,m.primaryDetection,
        m.gKronMag,m.rKronMag,m.iKronMag,m.zKronMag,m.yKronMag,m.gPSFMag,m.rPSFMag,m.iPSFMag,m.zPSFMag,m.yPSFMag,m.gKronMagErr,m.rKronMagErr,
        m.iKronMagErr,m.zKronMagErr,m.yKronMagErr,m.gPSFMagErr,m.rPSFMagErr,m.iPSFMagErr,m.zPSFMagErr,m.yPSFMagErr,s.gSerRadius,s.gSerMag,
        s.gSerAb,s.gSerNu,s.gSerPhi,s.gSerChisq,s.rSerRadius,s.rSerMag,s.rSerAb,s.rSerNu,s.rSerPhi,s.rSerChisq,s.iSerRadius,s.iSerMag,
        s.iSerAb,s.iSerNu,s.iSerPhi,s.iSerChisq,s.zSerRadius,s.zSerMag,s.zSerAb,s.zSerNu,s.zSerPhi,s.zSerChisq,s.ySerRadius,s.ySerMag,
        s.ySerAb,s.ySerNu,s.ySerPhi,s.ySerChisq,b.gpsfTheta,b.rpsfTheta,b.ipsfTheta,b.zpsfTheta,b.ypsfTheta,b.gKronRad,b.rKronRad,
        b.iKronRad,b.zKronRad,b.yKronRad,b.gPSFFlux,b.rPSFFlux,b.iPSFFlux,b.zPSFFlux,b.yPSFFlux,b.gpsfMajorFWHM,b.rpsfMajorFWHM,
        b.ipsfMajorFWHM,b.zpsfMajorFWHM,b.ypsfMajorFWHM,b.gpsfMinorFWHM,b.rpsfMinorFWHM,b.ipsfMinorFWHM,
        b.zpsfMinorFWHM,b.ypsfMinorFWHM,psc.ps_score
        FROM fGetNearbyObjEq(%s, %s, %s) nb
        INNER JOIN ObjectThin o on o.objid=nb.objid
        INNER JOIN StackObjectThin m on o.objid=m.objid
        LEFT JOIN HLSP_PS1_PSC.dbo.pointsource_scores psc on o.objid=psc.objid
        FULL JOIN StackModelFitSer s on o.objid=s.objid
        INNER JOIN StackObjectAttributes b on o.objid=b.objid WHERE m.primaryDetection = 1
        """
        la_query = the_query % (ra_deg, dec_deg, search_radius)

        # Format Query
        print('Querying 3PI ...')
        jobs = mastcasjobs.MastCasJobs(userid=wsid, password=password, context="PanSTARRS_DR1")
        results = jobs.quick(la_query, task_name="python cone search")

        # For New format
        catalog_data = table.Table(results)
        if len(catalog_data) >= 0:
            print(f'Found {len(catalog_data)} objects\n')
            output = catalog_data
        else:
            return None
    else:
        try:
            # Create coordinate object
            coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg))

            # Define keys by category
            keys = [
                # Identifiers
                'objID', 'distance',
                # Stack positions
                'raStack', 'decStack', 'raStackErr', 'decStackErr',
                # Mean positions
                'raStack', 'decStack', 'raStackErr', 'decStackErr',
                # PSF magnitudes
                'gPSFMag', 'gPSFMagErr',
                'rPSFMag', 'rPSFMagErr',
                'iPSFMag', 'iPSFMagErr',
                'zPSFMag', 'zPSFMagErr',
                'yPSFMag', 'yPSFMagErr',
                # Kron magnitudes
                'gKronMag', 'gKronMagErr',
                'rKronMag', 'rKronMagErr',
                'iKronMag', 'iKronMagErr',
                'zKronMag', 'zKronMagErr',
                'yKronMag', 'yKronMagErr',
                # Kron radii
                'gKronRad', 'rKronRad', 'iKronRad', 'zKronRad', 'yKronRad'
            ]

            # Query PS1
            print("Querying PanSTARRS...")
            catalog_data = Catalogs.query_region(
                coordinates=coords,
                catalog="PANSTARRS",
                radius=search_radius * u.arcmin,
                data_release=f"dr{DR}",
                table="stack"
            )
            print(f'Found {len(catalog_data)} objects\n')
        except Exception as e:
            print(f"Error querying PanSTARRS: {str(e)}")
            return None

        # Rename objID to objID_PS1
        output = catalog_data[keys]

    # Add '_3pi' suffix to all column names
    for col in output.colnames:
        output.rename_column(col, f"{col}_3pi")

    # Clean the table
    output = clean_table(output)

    # Remove obvious duplicates
    output = table.unique(output, keys='objID_3pi', keep='first')

    # Clean up duplicates
    if duplicate_distance > 0:
        output = merge_duplicates(output, 'raStack_3pi', 'decStack_3pi', duplicate_distance)

    # Remove any objects that don't have a single magnitude g, r, i, z, y
    psst_mags = np.array(output['gPSFMag_3pi', 'rPSFMag_3pi', 'iPSFMag_3pi', 'zPSFMag_3pi', 'yPSFMag_3pi'].to_pandas()).astype(float)
    # If all values are NaN, remove the row
    mask = np.any(np.isfinite(psst_mags), axis=1)
    output = output[mask]

    return output


def query_gaia(ra_deg, dec_deg, search_radius=1.0, DR=3, gaia_limit=10, use_vizier=False):
    """
    Query Gaia for objects within a search radius of given coordinates.

    Parameters
    -----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    search_radius : float
        Search radius in arcsec
    DR : int, default 3
        Gaia Data Release
    gaia_limit : int, default 10
        Maximum number of sources to return
    use_vizier : bool, default False
        If True, query Gaia via VizieR instead of Gaia archive

    Returns
    --------
    catalog_gaia : astropy.table.Table or None
        Table containing Gaia data
    """

    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.degree, u.degree), frame='icrs')

    if not use_vizier:
        # ---- Original behavior (Gaia archive) ----
        Gaia.ROW_LIMIT = gaia_limit
        Gaia.MAIN_GAIA_TABLE = f"gaiadr{DR}.gaia_source"

        gaia_query = Gaia.cone_search_async(coord, radius=u.Quantity(search_radius, u.arcsec))
        gaia_table = gaia_query.get_results()

        catalog_gaia = gaia_table[
            'ra', 'ra_error', 'dec', 'dec_error',
            'parallax', 'parallax_error', 'pm',
            'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
            'phot_g_mean_mag', 'phot_bp_mean_mag',
            'phot_rp_mean_mag'
        ]

        return catalog_gaia

    else:
        # ---- VizieR behavior ----
        if DR == 3:
            catalog_id = "I/355/gaiadr3"
        elif DR == 2:
            catalog_id = "I/345/gaia2"
        else:
            raise ValueError("VizieR Gaia only supports DR2 or DR3")

        vizier = Vizier(
            catalog=catalog_id,
            columns=[
                'RA_ICRS', 'e_RA_ICRS',
                'DE_ICRS', 'e_DE_ICRS',
                'Plx', 'e_Plx',
                'pmRA', 'e_pmRA',
                'pmDE', 'e_pmDE',
                'Gmag', 'BPmag', 'RPmag'
            ],
            row_limit=gaia_limit
        )

        result = vizier.query_region(
            coord,
            radius=u.Quantity(search_radius, u.arcsec),
            catalog=catalog_id
        )

        if len(result) == 0:
            return None

        catalog_gaia = result[0]

        # Rename columns to match Gaia archive naming as closely as possible
        catalog_gaia.rename_columns(
            ['RA_ICRS', 'e_RA_ICRS', 'DE_ICRS', 'e_DE_ICRS',
             'Plx', 'e_Plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE',
             'Gmag', 'BPmag', 'RPmag'],
            ['ra', 'ra_error', 'dec', 'dec_error',
             'parallax', 'parallax_error',
             'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
             'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
        )

        # Add total proper motion column (to match Gaia archive output)
        if 'pmra' in catalog_gaia.colnames and 'pmdec' in catalog_gaia.colnames:
            catalog_gaia['pm'] = (catalog_gaia['pmra']**2 + catalog_gaia['pmdec']**2)**0.5

        return catalog_gaia

def query_wise(ra_deg, dec_deg, search_radius=1.0, data_table="allwise_p3as_psd"):
    """
    Query WISE for objects within a search radius of given coordinates.

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    search_radius : float
        Search radius in arcsec
    data_table : str
        Name of the WISE data table to query

    Returns
    -------
    catalog_wise : astropy.table.Table or None
        Table containing WISE data
    """

    # Query Catalog
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.degree, u.degree), frame='icrs')
    catalog_wise = Irsa.query_region(
        coordinates=coord,
        catalog=data_table,
        spatial="Cone",
        radius=u.Quantity(search_radius, u.arcsec),
        columns="designation,ra,dec,sigra,sigdec,sigradec,w1mag,w1sigm,w2mag,w2sigm,w3mag,w3sigm,w4mag,w4sigm"
    )

    return catalog_wise


def query_2mass(ra_deg, dec_deg, search_radius=1.0):
    """
    Query 2MASS for objects within a search radius of given coordinates.

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    search_radius : float
        Search radius in arcmin

    Returns
    -------
    catalog_2mass : astropy.table.Table or None
        Table containing 2MASS data
    """

    # Query the 2MASS database
    coord = SkyCoord(ra_deg, dec_deg, unit="deg")
    print('Querying 2MASS ...')
    new_vizier = Vizier(catalog='II/246/out', columns=['_2MASS', 'RAJ2000', 'DEJ2000', 'errMaj', 'errMin',
                                                       'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag'], row_limit=1000)
    result_table = new_vizier.query_region(coord, radius=search_radius * u.arcmin, catalog='II/246/out')

    # If there was data, select columns
    if result_table:
        catalog_2MASS = result_table['II/246/out']
        # Clean up catalog
        catalog_2MASS = clean_table(catalog_2MASS)

        # Add '_2mass' suffix to all column names
        for col in catalog_2MASS.colnames:
            catalog_2MASS.rename_column(col, f"{col}_2mass")

        # Only real values
        catalog_2MASS = catalog_2MASS[np.isfinite(catalog_2MASS['RAJ2000_2mass'])]
    else:
        catalog_2MASS = table.Table()

    print(f'Found {len(catalog_2MASS)} objects \n')
    return catalog_2MASS


def query_CFHTLS(ra_deg, dec_deg, search_radius=1.0):
    """
    Query CFHTLS for objects within a search radius of given coordinates.

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    search_radius : float
        Search radius in arcmin

    Returns
    -------
    catalog_CFHTLS : astropy.table.Table or None
        Table containing CFHTLS data
    """

    # Query the CFHTLS database
    coord = SkyCoord(ra_deg, dec_deg, unit="deg")
    print('Querying CFHTLS ...')
    new_vizier1 = Vizier(catalog='II/317/cfhtls_d', columns=['CFHTLS', 'RAJ2000', 'DEJ2000',
                                                             'umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], row_limit=100000)
    new_vizier2 = Vizier(catalog='II/317/cfhtls_d', columns=['CFHTLS', 'RAJ2000', 'DEJ2000', 'ucl', 'gcl', 'rcl', 'icl', 'zcl',
                                                             'ycl', 'umagA', 'gmagA', 'rmagA', 'imagA', 'zmagA', 'ymagA'], row_limit=100000)
    result_table1 = new_vizier1.query_region(coord, radius=search_radius * u.arcmin, catalog='II/317/cfhtls_d')
    result_table2 = new_vizier2.query_region(coord, radius=search_radius * u.arcmin, catalog='II/317/cfhtls_d')

    # If there was data, select columns
    if result_table1:
        catalog_CFHTLS1 = result_table1['II/317/cfhtls_d']
        catalog_CFHTLS2 = result_table2['II/317/cfhtls_d']

        # Join tables back
        catalog_CFHTLS = table.join(catalog_CFHTLS1, catalog_CFHTLS2)

        # Clean up catalog
        catalog_CFHTLS = clean_table(catalog_CFHTLS)

        # add '_CFHTLS' suffix to all column names
        for col in catalog_CFHTLS.colnames:
            catalog_CFHTLS.rename_column(col, f"{col}_CFHTLS")

        # Only real values
        catalog_CFHTLS = catalog_CFHTLS[np.isfinite(catalog_CFHTLS['RAJ2000_CFHTLS'])]
    else:
        catalog_CFHTLS = table.Table()

    print(f'Found {len(catalog_CFHTLS)} objects \n')
    return catalog_CFHTLS


def merge_two_catalogs(catalog_psst, catalog_sdss, match_radius_arcsec=1.5):
    """
    Merge two catalogs based on celestial coordinates with a specified match radius.

    Parameters
    ----------
    catalog_psst : astropy.table.Table
        First catalog with raStack_3pi, decStack_3pi columns
    catalog_sdss : astropy.table.Table
        Second catalog with ra_sdss, dec_sdss columns
    match_radius_arcsec : float, optional
        Match radius in arcseconds, default is 1.5

    Returns
    -------
    merged_catalog : astropy.table.Table
        Merged catalog containing all columns from both catalogs
        and new ra_matched, dec_matched columns
    """

    # Create SkyCoord objects for both catalogs
    coords_psst = SkyCoord(ra=catalog_psst['raStack_3pi'],
                           dec=catalog_psst['decStack_3pi'],
                           unit='deg')

    coords_sdss = SkyCoord(ra=catalog_sdss['ra_sdss'],
                           dec=catalog_sdss['dec_sdss'],
                           unit='deg')

    # Find matches between the two catalogs
    idx_sdss, d2d, _ = match_coordinates_sky(coords_psst, coords_sdss)

    # Create masks for matches within the radius
    matches = d2d < match_radius_arcsec * u.arcsec

    # Create a table for matched sources
    matched_sources = table.Table()

    # For matched sources:
    if np.any(matches):
        # Get indices of matched sources
        psst_indices = np.where(matches)[0]
        sdss_indices = idx_sdss[matches]

        # Copy relevant rows from each catalog
        psst_matched = catalog_psst[psst_indices]
        sdss_matched = catalog_sdss[sdss_indices]

        # Calculate average coordinates
        ra_matched = (psst_matched['raStack_3pi'] + sdss_matched['ra_sdss']) / 2.0
        dec_matched = (psst_matched['decStack_3pi'] + sdss_matched['dec_sdss']) / 2.0

        # Create the matched part of the merged catalog
        matched_sources = table.hstack([psst_matched, sdss_matched])
        matched_sources['ra_matched'] = ra_matched
        matched_sources['dec_matched'] = dec_matched

        # Create masks for unmatched sources
        unmatched_psst_mask = ~np.in1d(np.arange(len(catalog_psst)), psst_indices)
        unmatched_sdss_mask = ~np.in1d(np.arange(len(catalog_sdss)), sdss_indices)

        # Get unmatched sources
        psst_unmatched = catalog_psst[unmatched_psst_mask]
        sdss_unmatched = catalog_sdss[unmatched_sdss_mask]

        # Create empty rows for psst_unmatched in sdss format
        if len(psst_unmatched) > 0:
            empty_sdss = table.Table()
            for col_name in catalog_sdss.colnames:
                col = catalog_sdss[col_name]
                if np.issubdtype(col.dtype, np.number):  # Handle all numeric types the same
                    empty_sdss[col_name] = np.full(len(psst_unmatched), np.nan, dtype=np.float64)
                else:
                    # For string or other types
                    empty_sdss[col_name] = np.full(len(psst_unmatched), '--', dtype=col.dtype)

            # Add psst unmatched sources with empty sdss columns
            psst_only = table.hstack([psst_unmatched, empty_sdss])
            psst_only['ra_matched'] = psst_unmatched['raStack_3pi']
            psst_only['dec_matched'] = psst_unmatched['decStack_3pi']
        else:
            psst_only = table.Table()

        # Create empty rows for sdss_unmatched in psst format
        if len(sdss_unmatched) > 0:
            empty_psst = table.Table()
            for col_name in catalog_psst.colnames:
                col = catalog_psst[col_name]
                if np.issubdtype(col.dtype, np.number):  # Handle all numeric types the same
                    empty_psst[col_name] = np.full(len(sdss_unmatched), np.nan, dtype=np.float64)
                else:
                    # For string or other types
                    empty_psst[col_name] = np.full(len(sdss_unmatched), '--', dtype=col.dtype)

            # Add sdss unmatched sources with empty psst columns
            sdss_only = table.hstack([empty_psst, sdss_unmatched])
            sdss_only['ra_matched'] = sdss_unmatched['ra_sdss']
            sdss_only['dec_matched'] = sdss_unmatched['dec_sdss']
        else:
            sdss_only = table.Table()

        # Combine all parts
        if len(psst_only) > 0 and len(sdss_only) > 0:
            merged_catalog = table.vstack([matched_sources, psst_only, sdss_only])
        elif len(psst_only) > 0:
            merged_catalog = table.vstack([matched_sources, psst_only])
        elif len(sdss_only) > 0:
            merged_catalog = table.vstack([matched_sources, sdss_only])
        else:
            merged_catalog = matched_sources

    else:
        # If no matches, just stack the catalogs with empty columns for the other
        empty_sdss = table.Table()
        for col_name in catalog_sdss.colnames:
            empty_sdss[col_name] = np.full(len(catalog_psst), np.nan, dtype=catalog_sdss[col_name].dtype)

        empty_psst = table.Table()
        for col_name in catalog_psst.colnames:
            empty_psst[col_name] = np.full(len(catalog_sdss), np.nan, dtype=catalog_psst[col_name].dtype)

        psst_part = table.hstack([catalog_psst, empty_sdss])
        psst_part['ra_matched'] = catalog_psst['raStack_3pi']
        psst_part['dec_matched'] = catalog_psst['decStack_3pi']

        sdss_part = table.hstack([empty_psst, catalog_sdss])
        sdss_part['ra_matched'] = catalog_sdss['ra_sdss']
        sdss_part['dec_matched'] = catalog_sdss['dec_sdss']

        merged_catalog = table.vstack([psst_part, sdss_part])

    # Make sure only objects with real coordinates are used
    merged_catalog = merged_catalog[np.isfinite(np.array(merged_catalog['ra_matched']).astype(float))]

    return merged_catalog


def get_catalog(object_name, ra_deg, dec_deg, search_radius=1.0, reimport_catalog=False,
                catalog_dir='catalogs', save_catalog=True, use_old=True, match_radius_arcsec=1.5):
    """
    Function to query SDSS and PSST catalogs, combine them, clean them, and return the merged catalog.
    Also save the output catalog to the catalog directory.

    Parameters
    ----------
    object_name : str
        Name of the object to be queried
    ra_deg : float
        Right Ascension in degrees
    dec_deg : float
        Declination in degrees
    search_radius : float
        Search radius in arcminutes
    reimport_catalog : bool
        If True, reimport the catalog from the catalog directory
    catalog_dir : str
        Directory where the catalog is saved
    save_catalog : bool
        If True, save the catalog to the catalog directory
    use_old : bool
        If True, use the old version of the query that requires an API key
    match_radius_arcsec : float
        Match radius in arcseconds for merging catalogs

    Returns
    -------
    merged_catalog : astropy.table.Table
        Merged catalog containing data from both SDSS and PSST
    """

    # Check if the catalog already exists
    catalog_path = f"{catalog_dir}/{object_name}.cat"

    # If the catalog exists and reimport is not requested, load it
    if not reimport_catalog and os.path.exists(catalog_path):
        print(f"\nLoading existing catalog from {catalog_path}")
        merged_catalog = table.Table.read(catalog_path, format='ascii')
        return merged_catalog
    else:
        print("\nQuerying catalogs...")

    # If the catalog does not exist or reimport is requested, query the catalogs
    catalog_sdss = query_sdss(ra_deg, dec_deg, search_radius=search_radius, use_old=use_old)
    catalog_psst = query_panstarrs(ra_deg, dec_deg, search_radius=search_radius, use_old=use_old)
    if catalog_psst is None:
        merged_catalog = catalog_sdss
        merged_catalog['ra_matched'] = catalog_sdss['ra_sdss']
        merged_catalog['dec_matched'] = catalog_sdss['dec_sdss']
    elif catalog_sdss is None:
        merged_catalog = catalog_psst
        merged_catalog['ra_matched'] = catalog_psst['raStack_3pi']
        merged_catalog['dec_matched'] = catalog_psst['decStack_3pi']
    else:
        merged_catalog = merge_two_catalogs(catalog_psst, catalog_sdss, match_radius_arcsec=match_radius_arcsec)

    # Calculate separations
    separation = calc_separations(merged_catalog['ra_matched'], merged_catalog['dec_matched'],
                                  ra_deg, dec_deg)
    merged_catalog['separation'] = separation

    # Sort the catalog by separation
    merged_catalog.sort('separation')

    # Save the merged catalog to the specified directory
    if save_catalog:
        os.makedirs(catalog_dir, exist_ok=True)
        merged_catalog.write(catalog_path, format='ascii', overwrite=True)
        print(f"Saved merged catalog to {catalog_path}")

    return merged_catalog


def calc_galaxyness(data_catalog, psf_key, kron_key, classification_catalog=None,
                    neighbors=20):
    """
    Calculate the galaxyness of objects in the catalog.
    The galaxyness is defined as the fraction of neighbors that are
    classified as galaxies.

    Parameters
    ----------
    data_catalog : astropy.table.Table
        Catalog containing data to classify
    psf_key : str
        Key for the PSF magnitude in the data catalog
    kron_key : str
        Key for the Kron magnitude in the data catalog
    classification_catalog : astropy.table.Table
        Reference CFHLST catalog containing classification
        information.
    neighbors : int
        Number of nearest neighbors to consider for classification

    Returns
    -------
    galaxyness : numpy.ndarray
        Array containing the fraction of neighbors classified as galaxies
    """

    # Classify objects as stars or galaxies
    if classification_catalog is None:
        classification_catalog = cached_catalog

    # Get PSF and Kron magnitudes for the observed objects
    target_psfs = np.array(data_catalog[psf_key])
    target_krons = np.array(data_catalog[kron_key])

    # Replace with nan for PSF mags dimmer than survey limit
    survey_limit = survey_limits[psf_key]
    target_psfs[target_psfs > survey_limit] = np.nan
    target_deltamag = target_psfs - target_krons

    # Get PSF and Kron magnitudes for the classification catalog
    catalog_psfs = np.array(classification_catalog[psf_key])
    catalog_krons = np.array(classification_catalog[kron_key])
    catalog_deltamag = catalog_psfs - catalog_krons

    # Precompute squared differences for efficiency
    psf_diff_squared = (catalog_psfs[:, None] - target_psfs[None, :]) ** 2
    deltamag_diff_squared = (catalog_deltamag[:, None] - target_deltamag[None, :]) ** 2

    # Calculate the separation in mag-deltamag space
    separations = np.sqrt(psf_diff_squared + deltamag_diff_squared).T

    # Find the closest neighbors
    closest = np.argpartition(separations, neighbors, axis=1)[:, :neighbors]
    catalog_natures = np.array(classification_catalog['Nature'])[closest]

    # Get the nature of the objects for each row
    galaxyness = np.sum(catalog_natures == 1.0, axis=1) / neighbors
    nan_rows = np.isnan(target_deltamag)
    galaxyness[nan_rows] = np.nan

    return galaxyness


def calculate_nature(data_catalog, neighbors=20):
    """
    Classify objects as stars or galaxies based on the CFHLST
    classification catalog.

    Parameters
    ----------
    data_catalog : astropy.table.Table
        Catalog containing data to classify
    neighbors : int
        Number of nearest neighbors to consider for classification

    Returns
    -------
    None
    """

    # Calculate galaxyness for each filter and survey in the catalog
    print("Calculating Nature...")
    # For PanSTARRS
    if 'gPSFMag_3pi' in data_catalog.colnames:
        galaxy_g_psst = calc_galaxyness(data_catalog, 'gPSFMag_3pi', 'gKronMag_3pi', neighbors=neighbors)
        galaxy_r_psst = calc_galaxyness(data_catalog, 'rPSFMag_3pi', 'rKronMag_3pi', neighbors=neighbors)
        galaxy_i_psst = calc_galaxyness(data_catalog, 'iPSFMag_3pi', 'iKronMag_3pi', neighbors=neighbors)
        galaxy_z_psst = calc_galaxyness(data_catalog, 'zPSFMag_3pi', 'zKronMag_3pi', neighbors=neighbors)
        galaxy_y_psst = calc_galaxyness(data_catalog, 'yPSFMag_3pi', 'yKronMag_3pi', neighbors=neighbors)
    else:
        galaxy_g_psst = np.full(len(data_catalog), np.nan)
        galaxy_r_psst = np.full(len(data_catalog), np.nan)
        galaxy_i_psst = np.full(len(data_catalog), np.nan)
        galaxy_z_psst = np.full(len(data_catalog), np.nan)
        galaxy_y_psst = np.full(len(data_catalog), np.nan)
    # For SDSS
    if 'psfMag_u_sdss' in data_catalog.colnames:
        galaxy_u_sdss = calc_galaxyness(data_catalog, 'psfMag_u_sdss', 'modelMag_u_sdss', neighbors=neighbors)
        galaxy_g_sdss = calc_galaxyness(data_catalog, 'psfMag_g_sdss', 'modelMag_g_sdss', neighbors=neighbors)
        galaxy_r_sdss = calc_galaxyness(data_catalog, 'psfMag_r_sdss', 'modelMag_r_sdss', neighbors=neighbors)
        galaxy_i_sdss = calc_galaxyness(data_catalog, 'psfMag_i_sdss', 'modelMag_i_sdss', neighbors=neighbors)
        galaxy_z_sdss = calc_galaxyness(data_catalog, 'psfMag_z_sdss', 'modelMag_z_sdss', neighbors=neighbors)
    else:
        galaxy_u_sdss = np.full(len(data_catalog), np.nan)
        galaxy_g_sdss = np.full(len(data_catalog), np.nan)
        galaxy_r_sdss = np.full(len(data_catalog), np.nan)
        galaxy_i_sdss = np.full(len(data_catalog), np.nan)
        galaxy_z_sdss = np.full(len(data_catalog), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Combine the results into a single array
        nature = np.nanmean([
            galaxy_g_psst, galaxy_r_psst, galaxy_i_psst, galaxy_z_psst, galaxy_y_psst,
            galaxy_u_sdss, galaxy_g_sdss, galaxy_r_sdss, galaxy_i_sdss, galaxy_z_sdss
        ], axis=0)

    # Replace any np.nan values in the resulting array with 0.5
    nature[np.isnan(nature)] = 0.5

    return nature


def default_radius(mag, a=196.0, b=0.257, c=-0.27, minimum_halflight=0.7):
    '''
    Relation between the host galaxy magnitude and most likely
    half-light radius in arcseconds.

    Parameters
    ----------
    mag : float or array-like
        Magnitude
    a, b, c : float
        Default parameters of the function
    minimum_halflight : float
        Default half light radius if none was found (default is 0.7)

    Returns
    -------
    radius : float or array-like
        Half-light radius value in arcsec
    '''
    radius = a * np.exp(-b * mag) + c
    radius[radius < minimum_halflight] = minimum_halflight
    return radius


def get_halflight(data_catalog, color, host_mag, minimum_halflight=0.7):
    '''
    Get the best estimate of the half light radius for
    either SDSS or 3PI

    Parameters
    ----------
    data_catalog : astropy.table.Table
        Catalog containing data to calculate halflight radius
    color : str
        Band to use for the calculation
    host_mag : array
        Array of host magnitudes used to estimate missing radii
    minimum_halflight : float
        Default half light radius if none was found (default is 0.7)

    Returns
    -------
    halflight_radius : np.array
        Array containing the half light radius in arcseconds
    '''

    if f'{color}KronRad_3pi' in data_catalog.colnames:
        # Get Sersic Index from 3PI
        band_name = f'{color}SerNu_3pi'
        if band_name in data_catalog.colnames:
            sersic_n = np.copy(data_catalog[band_name])
            # Assume a sersic index of 0.5 if there is none
            sersic_n[np.isnan(sersic_n)] = 0.5
        else:
            # Assume a sersic index of 0.5 for all objects that dont have one
            sersic_n = np.ones(len(data_catalog)) * 0.5

        # Get Sersic normalization, to convert to half light radius
        R_e = 2.5
        radius = 100000
        b_n = 1.9992 * sersic_n - 0.3271
        x = b_n * (radius / R_e) ** (1 / sersic_n)
        R_norm = ((R_e / b_n ** sersic_n) * (gammainc(3 * sersic_n, x) /
                                             gammainc(2 * sersic_n, x)) * (gamma(3 * sersic_n) / gamma(2 * sersic_n)))

        # Normalize Kron radius to half light radius
        if f'{color}KronRad_3pi' in data_catalog.colnames:
            halflight_radius = data_catalog[f'{color}KronRad_3pi'] / R_norm
        else:
            halflight_radius = np.nan * np.ones(len(data_catalog))
    elif f'petroR50_{color}_sdss' in data_catalog.colnames:
        halflight_radius = data_catalog[f'petroR50_{color}_sdss']
    else:
        halflight_radius = np.nan * np.ones(len(data_catalog))

    # Default half-light radius if none was found (0.7)
    default_halflight = default_radius(host_mag[np.isnan(halflight_radius)], minimum_halflight=minimum_halflight)
    halflight_radius[np.isnan(halflight_radius)] = default_halflight

    return halflight_radius


def get_host_mag(data_catalog, band, type='psf', survey=None,
                 impute_values=True):
    """
    Get the PSF magnitude of catalog in a given band.
    If PSST and SDSS both exist, average them; if not pick
    just the one that exists.

    Parameters
    ----------
    data_catalog : astropy.table.Table
        Catalog containing data to calculate host magnitude
    band : str
        Band to use for the calculation
    type : str
        Type of magnitude to use ('psf' or 'kron')
    survey : str
        Survey to use for the calculation (default is None)
        Other options are '3pi' or 'sdss'
    impute_values : bool
        If True, it replaces the NaN values of Kron
        magnitudes with the PSF magnitudes

    Returns
    -------
    host_mag : np.array
        Host magnitude in the specified band
    """

    if type == 'psf':
        # Check if the band exists in the catalog
        if f'{band}PSFMag_3pi' in data_catalog.colnames and f'psfMag_{band}_sdss' in data_catalog.colnames and survey is None:
            # If both exist, average them
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                host_mag = np.nanmean([data_catalog[f'{band}PSFMag_3pi'], data_catalog[f'psfMag_{band}_sdss']], axis=0)
        elif (f'{band}PSFMag_3pi' in data_catalog.colnames and survey is None) or (survey == '3pi'):
            # If only PSST exists
            host_mag = data_catalog[f'{band}PSFMag_3pi']
        elif (f'psfMag_{band}_sdss' in data_catalog.colnames and survey is None) or (survey == 'sdss'):
            # If only SDSS exists
            host_mag = data_catalog[f'psfMag_{band}_sdss']
        else:
            # If neither exists, return NaN
            host_mag = np.nan * np.ones(len(data_catalog))
    elif type == 'kron':
        # Check if the band exists in the catalog
        if f'{band}KronMag_3pi' in data_catalog.colnames and f'modelMag_{band}_sdss' in data_catalog.colnames and survey is None:
            # If both exist, average them
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                host_mag = np.nanmean([data_catalog[f'{band}KronMag_3pi'], data_catalog[f'modelMag_{band}_sdss']], axis=0)
        elif (f'{band}KronMag_3pi' in data_catalog.colnames and survey is None) or (survey == '3pi'):
            # If only PSST exists
            host_mag = data_catalog[f'{band}KronMag_3pi']
        elif (f'modelMag_{band}_sdss' in data_catalog.colnames and survey is None) or (survey == 'sdss'):
            # If only SDSS exists
            host_mag = data_catalog[f'modelMag_{band}_sdss']
        else:
            # If neither exists, return NaN
            host_mag = np.nan * np.ones(len(data_catalog))
        # Replace the nan values with the PSF magnitude
        if impute_values:
            if f'{band}PSFMag_3pi' in data_catalog.colnames:
                host_mag[~np.isfinite(host_mag)] = data_catalog[f'{band}PSFMag_3pi'][~np.isfinite(host_mag)]
            elif f'psfMag_{band}_sdss' in data_catalog.colnames:
                host_mag[~np.isfinite(host_mag)] = data_catalog[f'psfMag_{band}_sdss'][~np.isfinite(host_mag)]
    else:
        raise ValueError("Invalid type specified. Use 'psf' or 'kron'.")

    # Replace NaN values with the upper limit
    host_mag[~np.isfinite(host_mag)] = host_limit[band]
    return host_mag


def calculate_coincidence(separation, size, magnitude):
    '''
    Calculate the chance that a galaxy of size R_h and magnitude M falls
    within a separation R of a transient. The galaxies with the lowest
    chance probability will be selected as the best candidate hosts.

    Parameters
    ----------
    separation : float or np.array
        Separation between the host and transient [Arcseconds]
    size : float or np.array
        Half light radius of the galaxy [Arcseconds]
    Magnitude : float or np.array
        Magnitude of the galaxy

    Output
    ---------------
    P_cc : float or np.array
        Probability of chance coincidence
    '''
    # Observed number density of galaxies brighter than magnitude M (From Berger 2010)
    sigma = 10 ** (0.33 * (magnitude - 24) - 2.44) / (0.33 * np.log(10))
    # Effective radius
    R_effective = np.sqrt(np.abs(separation) ** 2 + 4 * np.abs(size) ** 2)
    # Probability of chance coincidence
    chance_coincidence = 1 - np.exp(-np.pi * R_effective ** 2 * sigma)

    return chance_coincidence


def catalog_operations(object_name, merged_catalog, ra_deg, dec_deg, Pcc_filter='i',
                       Pcc_filter_alternative='r', neighbors=20, recalculate_nature=False,
                       dust_map='SFD', minimum_halflight=0.7):
    """
    For all entries in a catalog, calculate and correct for extinction, estimate the nature
    of the object (galaxy vs. star), calculate the separation between the transient ra and dec
    and each object in the catalog, and calculate the probability of chance coincidence for each
    object.

    Parameters
    ----------
    object_name : str
        Name of the object to be queried
    merged_catalog : astropy.table.Table
        Merged catalog containing data from both SDSS and PSST
    ra_deg : float
        Right Ascension of the transient in degrees
    dec_deg : float
        Declination of the transient in degrees
    Pcc_filter : str
        Filter to use for Pcc calculation (default is 'i')
    Pcc_filter_alternative : str
        Alternative filter to use for Pcc calculation (default is 'r')
    neighbors : int
        Number of nearest neighbors to consider for Pcc calculation (default is 20)
    recalculate_nature : bool
        If True, recalculate the nature of the objects (default is False)
    dust_map : str
        Dust map to use for extinction calculation (default is 'SFD')
    minimum_halflight : float
        Default half light radius if none was found (default is 0.7)

    Returns
    -------
    data_catalog : astropy.table.Table
        Catalog with additional columns for extinction, nature, separation, and Pcc
    """

    # Query dust map
    E_BV = query_dust(merged_catalog['ra_matched'], merged_catalog['dec_matched'], dust_map=dust_map)
    R_V = 3.1
    ext = G23(Rv=R_V)

    # For each magnitude column, subtract the corresponding exintction at the right wavelength
    bands_psst = ['g', 'r', 'i', 'z', 'y']
    bands_sdss = ['u', 'g', 'r', 'i', 'z']

    # Copy catalog to avoid modifying the original
    data_catalog = merged_catalog.copy()

    if 'gPSFMag_3pi' in data_catalog.colnames:
        for band in bands_psst:
            cenwaves = np.array(psst_refs[band]) * u.AA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                correction = -2.5 * np.log10(ext.extinguish(cenwaves, Ebv=E_BV))

            data_catalog[f'{band}PSFMag_3pi'] -= correction
            data_catalog[f'{band}KronMag_3pi'] -= correction

    if 'psfMag_g_sdss' in data_catalog.colnames:
        for band in bands_sdss:
            cenwaves = np.array(sdss_refs[band]) * u.AA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                correction = -2.5 * np.log10(ext.extinguish(cenwaves, Ebv=E_BV))

            data_catalog[f'psfMag_{band}_sdss'] -= correction
            data_catalog[f'modelMag_{band}_sdss'] -= correction

    # Calculate the nature of the objects
    if recalculate_nature or 'object_nature' not in data_catalog.colnames:
        nature = calculate_nature(data_catalog, neighbors=neighbors)
        data_catalog['object_nature'] = nature

        # Save nature to input catalog
        merged_catalog['object_nature'] = nature
        merged_catalog.write(f'catalogs/{object_name}.cat', format='ascii', overwrite=True)

    # Calculate separations
    if 'separation' not in data_catalog.colnames:
        separation = calc_separations(data_catalog['ra_matched'], data_catalog['dec_matched'],
                                      ra_deg, dec_deg)
        data_catalog['separation'] = separation
    else:
        separation = data_catalog['separation']

    # Get host magnitudes in Pcc filter and Pcc filter alternative
    host_mag = get_host_mag(data_catalog, Pcc_filter, type='kron')
    host_mag_alternative = get_host_mag(data_catalog, Pcc_filter_alternative, type='kron')
    # Calculate half-light radius in Pcc filter and Pcc filter alternative
    halflight_radius = get_halflight(data_catalog, Pcc_filter, host_mag, minimum_halflight)
    halflight_radius_alternative = get_halflight(data_catalog, Pcc_filter_alternative, host_mag_alternative,
                                                 minimum_halflight)
    # Calculate probability of chance coincidence
    P_cc = calculate_coincidence(separation, halflight_radius, host_mag)
    P_cc_alternative = calculate_coincidence(separation, halflight_radius_alternative, host_mag_alternative)

    # Replace P_cc with Pcc_aletrnative if it is Nan
    P_cc[np.isnan(P_cc)] = P_cc_alternative[np.isnan(P_cc)]
    host_mag[np.isnan(host_mag)] = host_mag_alternative[np.isnan(host_mag)]
    halflight_radius[np.isnan(halflight_radius)] = halflight_radius_alternative[np.isnan(halflight_radius)]

    # Add columns to catalog
    data_catalog['chance_coincidence'] = P_cc
    data_catalog['effective_magnitude'] = host_mag
    data_catalog['halflight_radius'] = halflight_radius
    data_catalog['host_magnitude_g'] = get_host_mag(data_catalog, 'g', type='kron')
    data_catalog['host_magnitude_r'] = get_host_mag(data_catalog, 'r', type='kron')

    return data_catalog


def overwrite_with_glade(ra_deg, dec_deg, object_name, data_catalog, glade_cat=None,
                         max_separation_glade=60.0, dimmest_glade=16.0,
                         max_pcc_glade=0.01, max_distance_glade=1.0):
    """
    If overwrite GLADE is on, replace the best host with one found in the
    GLADE catalog.

    Parameters
    ----------
    ra_deg : float
        Right Ascension of the transient in degrees
    dec_deg : float
        Declination of the transient in degrees
    object_name : str
        Name of the object to be queried
    data_catalog : astropy.table.Table
        Catalog containing data to calculate host magnitude
    glade_cat : astropy.table.Table, optional
        GLADE catalog to use for replacement
    max_separation_glade : float, optional
        Maximum separation in arcseconds for GLADE replacement
    dimmest_glade : float, optional
        Dimmest magnitude for GLADE replacement
    max_pcc_glade : float, optional
        The maximum Pcc allowed for GLADE replacement
    max_distance_glade : float, optional
        The maximum distance between a GLADE object
        and the object in PSST/SDSS

    Returns
    -------
    best_host_glade : int
        Index of the best host in the data_catalog
    """

    if glade_cat is None:
        glade_cat = cached_glade

    # Calculate separations
    separations = calc_separations(glade_cat['RA'], glade_cat['DEC'], ra_deg, dec_deg)

    # Only select nearby bright objects with a redshift
    use = np.where((separations < max_separation_glade) &
                   (glade_cat['z'] > 0) &
                   (glade_cat['z'] < 0.1) &
                   (glade_cat['B'] < dimmest_glade))[0]

    if len(use) > 0:
        # Calculate Pcc
        glade_pcc = calculate_coincidence(separations[use], 1, glade_cat['B'][use])

        # Find the best host
        best_glade = np.argmin(glade_pcc)
        best_ra = glade_cat['RA'][use][best_glade]
        best_dec = glade_cat['DEC'][use][best_glade]
        best_pcc = glade_pcc[best_glade]

        # Find the closest match in data_catalog
        data_separations = calc_separations(data_catalog['ra_matched'], data_catalog['dec_matched'],
                                            best_ra, best_dec)
        best_host = np.argmin(data_separations)
        best_host_separation = data_separations[best_host]

        # Replace if the objects are close and the P_cc is low
        if (best_host_separation < max_distance_glade) and (best_pcc < max_pcc_glade):
            print('Overwriting host with GLADE ...')
            return best_host
        else:
            return None
    else:
        print('No GLADE objects found ...')
        return None


def get_best_host(data_catalog, star_separation=1.0, star_cut=0.1, best_index=None,
                  pcc_pcc_threshold=0.02, pcc_distance_threshold=8.0):
    """
    Find the best host in the data_catalog based on the Pcc and rulling out stars.

    Parameters
    ----------
    data_catalog : astropy.table.Table
        Catalog containing data to calculate host magnitude
    star_separation : float, optional
        Maximum separation in arcseconds to associate a star
    star_cut : float, optional
        Maximum allowed probability of galaxyness to call
        something a star.
    best_index : int, optional
        Force the index of the best host in the data_catalog
    pcc_pcc_threshold : float, optional
        Maximum allowed Pcc for the best host
    pcc_distance_threshold : float, optional
        Maximum allowed distance for the best host

    Returns
    -------
    host_radius : float
        Half light radius of the best host
    host_separation : float
        Separation of the best host
    host_ra : float
        Right Ascension of the best host
    host_dec : float
        Declination of the best host
    host_Pcc : float
        Probability of chance coincidence for the best host
    host_magnitude : float
        Magnitude of the best host
    host_magnitude_g : float
        g-band magnitude of the best host
    host_magnitude_r : float
        r-band magnitude of the best host
    host_nature : float
        Nature of the best host
    photoz : float
        Photometric redshift of the best host
    photoz_err : float
        Error in the photometric redshift of the best host
    specz : float
        Spectroscopic redshift of the best host
    specz_err : float
        Error in the spectroscopic redshift of the best host
    best_host : int
        Index of the best host in the data_catalog
    force_detection : bool
        Force detection flag for the best host
    """

    # Empty variables
    host_radius = None
    host_separation = None
    host_ra = None
    host_dec = None
    host_Pcc = None
    host_magnitude = None
    host_magnitude_g = None
    host_magnitude_r = None
    host_nature = None
    photoz = None
    photoz_err = None
    specz = None
    specz_err = None
    best_host = None
    force_detection = False

    if best_index is not None:
        # If a best index from GLADE is provided, use it
        best_host = best_index
        force_detection = True
    else:
        # If there is a nearby star, pick that as the host
        use_catalog = data_catalog[((data_catalog['object_nature'] <= star_cut) & (data_catalog['separation'] < star_separation)) |
                                   (data_catalog['object_nature'] > star_cut)]
        if len(use_catalog) > 0:
            best_galaxy = np.nanargmin(use_catalog['chance_coincidence'])
            best_pcc = use_catalog['chance_coincidence'][best_galaxy]
            best_separation = use_catalog['separation'][best_galaxy]

            # Use a step function for the Pcc association to remove very far outliers
            if (best_separation > 8) & (best_pcc > pcc_pcc_threshold):
                close_galaxies = use_catalog[use_catalog['separation'] <= pcc_distance_threshold]
                if len(close_galaxies) > 0:
                    # If there are close galaxies, pick the one with the lowest Pcc
                    best_close = np.nanargmin(close_galaxies['chance_coincidence'])
                    best_pcc = close_galaxies['chance_coincidence'][best_close]
        else:
            return (host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_magnitude_g, host_magnitude_r,
                    host_nature, photoz, photoz_err, specz, specz_err, best_host, force_detection)

        # If there is a very bright SDSS object, overwrite it and pick that one
        if 'modelMag_r_sdss' in use_catalog.colnames:
            # Select if modelMag_r_sdss < 16, type_sdss is 0 or 3, and object_nature > 0.0
            bright_galaxies = use_catalog[(use_catalog['modelMag_r_sdss'] < 16) & ((use_catalog['type_sdss'] == 0) | (use_catalog['type_sdss'] == 3)) &
                                          (use_catalog['chance_coincidence'] < pcc_pcc_threshold)]
            if len(bright_galaxies) > 0:
                best_bright = np.nanargmin(bright_galaxies['chance_coincidence'])
                bright_pcc = bright_galaxies['chance_coincidence'][best_bright]
                if bright_pcc != best_pcc:
                    force_detection = True
                    best_pcc = bright_pcc

        # Find the correct index of the best host in the input catalog
        best_host = np.where(data_catalog['chance_coincidence'] == best_pcc)[0][0]

        # Get properties of the best host
        host_radius = data_catalog['halflight_radius'][best_host]
        host_separation = data_catalog['separation'][best_host]
        host_ra = data_catalog['ra_matched'][best_host]
        host_dec = data_catalog['dec_matched'][best_host]
        host_Pcc = data_catalog['chance_coincidence'][best_host]
        host_magnitude = data_catalog['effective_magnitude'][best_host]
        host_magnitude_g = data_catalog['host_magnitude_g'][best_host]
        host_magnitude_r = data_catalog['host_magnitude_r'][best_host]
        host_nature = data_catalog['object_nature'][best_host]
        photoz = data_catalog['photoz'][best_host] if 'photoz' in data_catalog.colnames else None
        photoz_err = data_catalog['photoz_err'][best_host] if 'photoz_err' in data_catalog.colnames else None
        specz = data_catalog['specz'][best_host] if 'specz' in data_catalog.colnames else None
        specz_err = data_catalog['specz_err'][best_host] if 'specz_err' in data_catalog.colnames else None

    return (host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_magnitude_g, host_magnitude_r,
            host_nature, photoz, photoz_err, specz, specz_err, best_host, force_detection)
