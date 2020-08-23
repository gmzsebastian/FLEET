from scipy.special import gamma, gammainc
from astropy.coordinates import SkyCoord
from xml.etree import ElementTree
from dustmaps.sfd import SFDQuery
from astroquery.sdss import SDSS
from astropy import table
import pkg_resources
import numpy as np
import mastcasjobs
import extinction
import requests
import warnings
import pathlib
import glob
import time
import os

def angular_separation(lon1, lat1, lon2, lat2):
    '''
    Computes on-sky separation between one coordinate and another.

    Parameters
    ----------
    Coordinates in degrees

    Returns
    -------
    Separation in arcseconds
    '''

    # Convert to Radians
    RA1, DEC1, RA2, DEC2 = lon1 * np.pi / 180, lat1 * np.pi / 180, lon2 * np.pi / 180, lat2 * np.pi / 180

    # Do Math
    sdlon = np.sin(RA2 - RA1)
    cdlon = np.cos(RA2 - RA1)
    slat1 = np.sin(DEC1)
    slat2 = np.sin(DEC2)
    clat1 = np.cos(DEC1)
    clat2 = np.cos(DEC2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.arctan2(np.hypot(num1, num2), denominator) * 3600 * 180 / np.pi

def query_dust(ra_deg, dec_deg, dust_map = 'SFD'):
    '''
    Query dust maps to get reddening value. In order to use 
    the 'SF' map you need to download the dust maps, which 
    are queried locally by doing:

    from dustmaps.config import config
    config['data_dir'] = '/path/to/store/maps/in'

    import dustmaps.sfd
    dustmaps.sfd.fetch()

    The 'SFD' dust map will use a slower, online query

    Parameters
    ---------------
    RA_deg, DEC_deg : Coordinates of the object in degrees.
    dust_map: 'SF' or 'SFD', to query Schlafy and Finkbeiner 2011
               or Schlafy, Finkbeiner and Davis 1998
              set to 'none' to not correct for extinction

    Returns
    ---------------
    One merged Astropy Table catalog
    '''

    if dust_map == 'none':
        return 0

    if dust_map == 'SF':
        # Generate URL to query
        dust_url = 'https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr=%s+%s+equ+j2000'%(ra_deg, dec_deg)
        response = requests.get(dust_url)
        # Create xml response Tree
        tree = ElementTree.fromstring(response.content)
        # Extract mean reddening value from S&F
        for child in tree.iter('meanValueSandF'):
            reddeningSandF = child.text.replace('\n','').replace(' ','').replace('(mag)','')
        return float(reddeningSandF)
    elif dust_map == 'SFD':
        coord = SkyCoord(ra_deg, dec_deg, unit="deg")
        sfd = SFDQuery()
        ebv = sfd(coord)
        return ebv
    else:
        print("'dust_map' must be 'SF' or 'SFD'")
        return

flot = lambda x : np.array(x).astype(float) 

def merge_two_catalogs(catalog_1, catalog_2, ra_1, ra_2, dec_1, dec_2, max_separation = 1.5):
    '''
    Merge two catalogs based on RA and DEC, given a maximum separation between objects
    to match them.

    Parameters
    ---------------
    catalog_1, catalog_2 : The astropy tables with the catalogs
    ra_1, ra_2           : The names of the keywords with RA in cat1 and cat2
    dec_1, dec_2         : The names of the keywords with DEC in cat1 and cat2
    max_separation       : Maximum distance between objects in catalog to match

    Returns
    ---------------
    One merged Astropy Table catalog
    '''

    # Do catalogs have a length > 0
    cat1_exists  = True if len(catalog_1 ) > 0 else False
    cat2_exists  = True if len(catalog_2 ) > 0 else False

    # If any catalog has any sources, continue
    if np.array([cat1_exists, cat2_exists]).any():
        # Copy Catalogs or create empty tables
        cat_1 = table.Table(catalog_1) if cat1_exists else table.Table()
        cat_2 = table.Table(catalog_2) if cat2_exists else table.Table()

        # Get RAs and DECs if the catalogs exist
        ra1  = flot(cat_1[ra_1 ]) if cat1_exists else np.array([])
        ra2  = flot(cat_2[ra_2 ]) if cat2_exists else np.array([])
        dec1 = flot(cat_1[dec_1]) if cat1_exists else np.array([])
        dec2 = flot(cat_2[dec_2]) if cat2_exists else np.array([])

        # Add empty ra_matched and dec_matched columns to every catalog
        cat_1.add_column(table.Column(np.nan * np.ones(len(cat_1))), name = 'ra_matched' )
        cat_1.add_column(table.Column(np.nan * np.ones(len(cat_1))), name = 'dec_matched')

        cat_2.add_column(table.Column(np.nan * np.ones(len(cat_2))), name = 'ra_matched' )
        cat_2.add_column(table.Column(np.nan * np.ones(len(cat_2))), name = 'dec_matched')

        # List of RAs and DECs and corresponding catalog
        all_ras  = np.concatenate(( ra1,  ra2))
        all_dec  = np.concatenate((dec1, dec2))
        all_cats = np.concatenate([['1'] * len(cat_1),['2'] * len(cat_2)])

        # Length of catalogs
        catalogs_lengths = np.array([len(cat_1),len(cat_2)])
        catalogs_heads   = np.append(0, np.cumsum(catalogs_lengths))

        # Empty variable to fill in
        merged_list = np.array([])

        # For each object, match to any other object that exists
        for i in range(len(all_ras)):
            if i not in merged_list:
                distances = angular_separation(all_ras[i], all_dec[i], all_ras, all_dec)
                # Every object
                matched = np.where(distances < max_separation)[0]
                # Every object in other catalogs
                matched_medium = matched[all_cats[matched] != all_cats[i]]
                # Every object, but not objects that have been matched before
                matched_final = matched_medium[[k not in merged_list for k in matched_medium]]
                # Every object in other catalogs, and the object itself
                matched_final_plus = np.append(i, matched_final)

                # Calculate average RA and DEC
                matched_ras  = np.nanmean([all_ras[matched_final_plus]])
                matched_decs = np.nanmean([all_dec[matched_final_plus]])

                # Which catalogs does this correspond to 
                matched_cats = all_cats[matched_final_plus]

                # Remove one if there are two objects in the same catalog
                if len(matched_cats) != len(np.unique(matched_cats)):
                    final_cat_list = np.array([])
                    final_star_list = np.array([])
                    for i in range(len(matched_final_plus)):
                        if matched_cats[i] not in final_cat_list:
                            final_cat_list  = np.append(final_cat_list , matched_cats[i]      )
                            final_star_list = np.append(final_star_list, matched_final_plus[i])
                    final_star_list = final_star_list.astype(int)
                else:
                    final_cat_list  = matched_cats
                    final_star_list = matched_final_plus

                # Final Matched RAs
                cat_1['ra_matched'][final_star_list[np.where(final_cat_list == '1')[0]] - catalogs_heads[0]]  = matched_ras
                cat_2['ra_matched'][final_star_list[np.where(final_cat_list == '2')[0]] - catalogs_heads[1]]  = matched_ras

                # Final Matched DECs
                cat_1['dec_matched'][final_star_list[np.where(final_cat_list == '1')[0]] - catalogs_heads[0]] = matched_decs
                cat_2['dec_matched'][final_star_list[np.where(final_cat_list == '2')[0]] - catalogs_heads[1]] = matched_decs

                # Add stars to list of read stars
                merged_list = np.append(merged_list, matched_final)

        # Final Joined Catalog with matches RA and DEC
        joined_catalog = table.Table(np.array([np.nan, np.nan]), names=('ra_matched', 'dec_matched'))
        if cat1_exists  : joined_catalog = table.join(joined_catalog, cat_1 , uniq_col_name = 'ra_matched, dec_matched', join_type = 'outer')
        if cat2_exists  : joined_catalog = table.join(joined_catalog, cat_2 , uniq_col_name = 'ra_matched, dec_matched', join_type = 'outer')

        # Make format uniform
        for column_name in joined_catalog.colnames:
            joined_catalog[column_name].format = ''
            nans = np.nan * np.ones(len(joined_catalog))
            nans = nans.astype('str')
            try:
                valued   = joined_catalog[column_name].data.mask == False
            except:
                valued   = np.isfinite(joined_catalog[column_name].data)
            nans[valued] = np.array(joined_catalog[column_name][valued]).astype('str')
            joined_catalog[column_name] = table.Column(data = nans, name = column_name, dtype = 'str')
        # Clean empty cells in catalog
        joined_catalog = make_nan(joined_catalog)

        return joined_catalog[:-1]
    else:
        return table.Table()

def make_nan(catalog, replace = np.nan):
    '''
    Go through an astropy table and covert any empty values
    into a single aunified value specified by 'replace'
    '''

    for i in range(len(catalog)):
        for j in catalog[i].colnames:
            if str(catalog[i][j]) in [False, 'False', '', '-999', '-999.0', '--', 'n', '-9999.0', 'nan', b'']:
                catalog[i][j] = replace

    return catalog

def query_3pi(ra_deg, dec_deg, search_radius = 1.0):
    '''
    This program is meant as a client side example of querying a PSPS Database via Data 
    Retrevial Layer DRL from a query in a file and writing the results to a file.
    The function will only return objects with at least one detection

    # The list of parameters you can query is in: 
    https://outerspace.stsci.edu/display/PANSTARRS/PS1+StackObjectAttributes+table+fields
    The default jobType is fast, slow is the other option.

    # To use this function you need to have a 3PI_key.txt file in your home directory
    /Users/username/3PI_key.txt, the file should have the user_id and password in that order
    and separated by a space.

    Parameters
    ---------------
    ra_deg, dec_deg : Coordinates of the object in degrees.
    search_radius   : Search radius in arcminutes

    Returns
    -------------
    Table with data outlined in the the_query variable below

    '''

    # Get the PS1 MAST username and password from /Users/username/3PI_key.txt
    key_location = os.path.join(pathlib.Path.home(), '3PI_key.txt')
    wsid, password = np.genfromtxt(key_location, dtype = 'str')

    # 3PI query
    # Kron Magnitude and Radius, PSF Magnitude and radius, and sersic profile
    the_query = """
    SELECT o.objID,o.objInfoFlag,o.nDetections,o.raStack,o.decStack,o.raStackErr,o.decStackErr,nb.distance,m.primaryDetection,m.gKronMag,
    m.rKronMag,m.iKronMag,m.zKronMag,m.yKronMag,m.gPSFMag,m.rPSFMag,m.iPSFMag,m.zPSFMag,m.yPSFMag,m.gKronMagErr,m.rKronMagErr,
    m.iKronMagErr,m.zKronMagErr,m.yKronMagErr,m.gPSFMagErr,m.rPSFMagErr,m.iPSFMagErr,m.zPSFMagErr,m.yPSFMagErr,s.gSerRadius,s.gSerMag,
    s.gSerAb,s.gSerNu,s.gSerPhi,s.gSerChisq,s.rSerRadius,s.rSerMag,s.rSerAb,s.rSerNu,s.rSerPhi,s.rSerChisq,s.iSerRadius,s.iSerMag,
    s.iSerAb,s.iSerNu,s.iSerPhi,s.iSerChisq,s.zSerRadius,s.zSerMag,s.zSerAb,s.zSerNu,s.zSerPhi,s.zSerChisq,s.ySerRadius,s.ySerMag,
    s.ySerAb,s.ySerNu,s.ySerPhi,s.ySerChisq,b.gpsfTheta,b.rpsfTheta,b.ipsfTheta,b.zpsfTheta,b.ypsfTheta,b.gKronRad,b.rKronRad,
    b.iKronRad,b.zKronRad,b.yKronRad,b.gPSFFlux,b.rPSFFlux,b.iPSFFlux,b.zPSFFlux,b.yPSFFlux,b.gpsfMajorFWHM,b.rpsfMajorFWHM,
    b.ipsfMajorFWHM,b.zpsfMajorFWHM,b.ypsfMajorFWHM,b.gpsfMinorFWHM,b.rpsfMinorFWHM,b.ipsfMinorFWHM,b.zpsfMinorFWHM,b.ypsfMinorFWHM, psc.ps_score
    FROM fGetNearbyObjEq(%s, %s, %s) nb
    INNER JOIN ObjectThin o on o.objid=nb.objid
    INNER JOIN StackObjectThin m on o.objid=m.objid
    INNER JOIN HLSP_PS1_PSC.pointsource_scores psc on o.objid=psc.objid
    FULL JOIN StackModelFitSer s on o.objid=s.objid
    INNER JOIN StackObjectAttributes b on o.objid=b.objid WHERE m.primaryDetection = 1
    """ 
    la_query = the_query%(ra_deg, dec_deg, search_radius)
    
    # Format Query
    print('Querying 3PI ...')
    jobs    = mastcasjobs.MastCasJobs(userid=wsid, password=password, context="PanSTARRS_DR1")
    results = jobs.quick(la_query, task_name="python cone search")

    # For New format
    if type(results) != str:
        catalog_3pi = table.Table(results, dtype=[str] * len(results.columns))
        if len(catalog_3pi) == 0:
            print('Found %s objects'%len(catalog_3pi))
            return catalog_3pi

    # For Old format (Probably deprecated?)
    else:
        # Format data into astropy Table
        table_rows  = results.split()
        new_rows    = []
        for row in table_rows: new_rows.append(row.split(','))

        column_names = [name[1:name.find(']')] for name in new_rows[0]]

        # If no data was found, return an empty table
        if len(new_rows) > 1:
            catalog_3pi = table.Table(rows=new_rows[1:],names=column_names)
        else:
            catalog_3pi = table.Table()

    # Clean up 3pi's empty cells
    catalog_3pi = make_nan(catalog_3pi)

    # Append '3pi' to column name
    for i in range(len(catalog_3pi.colnames)):
        catalog_3pi[catalog_3pi.colnames[i]].name = catalog_3pi.colnames[i] + '_3pi'

    # Remove duplicates
    catalog_3pi = table.unique(catalog_3pi, keys = 'objID_3pi', keep = 'first')

    print('Found %s objects \n'%len(catalog_3pi))
    return catalog_3pi

def query_SDSS(ra_deg, dec_deg, search_radius = 1.0, timeout=60.0):
    '''
    Function to query the SDSS catalog around the specified coordinates
    The lisf of parameters to query come from:
    https://skyserver.sdss.org/dr12/en/help/browser/browser.aspx#&&history=description+PhotoObjAll+U

    If the function crashes try modifying the amount of whitespace in the SDSS_query, not sure
    why that fixes things. The bug has been reported.

    Parameters
    ---------------
    ra_deg, dec_deg : Coordinates of the object in degrees.
    search_radius   : Search radius in arcminutes
    timeout         : Timeout for the query

    Returns
    -------------
    Astropy table with all SDSS objects
    '''

    # Define Query
    SDSS_query = """SELECT p.objid, -- Object ID
               p.type, -- Type of object, Galaxy vs. Star or other
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
               pz.z, -- Photometric redshift
               pz.zErr, -- Error on the photometric redshift
               s.z, -- Spectroscopic redshift
               s.zErr -- Error on the Spectroscopic redshift
               FROM PhotoObj AS p , dbo.fGetNearbyObjEq(%s, %s, %s) AS n
               LEFT JOIN SpecPhotoAll s on n.objID=s.objID
               LEFT JOIN Photoz AS pz ON pz.objID = n.objID WHERE n.objID = p.objID
               """

    # Query the data, attempt twice
    print('Querying SDSS ...')
    try:
        try:
            catalog_SDSS = SDSS.query_sql(SDSS_query%(np.around(ra_deg, decimals = 5), np.around(dec_deg, decimals = 5), search_radius), timeout=timeout)
        except:
            print('Trying again ...')
            time.sleep(2)
            catalog_SDSS = SDSS.query_sql(SDSS_query%(np.around(ra_deg, decimals = 6), np.around(dec_deg, decimals = 6), search_radius), timeout=timeout)
    # If that also failed try with DR10 instead of DR12
    except:
        print('Trying DR10 ...')
        time.sleep(2)
        catalog_SDSS = SDSS.query_sql(SDSS_query%(ra_deg, dec_deg, search_radius), timeout=timeout, data_release = 10)

    # If there was data, return it
    if catalog_SDSS:
        # If the redshifts are not numbers, fix
        if catalog_SDSS['z'].dtype == bool:
            catalog_SDSS['z'] = table.Column(np.nan * np.ones(len(catalog_SDSS)))
            catalog_SDSS['zErr'] = table.Column(np.nan * np.ones(len(catalog_SDSS)))
        if catalog_SDSS['z1'].dtype == bool:
            catalog_SDSS['z1'] = table.Column(np.nan * np.ones(len(catalog_SDSS)))
            catalog_SDSS['zErr1'] = table.Column(np.nan * np.ones(len(catalog_SDSS)))

        # Clean up SDSS's empty cells
        catalog_SDSS = make_nan(catalog_SDSS)

        # Append 'sdss' to column name
        for i in range(len(catalog_SDSS.colnames)):
            catalog_SDSS[catalog_SDSS.colnames[i]].name = catalog_SDSS.colnames[i] + '_sdss'
    else:
        print('SDSS Broke, no objects found')
        catalog_SDSS = table.Table()

    print('Found %s objects \n'%len(catalog_SDSS))
    return catalog_SDSS

def query_everything(ra_deg, dec_deg, search_radius = 1.0, dust_map = 'SFD'):
    '''
    Query SDSS, 3PI, and the available dust maps, and then merge
    them into one big catalog

    Parameters
    ---------------
    ra_deg, dec_deg : Coordinates of the object in degrees
    search_radius   : Search radius in arcminutes
    dust_map        : 'SF' or 'SFD', to query Schlafy and Finkbeiner 2011
                      or Schlafy, Finkbeiner and Davis 1998
                      set to 'none' to not correct for extinction

    Returns
    ---------------
    One Astropy table with the merged catalog
    '''

    # Query Catalogs
    catalog_3pi  = query_3pi (ra_deg, dec_deg, search_radius)
    catalog_SDSS = query_SDSS(ra_deg, dec_deg, search_radius)

    if len(catalog_3pi) + len(catalog_SDSS) == 0:
        print('No sources found')
        return table.Table()

    # Join Catalogs
    joined_catalog = merge_two_catalogs(catalog_3pi,catalog_SDSS,'raStack_3pi','ra_sdss','decStack_3pi','dec_sdss')

    # If it exists
    if joined_catalog:
        # Get coordinates from 3PI
        if 'raStack_3pi' in joined_catalog.colnames:
            good_3pi_ra  = flot(joined_catalog['raStack_3pi' ])
            good_3pi_dec = flot(joined_catalog['decStack_3pi'])
        else:
            good_3pi_ra  = np.nan * np.ones(len(joined_catalog))
            good_3pi_dec = np.nan * np.ones(len(joined_catalog))

        # Get coordinates from SDSS
        if 'ra_sdss' in joined_catalog.colnames:
            good_sdss_ra  = flot(joined_catalog['ra_sdss' ])
            good_sdss_dec = flot(joined_catalog['dec_sdss'])
        else:
            good_sdss_ra  = np.nan * np.ones(len(joined_catalog))
            good_sdss_dec = np.nan * np.ones(len(joined_catalog))

        # Average them
        better_ra  = np.nanmean([good_3pi_ra , good_sdss_ra ], axis = 0)
        better_dec = np.nanmean([good_3pi_dec, good_sdss_dec], axis = 0)

        # Assign the average coordinates to a matched RA and DEC
        real_ra  = np.isfinite(better_ra)
        real_dec = np.isfinite(better_dec)
        joined_catalog[real_ra]['ra_matched']   = better_ra [real_ra]
        joined_catalog[real_dec]['dec_matched'] = better_dec[real_dec]

        # Append Extinction from Dust Maps
        all_ras  = flot(joined_catalog['ra_matched'])
        all_decs = flot(joined_catalog['dec_matched'])
        extinctions = query_dust(all_ras, all_decs, dust_map)
        joined_catalog.add_column(table.Column(extinctions),  name = 'extinction')

        return joined_catalog
    else:
        print('No sources found')
        return table.Table()

def clean_catalog(data_catalog_in):
    '''
    Clean the 3PI and SDSS catalogs from bad or missing data, returns
    the same catalog, but cleaned up.
    '''

    # Make sure only objects with real coordinates are used
    data_catalog_in = data_catalog_in[np.isfinite(flot(data_catalog_in['ra_matched']))]

    # Remove any objects that don't have any magnitudes in any of these u, g, r, i, z, or y
    try:
        magnitudes_3pi = np.array([np.nansum(i) for i in flot(data_catalog_in['gPSFMag_3pi'  ,'rPSFMag_3pi'  ,'iPSFMag_3pi'  ,'zPSFMag_3pi'  ,'yPSFMag_3pi'  ].to_pandas())])
    except:
        magnitudes_3pi = np.zeros(len(data_catalog_in))

    try:
        magnitudes_sdss = np.array([np.nansum(i) for i in flot(data_catalog_in['psfMag_u_sdss','psfMag_g_sdss','psfMag_r_sdss','psfMag_i_sdss','psfMag_z_sdss'].to_pandas())])
    except:
        magnitudes_sdss = np.zeros(len(data_catalog_in))

    # Crop it
    all_magnitudes   = magnitudes_3pi + magnitudes_sdss
    good_magnitudes  = all_magnitudes != 0
    data_catalog_out = data_catalog_in[good_magnitudes]
    
    return data_catalog_out

def get_catalog(object_name, ra_deg, dec_deg, search_radius = 1.0, dust_map = 'SFD', reimport_catalog = False):
    '''
    Generate Contextual catalog by querying 3PI and SDSS around given coordinates.
    Also get the extinction measurements for the catalog.

    Parameters
    -------------
    object_name      : Name of the object
    ra_deg, dec_deg  : Coordinates to query in degrees
    search_radius    : Search radius in arcminutes
    dust_map         : 'SF' or 'SFD', to query Schlafy and Finkbeiner 2011
                       or Schlafy, Finkbeiner and Davis 1998.
                       set to 'none' to not correct for extinction
    reimport_catalog : If True it will reimport catalog even if
                       it already exists

    Return
    ---------------
    Astropy table with all the objects around the transient
    '''

    # Create catalog folder if it doesn't exist
    if len(glob.glob('catalogs')) == 0:
        os.system("mkdir catalogs")

    # Catalog name
    catalog_name = 'catalogs/%s.cat'%object_name
    catalog_files = glob.glob('catalogs/*%s.cat'%object_name)
    if (len(catalog_files) == 0) | reimport_catalog:
        # Attempt to query everything twice
        try:
            data_catalog_in = query_everything(ra_deg, dec_deg, search_radius, dust_map = dust_map)
        except:
            data_catalog_in = query_everything(ra_deg, dec_deg, search_radius, dust_map = dust_map)
        if data_catalog_in:
            pass
        else:
            return table.Table()

        # Clean Catalog
        data_catalog_out = clean_catalog(data_catalog_in)

        # Write output
        data_catalog_out.write(catalog_name, format='ascii', overwrite=True)
        print('Wrote ', catalog_name)
    # Import existing catalog
    else:
        data_catalog_out = table.Table.read(catalog_files[0], format='ascii', guess=False)

    return data_catalog_out

def get_kron_and_psf(color, catalog, data):
    '''
    Extract the Kron and PSF magnitude from the 3PI or SDSS catalog

    Parameters
    ---------------
    color   : band of the data (u, g, r, i, z, y)
    catalog : 3pi or sdss
    data    : catalog data

    Output
    ---------------
    array of kron and array of psf magnitudes
    '''

    if catalog == '3pi':
        if '%sKronMag_3pi'%color in data.colnames:
            kron_magnitudes = flot(data['%sKronMag_3pi'%color])
            psf_magnitude   = flot(data['%sPSFMag_3pi'%color])
        else:
            return np.nan * np.ones(len(data)), np.nan * np.ones(len(data))
    elif catalog == 'sdss':
        if 'modelMag_%s_sdss'%color in data.colnames:
            kron_magnitudes = flot(data['modelMag_%s_sdss'%color])
            psf_magnitude   = flot(data['psfMag_%s_sdss'%color])
        else:
            return np.nan * np.ones(len(data)), np.nan * np.ones(len(data))
    else:
        print('%s not a catalog, it must be %s or %s'%(catalog, 'sdss', '3pi'))

    return kron_magnitudes, psf_magnitude

def estimate_nature(kron_mag, psf_mag, kron_magnitudes, psf_magnitude, clear_stars, clear_galaxy, color, catalog, neighbors = 20):
    '''
    Estimate the nature of the object given a PSF magnitude and a PSF - Kron magnitude

    Parameters
    ---------------
    kron_mag        : Single Kron magnitude
    psf_mag         : Single PSF magnitude
    kron_magnitudes : List of magnitudes from catalog
    psf_magnitude   : List of magnitudes from catalog
    clear_stars     : Which objects in the catalog are clearly stars 
    clear_galaxy    : Which objects in the catalog are clearly galaxies
    color           : band (u, g, r, i, z, y)
    catalog         : sdss or 3pi
    neighbors       : How many neighbors to consider when classifying the objects

    Output
    ---------------
    Probability from 0 to 1 of the object being a galaxy
    1 = Galaxy; 0 = Star
    
    '''

    # make sure the inputs are floats
    psf_mag  = float(psf_mag)
    kron_mag = float(kron_mag)
    if np.isnan(psf_mag) or np.isnan(kron_mag):
        return np.nan

    # if the magnitude is dimmer than the limit, just return 0.5
    if (catalog == '3pi' ) and (color == 'g') and (psf_mag > 23.64): return 0.5
    if (catalog == '3pi' ) and (color == 'r') and (psf_mag > 23.27): return 0.5
    if (catalog == '3pi' ) and (color == 'i') and (psf_mag > 22.81): return 0.5
    if (catalog == '3pi' ) and (color == 'z') and (psf_mag > 22.44): return 0.5
    if (catalog == '3pi' ) and (color == 'y') and (psf_mag > 22.86): return 0.5
    if (catalog == 'sdss') and (color == 'u') and (psf_mag > 23.42): return 0.5
    if (catalog == 'sdss') and (color == 'g') and (psf_mag > 23.17): return 0.5
    if (catalog == 'sdss') and (color == 'r') and (psf_mag > 22.59): return 0.5
    if (catalog == 'sdss') and (color == 'i') and (psf_mag > 22.04): return 0.5
    if (catalog == 'sdss') and (color == 'z') and (psf_mag > 21.58): return 0.5

    # Get PSF magnitudes
    mag_catalog_star        = psf_magnitude[clear_stars]
    mag_catalog_galaxy      = psf_magnitude[clear_galaxy]
    # Get PSF - Kron magnitudes
    deltamag_catalog        = psf_magnitude               - kron_magnitudes
    deltamag_catalog_star   = psf_magnitude[clear_stars]  - kron_magnitudes[clear_stars]
    deltamag_catalog_galaxy = psf_magnitude[clear_galaxy] - kron_magnitudes[clear_galaxy]
    # Calculate separation in mag-deltamag space
    deltamag = psf_mag - kron_mag
    separation = np.sqrt((deltamag_catalog - deltamag) ** 2 + (psf_magnitude - psf_mag) ** 2)
    # Find the closest 20 objects
    closest = separation.argsort()[:neighbors]
    # Are they galaxies or stars?
    n_stars    = len(np.where(np.array([i in np.where(clear_stars )[0] for i in closest]))[0])
    n_galaxies = len(np.where(np.array([i in np.where(clear_galaxy)[0] for i in closest]))[0])
    # Final fraction
    galaxyness = n_galaxies / (n_stars + n_galaxies)

    return galaxyness

def append_nature(classification_catalog, data_catalog_out, clear_stars, clear_galaxy, neighbors = 20):
    '''
    Add a column to the catalog of data with the estimated nature of each object,
    based on the classification from the CFHLST catalog. 0 is a star, 1 is a 
    galaxy, and everything in between.

    Parameters
    ---------------
    classification_catalog : CFHLST catalog
    data_catalog_out       : Catalog with data to classify
    clear_stars            : Which objects in the catalog are clearly stars 
    clear_galaxy           : Which objects in the catalog are clearly galaxies
    neighbors              : How many neighbors to consider when classifying the objects

    Output
    ---------------
    data_catalog with an extra column
    '''

    # Data to search for
    filters = ['g'  , 'r'   , 'i'   , 'z'   , 'y'  , 'u'   , 'g'   , 'r'   , 'i'   , 'z'   ]
    surveys = ['3pi', '3pi' , '3pi' , '3pi' , '3pi', 'sdss', 'sdss', 'sdss', 'sdss', 'sdss']

    # Estimate Nature in each filter
    print('Calculating Nature ...')
    nature_array = np.array([
    [estimate_nature(*get_kron_and_psf(filters[0], surveys[0], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[0], surveys[0], classification_catalog), clear_stars, clear_galaxy, filters[0], surveys[0], neighbors),
     estimate_nature(*get_kron_and_psf(filters[1], surveys[1], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[1], surveys[1], classification_catalog), clear_stars, clear_galaxy, filters[1], surveys[1], neighbors),
     estimate_nature(*get_kron_and_psf(filters[2], surveys[2], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[2], surveys[2], classification_catalog), clear_stars, clear_galaxy, filters[2], surveys[2], neighbors),
     estimate_nature(*get_kron_and_psf(filters[3], surveys[3], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[3], surveys[3], classification_catalog), clear_stars, clear_galaxy, filters[3], surveys[3], neighbors),
     estimate_nature(*get_kron_and_psf(filters[4], surveys[4], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[4], surveys[4], classification_catalog), clear_stars, clear_galaxy, filters[4], surveys[4], neighbors),
     estimate_nature(*get_kron_and_psf(filters[5], surveys[5], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[5], surveys[5], classification_catalog), clear_stars, clear_galaxy, filters[5], surveys[5], neighbors),
     estimate_nature(*get_kron_and_psf(filters[6], surveys[6], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[6], surveys[6], classification_catalog), clear_stars, clear_galaxy, filters[6], surveys[6], neighbors),
     estimate_nature(*get_kron_and_psf(filters[7], surveys[7], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[7], surveys[7], classification_catalog), clear_stars, clear_galaxy, filters[7], surveys[7], neighbors),
     estimate_nature(*get_kron_and_psf(filters[8], surveys[8], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[8], surveys[8], classification_catalog), clear_stars, clear_galaxy, filters[8], surveys[8], neighbors),
     estimate_nature(*get_kron_and_psf(filters[9], surveys[9], data_catalog_out[k:k+1]), *get_kron_and_psf(filters[9], surveys[9], classification_catalog), clear_stars, clear_galaxy, filters[9], surveys[9], neighbors)] for k in range(len(data_catalog_out))]).T

    # Average Nature (Ignoring 0.5's)
    nature_array[nature_array == 0.5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        average_nature = np.nanmean(nature_array, axis = 0)
    average_nature[np.isnan(average_nature)] = 0.5
    # Names for the nature columns
    column_names = ['nature_%s_%s'%(i, j) for i, j in zip(filters, surveys)]
    output_types = ['float64'] * len(filters)

    # Append nature to the input catalog
    data_catalog_out.add_column(table.Column(data = nature_array[0], name = column_names[0], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[1], name = column_names[1], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[2], name = column_names[2], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[3], name = column_names[3], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[4], name = column_names[4], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[5], name = column_names[5], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[6], name = column_names[6], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[7], name = column_names[7], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[8], name = column_names[8], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data = nature_array[9], name = column_names[9], dtype = 'float64'))
    data_catalog_out.add_column(table.Column(data =  average_nature, name = 'object_nature', dtype = 'float64'))

    # If there are any nan's make them 0.5
    data_catalog_out['object_nature'][np.isnan(flot(data_catalog_out['object_nature']))] = 0.5

    return data_catalog_out

def get_separation(ra_deg, dec_deg, data_catalog):
    '''
    Get the separation between the given RA and DEC
    and the objects in a catalog. Gaia superseeds anything.

    Parameters
    ---------------
    RA_deg, DEC_deg : Coordinates of the object in degrees.
    data_catalog: Catalog with coordinates of objects

    Output
    ---------------
    Separation in arcseconds
    '''

    # Get RA and DEC
    ra_objects  = flot(data_catalog['ra_matched'])
    dec_objects = flot(data_catalog['dec_matched'])

    # superseed with Gaia
    if 'RA_ICRS_gaia' in data_catalog.colnames:
        ra_gaia  = flot(data_catalog['RA_ICRS_gaia'])
        dec_gaia = flot(data_catalog['DE_ICRS_gaia'])

        # Which objects are in Gaia
        in_gaia = np.where(np.isfinite(ra_gaia))

        # Replace values
        ra_objects [in_gaia] = ra_gaia [in_gaia]
        dec_objects[in_gaia] = dec_gaia[in_gaia]

    # Calculate separation to target
    separation = angular_separation(ra_deg, dec_deg, ra_objects, dec_objects)

    return separation

def get_halflight(data_catalog, color, catalog = ''):
    '''
    Get the best estimate of the half light radius for
    either SDSS or 3PI

    Parameters
    ---------------
    data_catalog : Astropy table with data_catalog
    color        : which filter to use
    catalog      : Force to use 3PI or SDSS
    '''

    if catalog in ['3pi', '']:
        # Get Sersic Index from 3PI
        if '%sSerNu_3pi'%color in data_catalog.colnames:
            sersic_n = flot(np.copy(data_catalog['%sSerNu_3pi'%color]))
            # Assume a sersic index of 0.4 if there is none
            sersic_n[np.isnan(sersic_n)] = 0.4
        else:
            # Assume a sersic index of 0.5 for all objects that dont have one
            sersic_n = np.ones(len(data_catalog)) * 0.5

        # Get Sersic normalization, to convert to half light radius
        R_e    = 2.5
        radius = 100000
        b_n    = 1.9992 * sersic_n - 0.3271
        x      = b_n * (radius / R_e) ** (1 / sersic_n)
        R_norm = (R_e / b_n ** sersic_n) * (gammainc(3 * sersic_n, x) / gammainc(2 * sersic_n, x)) * (gamma(3 * sersic_n) / gamma(2 * sersic_n))

        # Normalize Kron radius to half light radius
        if '%sKronRad_3pi'%color in data_catalog.colnames:
            halflight_radius = flot(data_catalog['%sKronRad_3pi'%color]) / R_norm
        else:
            halflight_radius = np.nan * np.ones(len(data_catalog))

    if catalog == '':
        # Replace the radius with sdss if it exists
        if 'petroR50_%s_sdss'%color in data_catalog.colnames:
            SDSS_radii = flot(data_catalog['petroR50_%s_sdss'%color])
            radius_exists = np.where(np.isfinite(SDSS_radii))[0]
            halflight_radius[radius_exists] = SDSS_radii[radius_exists]
    elif catalog == 'sdss':
        if 'petroR50_%s_sdss'%color in data_catalog.colnames:
            halflight_radius = flot(data_catalog['petroR50_%s_sdss'%color])
        else:
            halflight_radius = np.nan * np.ones(len(data_catalog))

    return halflight_radius

def get_sdss_mag(data_catalog, color):
    '''
    For SDSS catalog
    Return Kron magnitude if it exists
    If not, the use PSF magnitude
    '''
    magnitude           = flot(data_catalog['modelMag_%s_sdss'%color])
    bad_data            = np.isnan(magnitude)
    magnitude[bad_data] = flot(data_catalog['psfMag_%s_sdss'%color])[bad_data]

    return magnitude

def get_3pi_mag(data_catalog, color):
    '''
    For 3PI catalog
    Return Kron magnitude if it exists
    If not, the use PSF magnitude.
    '''
    magnitude           = flot(data_catalog['%sKronMag_3pi'%color])
    bad_data            = np.isnan(magnitude)
    magnitude[bad_data] = flot(data_catalog['%sPSFMag_3pi'%color])[bad_data]

    return magnitude

def get_magnitudes(data_catalog, color, catalog, with_limits = False):
    '''
    Return Magnitudes in a given band from a catalog. If with_limits 
    is True, then also return the upper limit for that band.

    Parameters
    ---------------
    data_catalog : Astropy table with data
    color        : Band to get data in
    catalog      : 'sdss' or '3pi'

    Output
    ---------------
    Magnitude, and maybe upper limit

    '''
    if catalog == 'sdss':
        if 'psfMag_%s_sdss'%color in data_catalog.colnames:
            magnitude = get_sdss_mag(data_catalog, color)
        else:
            magnitude = np.nan * np.ones(len(data_catalog))
    elif catalog == '3pi' :
        if '%sPSFMag_3pi'%color in data_catalog.colnames:
            magnitude = get_3pi_mag (data_catalog, color)
        else:
            magnitude = np.nan * np.ones(len(data_catalog))
    else : print('%s not a catalog, it must be %s or %s'%(catalog, 'sdss', '3pi'))

    if with_limits:
        # if the magnitude is dimmer than the limit
        if (catalog == '3pi' ) and (color == 'g') : upper_limit = 23.64
        if (catalog == '3pi' ) and (color == 'r') : upper_limit = 23.27
        if (catalog == '3pi' ) and (color == 'i') : upper_limit = 22.81
        if (catalog == '3pi' ) and (color == 'z') : upper_limit = 22.44
        if (catalog == '3pi' ) and (color == 'y') : upper_limit = 22.86
        if (catalog == 'sdss') and (color == 'u') : upper_limit = 23.42
        if (catalog == 'sdss') and (color == 'g') : upper_limit = 23.17
        if (catalog == 'sdss') and (color == 'r') : upper_limit = 22.59
        if (catalog == 'sdss') and (color == 'i') : upper_limit = 22.04
        if (catalog == 'sdss') and (color == 'z') : upper_limit = 21.58

        return magnitude, upper_limit
    else:
        return magnitude

def calculate_coincidence(separation, size, magnitude):
    '''
    Calculate the chance that a galaxy of size R_h and magnitude M falls
    within a separation R of a transient. The galaxies with the lowest
    chance probability will be selected as the best candidate hosts.

    Parameters
    ---------------
    separation : Separation between the host and transient [Arcseconds]
    size       : Half light radius of the galaxy [Arcseconds]
    Magnitude  : Magnitude of the galaxy

    Output
    ---------------
    P_cc = Probability of chance coincidence
    '''

    # Observed number density of galaxies brighter than magnitude M (From Berger 2010)
    sigma = 10 ** (0.33 * (magnitude - 24) - 2.44) / (0.33 * np.log(10))

    # Effective radius
    R_effective = np.sqrt(np.abs(separation) ** 2 + 4 * np.abs(size) ** 2)

    # Probability of chance coincidence
    chance_coincidence = 1 - np.exp(-np.pi * R_effective ** 2 * sigma)

    return chance_coincidence

def mag_size_coincidence(data_catalog, separation, Pcc_filter = 'i', Pcc_filter_alternative = 'r'):
    '''
    Get the magnitude, size, and chance coincidence of every
    object in the catalog in a specified filter. It will 
    get the average of SDSS and 3PI.

    Parameters
    ---------------
    data_catalog           : Astropy table with data
    separation             : Separation in arcseconds between transient and objects
    Pcc_filter             : The effective magnitude, radius, and Pcc
                             are calculated in this filter.
    Pcc_filter_alternative : If Pcc_filter is not found, use this one
                             as an acceptable alternative.

    Output
    ---------------
    Magniutde, Size, Chance Coincidence
    '''

    if Pcc_filter == 'u' : upper_limit = 23.42
    if Pcc_filter == 'g' : upper_limit = 23.64
    if Pcc_filter == 'r' : upper_limit = 23.27
    if Pcc_filter == 'i' : upper_limit = 22.81
    if Pcc_filter == 'z' : upper_limit = 22.44
    if Pcc_filter == 'y' : upper_limit = 22.86

    # Get half light radii, use SDSS if it exists, if not 3PI
    halflight_radius   = get_halflight(data_catalog, Pcc_filter)
    halflight_radius_2 = get_halflight(data_catalog, Pcc_filter_alternative)
    # If there was no half-light radius, use the r one
    halflight_radius[np.isnan(halflight_radius)] = halflight_radius_2[np.isnan(halflight_radius)]
    # Default halflight radius if none was found
    halflight_radius[np.isnan(halflight_radius)] = 0.7

    # Get magnitudes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        magnitude_sdss   = get_magnitudes(data_catalog, Pcc_filter, 'sdss')
        magnitude_3pi    = get_magnitudes(data_catalog, Pcc_filter, '3pi' )
        hosts_magnitudes = np.nanmean([magnitude_sdss, magnitude_3pi], axis = 0)
        magnitude_sdss_r = get_magnitudes(data_catalog, Pcc_filter_alternative, 'sdss')
        magnitude_3pi_r  = get_magnitudes(data_catalog, Pcc_filter_alternative, '3pi' )
        magnitude_r      = np.nanmean([magnitude_sdss_r, magnitude_3pi_r], axis = 0)
        # If there was no i magnitude, use r
        hosts_magnitudes[np.isnan(hosts_magnitudes)] = magnitude_r[np.isnan(hosts_magnitudes)]
        # If there was no i nor r, use the upper limit
        hosts_magnitudes[np.isnan(hosts_magnitudes)] = upper_limit

    # Calculate Chance Coincidence
    chance_coincidence = calculate_coincidence(separation, halflight_radius, hosts_magnitudes)

    return hosts_magnitudes, halflight_radius, chance_coincidence

# Import CFHLST data to classify objects as star/galaxy
classification_catalog_filename = pkg_resources.resource_filename(__name__, 'classification_catalog.dat')
classification_catalog = table.Table.read(classification_catalog_filename, format='ascii', guess=False)
clear_stars            = np.array(classification_catalog['Nature']) == 0.0
clear_galaxy           = np.array(classification_catalog['Nature']) == 1.0

def catalog_operations(data_catalog_out, ra_deg, dec_deg, Pcc_filter = 'i', Pcc_filter_alternative = 'r', neighbors = 20):
    '''
    Perform basic operations on the catalog related to the transient.

    Parameters
    -------------
    data_catalog_out       : Input astropy table with all objects
    ra_deg, dec_deg        : Coordinates of transient in degrees
    Pcc_filter             : The effective magnitude, radius, and Pcc
                             are calculated in this filter.
    Pcc_filter_alternative : If Pcc_filter is not found, use this one
                             as an acceptable alternative.
    neighbors              : How many neighbors to consider when classifying the objects

    Return
    ---------------
    Astropy table with all the objects around the transient and additional information
    '''

    # Correct for Extinction
    E_BV = flot(data_catalog_out['extinction'])
    R_V = 3.1
    u_correct = np.array([extinction.ccm89(np.array([3594.90]), i * R_V, R_V)[0] for i in E_BV])
    g_correct = np.array([extinction.ccm89(np.array([4640.40]), i * R_V, R_V)[0] for i in E_BV])
    r_correct = np.array([extinction.ccm89(np.array([6122.30]), i * R_V, R_V)[0] for i in E_BV])
    i_correct = np.array([extinction.ccm89(np.array([7439.50]), i * R_V, R_V)[0] for i in E_BV])
    z_correct = np.array([extinction.ccm89(np.array([8897.10]), i * R_V, R_V)[0] for i in E_BV])
    y_correct = np.array([extinction.ccm89(np.array([9603.10]), i * R_V, R_V)[0] for i in E_BV])

    if 'gKronMag_3pi' in data_catalog_out.colnames:
        data_catalog_out['gKronMag_3pi'] = flot(data_catalog_out['gKronMag_3pi']) - g_correct
        data_catalog_out['rKronMag_3pi'] = flot(data_catalog_out['rKronMag_3pi']) - r_correct
        data_catalog_out['iKronMag_3pi'] = flot(data_catalog_out['iKronMag_3pi']) - i_correct
        data_catalog_out['zKronMag_3pi'] = flot(data_catalog_out['zKronMag_3pi']) - z_correct
        data_catalog_out['yKronMag_3pi'] = flot(data_catalog_out['yKronMag_3pi']) - y_correct
        data_catalog_out['gPSFMag_3pi' ] = flot(data_catalog_out['gPSFMag_3pi' ]) - g_correct
        data_catalog_out['rPSFMag_3pi' ] = flot(data_catalog_out['rPSFMag_3pi' ]) - r_correct
        data_catalog_out['iPSFMag_3pi' ] = flot(data_catalog_out['iPSFMag_3pi' ]) - i_correct
        data_catalog_out['zPSFMag_3pi' ] = flot(data_catalog_out['zPSFMag_3pi' ]) - z_correct
        data_catalog_out['yPSFMag_3pi' ] = flot(data_catalog_out['yPSFMag_3pi' ]) - y_correct

    if 'psfMag_u_sdss' in data_catalog_out.colnames:
        data_catalog_out['psfMag_u_sdss'  ] = flot(data_catalog_out['psfMag_u_sdss'  ]) - g_correct
        data_catalog_out['psfMag_g_sdss'  ] = flot(data_catalog_out['psfMag_g_sdss'  ]) - r_correct
        data_catalog_out['psfMag_r_sdss'  ] = flot(data_catalog_out['psfMag_r_sdss'  ]) - i_correct
        data_catalog_out['psfMag_i_sdss'  ] = flot(data_catalog_out['psfMag_i_sdss'  ]) - z_correct
        data_catalog_out['psfMag_z_sdss'  ] = flot(data_catalog_out['psfMag_z_sdss'  ]) - y_correct
        data_catalog_out['modelMag_u_sdss'] = flot(data_catalog_out['modelMag_u_sdss']) - g_correct
        data_catalog_out['modelMag_g_sdss'] = flot(data_catalog_out['modelMag_g_sdss']) - r_correct
        data_catalog_out['modelMag_r_sdss'] = flot(data_catalog_out['modelMag_r_sdss']) - i_correct
        data_catalog_out['modelMag_i_sdss'] = flot(data_catalog_out['modelMag_i_sdss']) - z_correct
        data_catalog_out['modelMag_z_sdss'] = flot(data_catalog_out['modelMag_z_sdss']) - y_correct

    # Append nature to catalog [0 = star, 1 = galaxy]
    data_catalog = append_nature(classification_catalog, data_catalog_out, clear_stars, clear_galaxy, neighbors)

    # Calculate separation to each object
    separation = get_separation(ra_deg, dec_deg, data_catalog)
    data_catalog.add_column(table.Column(data = separation, name = 'separation', dtype = 'float64'))

    # Calculate Probability of Chance Coincidence and Radii
    hosts_magnitudes, halflight_radius, chance_coincidence = mag_size_coincidence(data_catalog, separation, Pcc_filter, Pcc_filter_alternative)
    data_catalog.add_column(table.Column(data = hosts_magnitudes  , name = 'effective_magnitude', dtype = 'float64'))
    data_catalog.add_column(table.Column(data = halflight_radius  , name = 'halflight_radius'   , dtype = 'float64'))
    data_catalog.add_column(table.Column(data = chance_coincidence, name = 'chance_coincidence' , dtype = 'float64'))

    return data_catalog

def get_best_host(data_catalog, star_separation = 1, star_cut = 0.1):
    '''
    From a list of objects, find the best host for a given transient,
    based on the probability of chance coincidence. Rulling out stars.

    Parameters
    -------------
    data_catalog    : Astropy table with data 
    star_separation : A star needs to be this close [in Arcsec]
    star_cut        : maximum allowed probability of being a star

    Return
    ---------------
    host_radius     : The half-light radius of the best host in arcsec
    host_separation : The transient-host separation in arcsec
    host_Pcc        : The probability of chance coincidence for the best host
    host_magnitude  : The magnitude of the best host
    '''
    
    # If it's close and starry, pick that one
    stars = np.where((data_catalog['separation'] < star_separation) & (data_catalog['object_nature'] <= star_cut))[0]
    if len(stars) > 0:
        best_host = stars[0]
    else:
        # Else, pick the one with the lowest Pcc that is a galaxy
        galaxies_catalog = data_catalog[data_catalog['object_nature'] > star_cut]
        best_Pcc  = np.min(galaxies_catalog['chance_coincidence'])
        best_host = np.where(data_catalog['chance_coincidence'] == best_Pcc)[0][0]

    # Properties of the best host
    host_radius     = data_catalog['halflight_radius'   ][best_host]
    host_separation = data_catalog['separation'         ][best_host]
    host_Pcc        = data_catalog['chance_coincidence' ][best_host]
    host_magnitude  = data_catalog['effective_magnitude'][best_host]
    host_nature     = data_catalog['object_nature'      ][best_host]

    if 'z_sdss' in data_catalog.colnames:
        photoz          = data_catalog['z_sdss'][best_host]
        photoz_err      = data_catalog['zErr_sdss'][best_host]
        specz           = data_catalog['z1_sdss'][best_host]
        specz_err       = data_catalog['zErr1_sdss'][best_host]
    else:
        photoz = photoz_err = specz = specz_err = np.nan

    return host_radius, host_separation, host_Pcc, host_magnitude, host_nature, photoz, photoz_err, specz, specz_err

# Extinction
def get_extinction(ra_deg, dec_deg, dust_map = 'SFD'):
    '''
    Get the extinction only in g and r band for the light 
    curve fitting

    Parameters
    ---------------
    ra_deg, dec_deg : Coordinates of the object in degrees.
    dust_map: 'SF' or 'SFD', to query Schlafy and Finkbeiner 2011
               or Schlafy, Finkbeiner and Davis 1998

    Returns
    ---------------
    Floats of g and r extinction correction
    '''
    ebv = query_dust(ra_deg, dec_deg, dust_map = dust_map)
    R_V = 3.1
    g_correct = extinction.ccm89(np.array([4640.40]), ebv * R_V, R_V)[0]
    r_correct = extinction.ccm89(np.array([6122.30]), ebv * R_V, R_V)[0]

    return g_correct, r_correct
