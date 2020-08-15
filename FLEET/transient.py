from astropy.coordinates import SkyCoord
from collections import OrderedDict
from astropy import units as u
from astropy.time import Time
from dateutil import parser
from astropy import table
import numpy as np
import requests
import warnings
import pathlib
import json
import glob
import time
import os

def transient_origin(object_name_in):
    '''
    Based on the transient name, determine if its from a TNS or
    ZTF name, or just another custom name. Modify the name if 
    its from ZTF or TNS by removing the 'AT', 'SN', or 'ZTF' prefix.
    '''

    # If the transient name starts with 'SN', 'AT', or 'ZTF'
    # Its easy to identify the source of the transient
    if object_name_in[:2] in ['AT', 'SN']:
        transient_source = 'TNS'
    elif object_name_in[:3] in ['ZTF']:
        transient_source = 'ZTF'
    # If the transient name starts with e.g. 200, assume TNS
    elif object_name_in[:3] in np.arange(190, 210, 1).astype(str):
        transient_source = 'TNS'
    # If its a 9-character long name that starts with 20, assume ZTF
    elif (object_name_in[:2] == '20') & (len(object_name_in) == 9):
        transient_source = 'ZTF'
    # If none of these match, assume the transient name is not from
    # from TNS nor ZTF
    else:
        transient_source = 'other'

    # Modify object name to standardize it
    if transient_source == 'TNS':
        object_name = object_name_in.replace(' ', '').replace('_', '').replace('-', '').replace('SN2', '2').replace('AT2', '2').replace('SN1', '1').replace('AT1', '1')
    elif transient_source == 'ZTF':
        object_name = object_name_in.replace(' ', '').replace('_', '').replace('-', '').replace('ZTF', '')
    else:
        object_name = object_name_in

    return transient_source, object_name

def convert_coords(ra_in, dec_in):
    '''
    Convert RA and DEC from 'hh:mm:ss', '+dd:mm:ss.s' format
    to degrees. And just return RA and DEC if its already in
    degrees. ra_in, and dec_in must be either strings, floats, 
    integers, or a numpy array.
    '''

    # Make sure RA and DEC are in the same data type
    if type(ra_in) != type(dec_in):
        # But allow integer/float combination
        if (type(ra_in) in [float, int]) & (type(dec_in) in [float, int]):
            pass
        else:
            print('Type of ra and dec must be the same')
            return

    # Convert numpy arrays to single values
    if type(ra_in) == np.ndarray:
        if len(ra_in) == 1:
            ra_in  = str(ra_in[0])
            dec_in = str(dec_in[0])
        else:
            print('ra and dec must be single numbers, arrays not accepted')
            return

    # If its a string it's likely hms dms format
    if type(ra_in) == str:
        # Try converting it to a float first
        # In case it's in degrees
        try:
            ra_deg  = float(ra_in)
            dec_deg = float(dec_in)
        except:
            coord = SkyCoord(ra_in, dec_in, unit=(u.hourangle, u.deg))
            ra_deg, dec_deg = coord.ra.deg, coord.dec.deg

    # If its a float or integer, assume its in degrees already
    elif type(ra_in) in [float, int]:
        ra_deg  = ra_in
        dec_deg = dec_in

    # Format must be string, integer, or float
    else:
        print('Format of ra and dec not recongized')
        return 

    return ra_deg, dec_deg

def get_ZTF_coords(ztf_name):
    '''
    Query the MARS database to get the coordinates of a given ZTF
    objects. If none found return empty '--' coordinates

    Parameters
    -------------
    ztf_name : Name of ZTF object, e.g. 19aaaaabc

    Return
    ---------------
    RA and DEC in degrees
    '''

    # Cone search format
    mars_link = 'https://mars.lco.global/?sort_value=jd&sort_order=desc&objectId=ZTF' + ztf_name + '&format=json'
    # Attempt to query MARS twice
    try:
        print('Quering ZTF for coordinates ...')
        mars_request = requests.get(mars_link).json()
    except:
        print('Trying again ...')
        time.sleep(3)
        mars_request = requests.get(mars_link).json()
    print('\n')
    # Extract object
    candidate = mars_request['results']

    # Extract output data
    if len(candidate) > 0:
        ra_deg, dec_deg = candidate[0]['candidate']['ra'], candidate[0]['candidate']['dec']
        return convert_coords(ra_deg, dec_deg)
    else:
        return '--', '--'

def get_tns_name(ra_deg, dec_deg, acceptance_radius = 3):
    '''
    Do a cone search of a radius acceptance_radius around
    coordinates RA and DEC for the TNS. Return the name of
    the objects found. You must have a file called tns_key.txt
    in your home directory with a TNS api key in order to use
    this function.

    Parameters
    -------------
    ra_deg, dec_deg   : Coordinates of the object in degrees.
    acceptance_radius : Search radius in arcseconds

    Return
    ---------------
    Name of TNS object, returns '--' if not found
    '''

    # Empty Default name
    tns_name = '--'

    # Get Usename and password from tns_key.txt
    key_location = os.path.join(pathlib.Path.home(), 'tns_key.txt')
    api_key      = str(np.genfromtxt(key_location, dtype = 'str'))

    # Build JSON query
    url = "https://wis-tns.weizmann.ac.il/api/get"
    json_list=[("ra",str(ra_deg)), ("dec",str(dec_deg)), ("radius",str(acceptance_radius)), ("units","arcsec"), ("objname",""), ("internal_name","")]
    search_url=url+'/search'
    try:
        # change json_list to json format
        json_file=OrderedDict(json_list)
        # construct the list of (key,value) pairs
        search_data=[('api_key',(None, api_key)),('data',(None,json.dumps(json_file)))]
        # search obj using request module
        response=requests.post(search_url, files=search_data)
        # Get data
        data = response.json()['data']['reply']
        if len(data) > 0:
            tns_name = data[0]['objname']
    except Exception as e:
        print('Error querying TNS : \n'+str(e))

    return tns_name

def get_ztf_name_lightcurve(ra_deg, dec_deg, acceptance_radius = 3, import_ZTF = True, object_name = '--'):
    '''
    Query the MARS database to search for ZTF objects at a given 
    RA and DEC within some acceptance radius. Also download photometry
    if it is available.

    Parameters
    -------------
    ra_deg, dec_deg   : Coordinates of the object in degrees
    acceptance_radius : Search radius in arcseconds
    import_ZTF        : Import ZTF data? If False it will read
                        an existing file
    object_name       : For reading the existing file name

    Return
    ---------------
    Light curve info, Name of ZTF object
    '''

    # Empty Table
    ztf_data = table.Table(names = ['ZTF_MJD', 'ZTF_PSF', 'ZTF_PSFerr', 'ZTF_filter'])
    ztf_name = '--'

    # Read in local ZTF data if it exists
    if (import_ZTF == False) & (object_name != '--'):
        ztf_file = glob.glob('ztf/*%s_ztf.txt'%object_name)
        if len(ztf_file) > 0:
            ztf_data = table.Table(table.Table.read(ztf_file[0], format='ascii', guess=False), dtype = ['float64', 'float64', 'float64', 'S25'])
        else:
            print('No local ZTF data found')
        return ztf_data, object_name
    # If you know there is no ZTF data or just want to avoid downloading it
    elif (import_ZTF == False) & (object_name == '--'):
        print('Did not attempt to download ZTF data')
        return ztf_data, object_name

    # Create folder to store ZTF photometry, if it doesn't exist
    if len(glob.glob('ztf')) == 0:
        os.system("mkdir ztf")

    # Cone search format around given coordinates
    mars_link = 'https://mars.lco.global/?sort_value=jd&sort_order=desc&cone=' + str(ra_deg) + '%2C' + str(dec_deg) + '%2C' + str(acceptance_radius / 3600) + '&format=json'
    # Attempt to query MARS twice
    try:
        print('Quering ZTF ...')
        mars_request = requests.get(mars_link).json()
    except:
        print('Trying again ...')
        time.sleep(3)
        mars_request = requests.get(mars_link).json()
    # Extract object
    candidate = mars_request['results']

    # Extract output data if an object was found
    if len(candidate) > 0:
        output_data = np.array([(target['candidate']['jd'] - 2400000.5, target['candidate']['magpsf'], target['candidate']['sigmapsf'], target['candidate']['filter']) for target in candidate])

        # Convert to Astropy Table
        column_names = ['ZTF_MJD', 'ZTF_PSF', 'ZTF_PSFerr', 'ZTF_filter']
        ztf_data = table.Table(rows = output_data, names = column_names, dtype = ['float64', 'float64', 'float64', 'S25'])

        # Find object name
        ztf_name = candidate[0]['objectId']

        # Save output
        ztf_data.write('ztf/' + ztf_name + '_ztf.txt', format='ascii', overwrite=True)
        print('Saved ', 'ztf/' + ztf_name + '_ztf.txt \n')
    else: print('No ZTF data found \n')

    return ztf_data, ztf_name

def querry_multiple_osc(object_names, skip = 0, block_size = 20, return_table = False, import_OSC = True):
    '''
    Either query multiple objects from the OSC and save the output data to ./osc
    Or query one object and return the photometry table.

    Parameters
    -------------
    object_names : list of names for OSC in a single string 'AT2018aa,AT2018ab,AT2018ac' or a single name
    skip         : Skip a certain number of objects from the begining
    block_size   : Query these many objects at the same time
    return_table : Return the table? When its only one object
    import_OSC   : If false it will just read the existing data for a single object

    Return
    ---------------
    Saves the output to a text file in ./osc
    Returns a list of all the failed names, which can be fed back into
    the same function.
    If return_table == True then it returns an astropy table with photometry
    '''

    # Empty Default Table
    osc_data = table.Table(names = ['OSC_MJD', 'OSC_Mag', 'OSC_Magerr', 'OSC_filter', 'OSC_UL', 'OSC_telescope'])

    # Read existing file
    if import_OSC == False:
        osc_file = glob.glob('osc/*%s_osc.txt'%object_names)
        if len(osc_file) > 0:
            osc_data = table.Table(table.Table.read(osc_file[0], format='ascii', guess=False), dtype = ['float64', 'float64', 'float64', 'S25', 'S25', 'S25'])
        else:
            print('No local OSC data found')
        return osc_data

    # Empty array
    empties = ['',' ','None','--', '-', b'',b' ',b'None',b'--', b'-', None, np.nan, 'nan', b'nan', '0', 0]

    # Split object names into individual names
    object_names = object_names.replace(' ', '').replace('-', '').replace('_', '')
    object_names = object_names.replace('ASASSN', 'ASASSN-')
    to_query     = np.array([str(i) for i in object_names.split(',')][skip:])

    # Select only the objects that are not in ZTF
    good = [i[:3] != 'ZTF' for i in to_query]

    # Create folder to store photometry, if it doesn't exist
    if len(glob.glob('osc')) == 0:
        os.system("mkdir osc")

    steps = int(np.ceil(len(to_query[good]) / 1.0 / block_size))
    not_in_osc = []
    for step in range(steps):
        # Create OSC link for query
        osc_link    = 'https://astrocats.space/api/'
        for j in range(len(to_query[good][step*block_size:(step+1)*block_size])):
            osc_link += to_query[good][step*block_size:(step+1)*block_size][j]
            if j < len(to_query[good][step*block_size:(step+1)*block_size]) - 1:
                osc_link += '+'
        osc_link += '/photometry/time+magnitude+e_magnitude+band+upperlimit+telescope'

        # Request Data
        print('Querying OSC ...')
        osc_request = requests.get(osc_link).json()

        # Get Photometry from each object
        query_step = to_query[good][step*block_size:(step+1)*block_size]
        for k in range(len(query_step)):
            try:
                # Get the data and remove any lines with no time
                raw_data = np.array(osc_request[query_step[k]]['photometry'])

                # Make empties nans
                OSC_MJD       = np.array([np.nan if i in empties else i for i in raw_data.T[0]])
                OSC_Mag       = np.array([np.nan if i in empties else i for i in raw_data.T[1]])
                OSC_Magerr    = np.array([np.nan if i in empties else i for i in raw_data.T[2]])
                OSC_filter    = np.array([  '--' if i in empties else i for i in raw_data.T[3]])
                OSC_UL        = np.array([  '--' if i in empties else i for i in raw_data.T[4]])
                OSC_telescope = np.array([  '--' if i in empties else i for i in raw_data.T[5]])
                total_data = np.array([OSC_MJD, OSC_Mag, OSC_Magerr, OSC_filter, OSC_UL, OSC_telescope]).T

                # Convert to Astropy Table
                column_names = ['OSC_MJD', 'OSC_Mag', 'OSC_Magerr', 'OSC_filter', 'OSC_UL', 'OSC_telescope']
                if len(raw_data) == 1:
                    osc_data = table.Table(data = total_data, names = column_names, dtype = ['float64', 'float64', 'float64', 'S25', 'S25', 'S25'])
                else:
                    osc_data = table.Table(rows = total_data, names = column_names, dtype = ['float64', 'float64', 'float64', 'S25', 'S25', 'S25'])

                # Save output
                query_step[k] = query_step[k].replace('-', '') # For ASASSN names
                osc_data.write('osc/' + query_step[k] + '_osc.txt', format='ascii', overwrite=True)
                print('Saved ' + query_step[k] + '_osc.txt \n')
                if return_table : return osc_data

            except Exception as e:
                print(query_step[k] + ' Not yet in the OSC : ' + str(e) + '\n')
                not_in_osc.append(query_step[k])
                if return_table : return osc_data

    # Return names that failed
    names_not_found = ','.join(not_in_osc)
    return names_not_found

def get_tns_coords(tns_name):
    '''
    Get the ra and dec of an object from the TNS given its name.
    A tns key is required to run this function, which must be
    in your home directory.

    Parameters
    -------------
    tns_name : Name of the object i.e. 2016iet

    Return
    ---------------
    ra_deg, dec_deg in degrees floats. If not found then '--'
    '''

    # Get TNS key
    key_location = os.path.join(pathlib.Path.home(), 'tns_key.txt')
    api_key      = str(np.genfromtxt(key_location, dtype = 'str'))

    # Build JSON query
    url       = "https://wis-tns.weizmann.ac.il/api/get"
    json_list = [("objname",tns_name), ("photometry","0")]
    get_url   = os.path.join(url, 'object')                 

    try:
        # change json_list to json format
        json_file=OrderedDict(json_list)                    
        # construct the list of (key,value) pairs
        get_data=[('api_key',(None, api_key)),              
                     ('data',(None,json.dumps(json_file)))] 
        # search obj using request module
        warnings.filterwarnings("ignore")
        print('Querying TNS for coordinates ... \n')
        response=requests.post(get_url, files=get_data, verify=False)     
        # Get data
        data = response.json()['data']['reply']

        # If there was an object found, return its info
        if len(data) == 1:
            return '--', '--'
        else:
            # Convert Date
            TNS_date_object  = parser.parse(data['discoverydate'])
            TNS_astropy_date = Time(TNS_date_object)
            TNS_date_MJD     = TNS_astropy_date.mjd

            # Extract data
            ra_in  = data['radeg']
            dec_in = data['decdeg']
            ra_deg, dec_deg = convert_coords(ra_in, dec_in)
            return ra_deg, dec_deg

    except Exception as e:
        print('Error Quering TNS : \n'+str(e))
        return '--', '--'

def get_transient_info(object_name_in = '', ra_in = '', dec_in = '', acceptance_radius = 3, import_ZTF = True, import_OSC = True, import_lightcurve = True):
    '''
    Get the coordaintes and name for a transient. Either the coordinates
    and/or the name must be specified. The function will search for the
    missing values in ZTF or TNS. Photometry from ZTF and OSC will also be
    downloaded if available.

    Parameters
    -------------
    object_name_in    : Name of the object
    ra_in, dec_in     : Coordinates of the object
    acceptance_radius : Search radius in arcseconds
    import_ZTF        : Import ZTF data or read local file?
    import_OSC        : Import OSC data or read local file?
    import_lightcurve : Only works when coordinates AND name are 
                        specified. If False it will return empty
                        ztf_data and osc_data tables

    Return
    ---------------
    ra_deg, dec_deg  : Coordinates of the object in degress
    transient_source : Source origin of the transient (TNS, ZTF, Other)
    object_name      : Standard name of the transient
    ztf_data         : Astropy Table with or without data
    ztf_name         : Name of transient in ZTF, if it exists
    tns_name         : Name of transient in OSC, if it exists
    osc_data         : Astropy Table with or without data
    '''

    # Empty variables
    osc_data = table.Table(names = ['OSC_MJD', 'OSC_Mag', 'OSC_Magerr', 'OSC_filter', 'OSC_UL', 'OSC_telescope'])
    ztf_data = table.Table(names = ['ZTF_MJD', 'ZTF_PSF', 'ZTF_PSFerr', 'ZTF_filter'])

    ###### If there are no coordinates, but only a name ######

    if (ra_in == '') & (dec_in == '') & (object_name_in != ''):
        # Get transient source and modified name
        transient_source, object_name = transient_origin(object_name_in)

        # Get coordiantes from the object name
        if transient_source == 'ZTF':
            ra_deg, dec_deg = get_ZTF_coords(object_name)
            if ra_deg != '--':
                # Query ZTF for light curve information
                ztf_data, ztf_name = get_ztf_name_lightcurve(ra_deg, dec_deg, acceptance_radius, import_ZTF, object_name)
                # Search for corresponding TNS object
                tns_name = get_tns_name(ra_deg, dec_deg, acceptance_radius)
                if tns_name != '--':
                    # Query OSC for light curve information
                    osc_data = querry_multiple_osc(tns_name, return_table = True, import_OSC = import_OSC)
            else:
                print('ZTF object %s not found'%object_name)
                return ['--'] * 8
        elif transient_source == 'TNS':
            ra_deg, dec_deg = get_tns_coords(object_name)
            tns_name = object_name
            # Query OSC for light curve information
            osc_data = querry_multiple_osc(object_name, return_table = True, import_OSC = import_OSC)
            if ra_deg != '--':
                # Query ZTF for light curve information
                ztf_data, ztf_name = get_ztf_name_lightcurve(ra_deg, dec_deg, acceptance_radius, import_ZTF)
            else:
                print('TNS object %s not found'%object_name)
                return ['--'] * 8
        else:
            print('object_name %s not recognized, ra and dec required'%object_name)
            return ['--'] * 8

    ###### If there is no name, but only coordinates ######

    elif (ra_in != '') & (dec_in != '') & (object_name_in == ''):
        # Standarize coordiante format
        try:
            ra_deg, dec_deg = convert_coords(ra_in, dec_in)
        except:
            print('Invalid ra, dec input : %s, %s'%(ra_in, dec_in))
            return ['--'] * 8

        # Search the TNS
        tns_name = get_tns_name(ra_deg, dec_deg, acceptance_radius)
        if tns_name != '--':
            # Query OSC for light curve information
            osc_data = querry_multiple_osc(tns_name, return_table = True, import_OSC = import_OSC)
            object_name_in = tns_name
            transient_source, object_name = transient_origin(object_name_in)

        # Search ZTF
        ztf_data, ztf_name = get_ztf_name_lightcurve(ra_deg, dec_deg, acceptance_radius, import_ZTF)
        if tns_name == '--':
            object_name_in = ztf_name
            transient_source, object_name = transient_origin(object_name_in)

        if (ztf_name == '--') & (tns_name == '--'):
            print('object not found for ra, dec = %s, %s. Name required'%(ra_deg, dec_deg))
            return ['--'] * 8

    ###### if there are coordaintes, and a name ######

    elif (ra_in != '') & (dec_in != '') & (object_name_in != ''):
        # Standarize coordiante format
        try:
            ra_deg, dec_deg = convert_coords(ra_in, dec_in)
        except:
            print('Invalid ra, dec input : %s, %s'%(ra_in, dec_in))
            return ['--'] * 8

        # Get transient source and modified name
        transient_source, object_name = transient_origin(object_name_in)
        ztf_name = '--'
        tns_name = '--'

        # Search the TNS
        if (transient_source != 'ZTF') & import_lightcurve:
            # Query OSC for light curve information
            osc_data = querry_multiple_osc(object_name, return_table = True, import_OSC = import_OSC)
            # Query ZTF
            ztf_data, ztf_name = get_ztf_name_lightcurve(ra_deg, dec_deg, acceptance_radius, import_ZTF)

        # Search for corresponding TNS source from ZTF
        if (transient_source == 'ZTF') & import_lightcurve:
            tns_name = get_tns_name(ra_deg, dec_deg, acceptance_radius)
            if tns_name != '--':
                # Query OSC for light curve information
                osc_data = querry_multiple_osc(tns_name, return_table = True, import_OSC = import_OSC)
            ztf_data, ztf_name = get_ztf_name_lightcurve(ra_deg, dec_deg, acceptance_radius, import_ZTF, object_name)
        
        # ZTF name
        if (transient_source == 'ZTF') & (ztf_name == '--'):
            ztf_name = object_name
        # TNS name
        if (transient_source == 'TNS') & (tns_name == '--'):
            tns_name = object_name

    return ra_deg, dec_deg, transient_source, object_name, ztf_data, ztf_name, tns_name, osc_data

def ignore_data(object_name, output_table):
    '''
    Search the ./ignore folder to find photomety that needs to be ignored.
    Take the input astropy table and modify its 'Ignore' column to specify
    which datapoints will be ignored

    Parameters
    ---------------
    object_name  : Name of the object
    output_table : Data with photometry

    Output
    ---------------
    Same output_table but with modified Ignore column
    '''

    # Generate folder called ignore if it does not exist
    if len(glob.glob('ignore')) == 0:
        os.system("mkdir ignore")

    # Search for files
    ignore_file = glob.glob('ignore/*%s.txt'%object_name)

    if len(ignore_file) != 0:
        # Ignore data bounded by these ranges
        ignore_range = np.genfromtxt(ignore_file[0])

        # If there's only one line
        if ignore_range.shape == (4,):
            min_MJD, max_MJD, min_mag, max_mag = ignore_range

            data_ignored = (output_table['MJD'] > min_MJD) & (output_table['MJD'] < max_MJD) & (output_table['Mag'] > min_mag) & (output_table['Mag'] < max_mag)
            output_table['Ignore'][data_ignored] = 'True'

        # If there are multiple lines
        else:
            for i in range(len(ignore_range)):
                min_MJD, max_MJD, min_mag, max_mag = ignore_range[i]

                data_ignored = (output_table['MJD'] > min_MJD) & (output_table['MJD'] < max_MJD) & (output_table['Mag'] > min_mag) & (output_table['Mag'] < max_mag)
                output_table['Ignore'][data_ignored] = 'True'

    return output_table

def import_my_data(object_name, import_local = True):
    '''
    Search the photometry/ folder to find photomety taken
    with your own instrument.

    Parameters
    ---------------
    object_name       : Name of the object
    import_local : Try to find local photometry?

    Output
    ---------------
    Astropy table with your data
    '''

    # Empty default table
    my_data = table.Table(names = ['my_MJD', 'my_Mag', 'my_Err', 'my_Filter', 'my_Telescope'])

    if import_local == False:
        print('Did not attempt to search for local data \n')
        return my_data
    else:
        print('Looking for local data in ./photometry/')
    # Search for files
    my_file = glob.glob('photometry/*%s.txt'%object_name)

    # If there is no photometry return False and '--'
    if len(my_file) == 0:
        print('Local data for %s not found \n'%object_name)
        return my_data

    # Import data
    if len(my_file) > 0:
        print('Found local data %s'%my_file[0] + '\n')
        my_data = table.Table(table.Table.read(my_file[0], format='ascii', guess=False), dtype = ['float64', 'float64', 'float64', 'S25', 'S25'], names = ['my_MJD', 'my_Mag', 'my_Err', 'my_Filter', 'my_Telescope'])

    return my_data
 
def generate_lightcurve(ztf_data, osc_data, object_name = '--', ztf_name = '--', tns_name = '--', import_lightcurve = True, import_local = True):
    '''
    Gather all the available photometry and merge it into one astropy table.

    Parameters
    -------------
    ztf_data          : Astropy Table with or without ZTF data from get_transient_info()
    osc_data          : Astropy Table with or without OSC data from get_transient_info()
    object_name       : Standard name of the transient
    ztf_name          : Equivalent ZTF name of the transient
    tns_name          : Equivalent TNS name of the transient
    import_lightcurve : If False it will just read the existing light curve file and
                        ignore ztf_data and osc_data
    import_local      : Import local photometry

    Return
    ---------------
    output_table  : An astropy table with all the photometry
    '''

    # Read existing light curve data
    if import_lightcurve == False:
        lightcurve_name = 'lightcurves/*%s.txt'%object_name
        exists = glob.glob(lightcurve_name)
        if len(exists) == 0:
            print('lightcurve data required, import_lightcurve can not be False')
            return
        else:
            try:
                output_types=['float64','float64','float64','S25','S25','S25','S25','S25','S25']
                output_table = table.Table(table.Table.read(exists[0], format='ascii', guess=False), dtype = output_types)
            except:
                output_types=['float64','float64','float64','S25','S25','S25','S25','S25','S25','S25','S25','S25']
                output_table = table.Table(table.Table.read(exists[0], format='ascii', guess=False), dtype = output_types)

        return output_table

    # Combined Table Parameters
    output_names = ['MJD'    , 'Mag'    , 'MagErr' , 'Telescope', 'Filter', 'Source', 'UL' , 'Name', 'Ignore']
    output_types = ['float64', 'float64', 'float64', 'S25'      , 'S25'   , 'S25'   , 'S25', 'S25' , 'S25'   ]

    # Import my data if it exists
    my_data = import_my_data(object_name, import_local)

    # Length of tables
    osc_len, my_len, ztf_len = len(osc_data), len(my_data), len(ztf_data)

    flot = lambda x : np.array(x).astype(float)
    if (ztf_name != '--') & ('ZTF' not in ztf_name): ztf_name = 'ZTF' + ztf_name
    # Concatenate and format all tables
    all_times      = np.concatenate([flot(osc_data['OSC_MJD'])           , flot(my_data['my_MJD'])             , flot(ztf_data['ZTF_MJD'])        ])
    all_mags       = np.concatenate([flot(osc_data['OSC_Mag'])           , flot(my_data['my_Mag'])             , flot(ztf_data['ZTF_PSF'])        ])
    all_magerrs    = np.concatenate([flot(osc_data['OSC_Magerr'])        , flot(my_data['my_Err'])             , flot(ztf_data['ZTF_PSFerr'])     ])
    all_telescopes = np.concatenate([np.array(osc_data['OSC_telescope']) , np.array(my_data['my_Telescope'])   , ['ZTF'] * ztf_len                ]).astype('str')
    all_filters    = np.concatenate([np.array(osc_data['OSC_filter'])    , np.array(my_data['my_Filter'])      , np.array(ztf_data['ZTF_filter']) ]).astype('str')
    all_sources    = np.concatenate([['OSC'] * osc_len                   , ['Local'] * my_len                  , ['ZTF'] * ztf_len                ]).astype('str')
    upperlims      = np.concatenate([np.array(osc_data['OSC_UL'])        , np.array(my_data['my_Err'])         , np.array(ztf_data['ZTF_PSFerr']) ]).astype('str')
    all_upperlims  = np.array([i in [True, -1.0, 'True', '-1', '-1.0', '-1.', b'True', b'-1', b'-1.0', b'-1.'] for i in upperlims])
    all_names      = np.concatenate([[tns_name] * osc_len                , [object_name] * my_len              , [ztf_name] * ztf_len             ]).astype('str')
    all_ignores    = np.concatenate([['False'] * osc_len                 , ['False'] * my_len                  , ['False'] * ztf_len              ])

    # Create output Table
    output_columns  = [all_times , all_mags  , all_magerrs , all_telescopes , all_filters , all_sources , all_upperlims, all_names, all_ignores]
    output_table_in = table.Table(data = np.array(output_columns).T, names = output_names, dtype = output_types)

    # Remove bad lines with no useful data
    real = np.where(np.isfinite(output_table_in['Mag']) & np.isfinite(output_table_in['MJD']))[0]
    output_table = output_table_in[real]

    # Create folder to store lightcurve data, if it doesn't exist
    if len(glob.glob('lightcurves')) == 0:
        os.system("mkdir lightcurves")
    output_table.write('lightcurves/' + object_name + '.txt', format='ascii', overwrite=True)
    print('Wrote ', 'lightcurves/' + object_name + '.txt')

    return output_table
