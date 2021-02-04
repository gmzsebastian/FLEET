from matplotlib.ticker import ScalarFormatter
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from FLEET.lightcurve import linex
from FLEET.catalog import get_kron_and_psf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.dates as md
from astropy.time import Time
from bs4 import BeautifulSoup
from PyAstronomy import pyasl
from astropy import table
from astral import Astral
from io import BytesIO
from PIL import Image
import numpy as np
import warnings
import datetime
import requests
import urllib
import ephem
import glob
import sys
import os

def redshift_magnitude(magnitude, redshift, sigma = 0):
    """
    This function will calculate the absolute magnitude of an object 
    given it's apparent magntiude and redshift. Taken from 
    http://www.astro.ufl.edu/~guzman/ast7939/projects/project01.html
    And not applying a K, extinction, nor color correction.

    Parameters
    ---------------
    magnitude : Apparent magnitude of the source
    redshift  : Redshift of the source
    sigma     : Errorbar on the redshift

    Output
    ---------------
    Absolute Magnitude
    """

    # Luminosity Distance in parsecs 
    D_L = Distance(z = redshift).pc

    # Distance Modulus
    DM = 5 * np.log10(D_L / 10)

    # Absolute Magnitude
    M = magnitude - DM

    if sigma == 0:
        return M, ''
    else:
        D_Lplus = Distance(z = redshift + sigma).pc
        if redshift - sigma >= 0:
            D_Lminus = Distance(z = redshift - sigma).pc
        else:
            D_Lminus = Distance(z = 0).pc

        DMplus = 5 * np.log10(D_Lplus / 10)
        DMminus = 5 * np.log10(D_Lminus / 10)

        Mplus = DMplus - DM
        Mminus = DM - DMminus

        error = np.abs(0.5 * (Mplus + Mminus))

    return M, error

def plot_colors(all_bands):
    '''
    Generate a list of matplotlib readable colors based on
    the input bands. Set the color to black if the filter
    is not found.
    '''

    # Make Color Array
    all_colors = np.copy(all_bands.astype('S16')).astype('str')

    # Select region for each filter
    u_colors      = ((all_colors == "u") | (all_colors == "u'"))
    g_colors      = ((all_colors == "g") | (all_colors == "g'"))
    r_colors      = ((all_colors == 'r') | (all_colors == "r'") | (all_colors == 'R'))
    i_colors      = ((all_colors == 'i') | (all_colors == "i'") | (all_colors == 'I'))
    z_colors      = ((all_colors == "z") | (all_colors == "z'")) 
    V_colors      =  all_colors  == 'V'     
    B_colors      =  all_colors  == 'B'     
    C_colors      =  all_colors  == 'C'     
    w_colors      =  all_colors  == 'w'     
    G_colors      =  all_colors  == 'G'     
    orange_colors =  all_colors  == 'orange'
    cyan_colors   =  all_colors  == 'cyan'  
    Clear_colors  =  all_colors  == 'Clear' 

    # Data that doesn't fall under any other filter
    k_colors = ((u_colors+g_colors+r_colors+i_colors+z_colors+V_colors+B_colors+C_colors+w_colors+G_colors+orange_colors+cyan_colors+Clear_colors) == False)

    # Change Filters to Python colors
    all_colors[u_colors]      = 'navy'
    all_colors[g_colors]      = 'g'
    all_colors[r_colors]      = 'r'
    all_colors[i_colors]      = 'maroon'
    all_colors[z_colors]      = 'saddlebrown'
    all_colors[V_colors]      = 'lawngreen'
    all_colors[B_colors]      = 'darkcyan'
    all_colors[C_colors]      = 'c'
    all_colors[w_colors]      = 'goldenrod'
    all_colors[G_colors]      = 'orange'
    all_colors[orange_colors] = 'gold'
    all_colors[cyan_colors]   = 'blue'
    all_colors[Clear_colors]  = 'magenta'
    all_colors[k_colors]      = 'k'

    colors = np.array(np.ndarray.tolist(all_colors))

    return colors

flot = lambda x : np.array(x).astype(float) 

def plot_1a(time, y_0 = 0.95717, m = 0.01817, t_0 = 4.63783, g_0 = -0.91248, sigma_0 = 11.18299, tau = -17.77671, theta_0 = 19.384):
    '''
    Generate a type 1a supernova light curve from an analytical function:
    Slope + Gaussian + Exponential Rise
    As described in https://arxiv.org/pdf/1707.07614.pdf

    g_model = plot_1a(timex, 2.02715, 0.0132083, -1.24344, -1.582490, 10.1705, -21.8477, 15.3465)
    r_model = plot_1a(timex, 0.92540, 0.0182386,  3.95539, -0.884576, 11.0422, -17.5522, 18.7745)
    i_model = plot_1a(timex, 0.79161, 0.0179976,  7.63109, -0.723820, 11.3517, -16.4830, 18.0652)

    Parameters
    --------------
    time    : Array of times
    y_0     : Magntiude offset
    m       : Slope of late time decay
    t_0     : Time of maximum for gaussian
    g_0     : Height of gaussian
    sigma_0 : Width of gaussian
    tau     : Explosion time
    theta_0 : Rate of exponential rise

    Output
    --------------
    Magnitudes of supernova light curve
    '''

    # Components
    slope       = y_0 + m * (time - t_0)
    gaussian    = g_0 * np.exp(-(time - t_0) ** 2 / (2 * sigma_0 ** 2))
    exponential = 1 - np.exp((tau - time) / theta_0)

    # Result
    magnitude = (slope + gaussian) / exponential
    return time - time[np.argmin(magnitude)], magnitude - magnitude[np.argmin(magnitude)]

# Model Type Ia
time_1a_model_r, magnitude_1a_model_r = plot_1a(np.linspace(-15, 120, 500), 0.92540, 0.0182386,  3.95539, -0.884576, 11.0422, -17.5522, 18.7745)

def quick_plot(object_name, ra_deg, dec_deg, output_table, first_mjd, bright_mjd, g_correct = 0, r_correct = 0):
    '''
    Create a diagnosis plot with the light curve and field image of a transient
    Parameters
    ---------------
    object_name      : Name of the transient
    ra_deg, dec_deg  : Coordiantes of the transient in degrees
    output_table     : Data with photometry
    first_mjd        : First MJD in either g or r from fit_linex()
    bright_mjd       : Brightest MJD in either g or r from fit_linex()
    g_correct        : extinction value in g band
    r_correct        : extinction value in r band
    Output
    ---------------
    Saves an output to plots/ directory
    '''

    # Correct for extinction
    output_table_correct = extinct(output_table, g_correct, r_correct)

    # Create folder to store images, if it doesn't exist
    if len(glob.glob('plots')) == 0:
        os.system("mkdir plots")

    # Plot Lightcurve
    plot_lightcurve(1, 1, 1, output_table_correct, first_mjd, bright_mjd, full_range = True)

    plt.savefig('plots/%s_quick.pdf'%object_name, bbox_inches = 'tight')
    plt.clf(); plt.close('all')

def plot_lightcurve(sub_y, sub_x, sub_n, output_table, first_mjd, bright_mjd, red_amplitude = np.nan, red_amplitude2 = np.nan, red_offset = np.nan, red_magnitude = np.nan, green_amplitude = np.nan, green_amplitude2 = np.nan, green_offset = np.nan, green_magnitude = np.nan, subtract_phase = 0, add_phase = 0, plot_model = True, full_range = False, plot_comparison = True):
    '''
    Plot the light curve and model for a given transient

    Parameters
    ---------------
    sub_y, sub_x, sub_n  : Subplot parameters (rows, columns, plot number)
    output_table     : Data with photometry
    bright_mjd       : Brightest MJD in either g or r from fit_linex()
    first_mjd        : First MJD in either g or r from fit_linex()

    red_amplitude    : Parameters from fit_linex()
    red_amplitude2   :
    red_offset       :
    red_magnitude    :
    green_amplitude  :
    green_amplitude2 :
    green_offset     :
    green_magnitude  :

    subtract_phase   : Days to subtract from minimum x limit
    add_phase        : Days to add to maximum x limit
    plot_model       : Plot the model
    full_range       : Plot the entire photometry (True), or just
                       the relevant portion (False)
    plot_comparison  : Plot a Ia light curve?
    '''

    # Extract all data
    all_magnitudes  =     flot(output_table['Mag'      ])
    all_times       =     flot(output_table['MJD'      ])
    all_sigmas      =     flot(output_table['MagErr'   ])
    all_filters     = np.array(output_table['Filter'   ]).astype('str')
    all_sources     = np.array(output_table['Source'   ]).astype('str')
    all_upperlimits = np.array(output_table['UL'       ]).astype('str')
    all_ignores     = np.array(output_table['Ignore'   ]).astype('str')
    if full_range:
        all_phases  = all_times
    else:
        all_phases  = all_times - bright_mjd

    # Which are Upper limits and ignored data
    upper_limit   = np.where((all_upperlimits == 'True' ) & (all_ignores == 'False'))[0]
    detection     = np.where((all_upperlimits == 'False') & (all_ignores == 'False'))[0]
    # Ignored
    upper_ignore  = np.where((all_upperlimits == 'True' ) & (all_ignores == 'True'))[0]
    detect_ignore = np.where((all_upperlimits == 'False') & (all_ignores == 'True'))[0]

    # Select Plot Colors
    all_colors = plot_colors(all_filters)

    # Select groups of Data (detectiond and upper limits)
    is_det_ZTF   = np.where(all_sources[detection] == 'ZTF')[0]
    is_det_Local  = np.where(all_sources[detection] == 'Local')[0]
    is_det_other = np.where([i not in ['ZTF', 'Local'] for i in all_sources[detection]])[0]

    # Set Magnitude Limits to ± 0.5 the limits
    real_magnitudes = all_magnitudes[np.isfinite(all_magnitudes)]
    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_yaxis()
    plt.ylim(np.nanmax(real_magnitudes) + 0.5, np.nanmin(real_magnitudes) - 0.5)

    # Set Phase Limits
    # Minimum
    brightest_phase = bright_mjd - first_mjd
    if brightest_phase > 100:
        MJD_minimum = bright_mjd - 100
    else:
        MJD_minimum  = first_mjd - 5
    phase_minimum  = MJD_minimum - bright_mjd

    # Maximum
    latest_phase = np.nanmax(all_phases[detection])
    latest_mjd   = np.nanmax(all_times[detection])
    if latest_phase >= 30:
        MJD_maximum = latest_mjd + 5
    else:
        MJD_maximum = latest_mjd + 35
    phase_maximum  = MJD_maximum - bright_mjd

    # Include offsets if specified
    if full_range:
        plt.xlim(np.nanmin(all_times), np.nanmax(all_times))
    else:
        plt.xlim(phase_minimum - subtract_phase, phase_maximum + add_phase)

    # Plot Data (detections)
    plt.errorbar(all_phases[detection][is_det_ZTF]  , all_magnitudes[detection][is_det_ZTF]  , all_sigmas[detection][is_det_ZTF]  , ecolor = all_colors[detection][is_det_ZTF].astype('str')  ,     fmt = 'd'   , alpha = 0.8, ms = 0)
    plt.errorbar(all_phases[detection][is_det_Local] , all_magnitudes[detection][is_det_Local] , all_sigmas[detection][is_det_Local] , ecolor = all_colors[detection][is_det_Local].astype('str') ,     fmt = '*'   , alpha = 0.8, ms = 0)
    plt.errorbar(all_phases[detection][is_det_other], all_magnitudes[detection][is_det_other], all_sigmas[detection][is_det_other], ecolor = all_colors[detection][is_det_other].astype('str'),     fmt = '.'   , alpha = 0.8, ms = 0)
    plt.scatter (all_phases[detection][is_det_ZTF]  , all_magnitudes[detection][is_det_ZTF]  ,                                       color = all_colors[detection][is_det_ZTF].astype('str')  ,  marker = 'd'   , alpha = 0.8, s = 90)
    plt.scatter (all_phases[detection][is_det_Local] , all_magnitudes[detection][is_det_Local] ,                                       color = all_colors[detection][is_det_Local].astype('str') ,  marker = '*'   , alpha = 0.8, s = 90)
    plt.scatter (all_phases[detection][is_det_other], all_magnitudes[detection][is_det_other],                                       color = all_colors[detection][is_det_other].astype('str'),  marker = '.'   , alpha = 0.8, s = 90)
    # Plot Data (Upper limits)
    plt.scatter (all_phases[upper_limit], all_magnitudes[upper_limit], color = all_colors[upper_limit].astype('str'), marker = 'v', alpha = 0.5, s = 90)

    ### Plot Ignored Data
    ignore_alpha = 0.15
    # Select groups of Data (detectiond and upper limits)
    was_det_ZTF   = np.where(all_sources[detect_ignore] == 'ZTF')[0]
    was_det_Local  = np.where(all_sources[detect_ignore] == 'Local')[0]
    was_det_other = np.where([i not in ['ZTF', 'Local'] for i in all_sources[detect_ignore]])[0]
    # (detections)
    plt.errorbar(all_phases[detect_ignore][was_det_ZTF]  , all_magnitudes[detect_ignore][was_det_ZTF]  , all_sigmas[detect_ignore][was_det_ZTF]  , ecolor = all_colors[detect_ignore][was_det_ZTF].astype('str')  ,     fmt = 'd'   , alpha = ignore_alpha, ms = 0)
    plt.errorbar(all_phases[detect_ignore][was_det_Local] , all_magnitudes[detect_ignore][was_det_Local] , all_sigmas[detect_ignore][was_det_Local] , ecolor = all_colors[detect_ignore][was_det_Local].astype('str') ,     fmt = '*'   , alpha = ignore_alpha, ms = 0)
    plt.errorbar(all_phases[detect_ignore][was_det_other], all_magnitudes[detect_ignore][was_det_other], all_sigmas[detect_ignore][was_det_other], ecolor = all_colors[detect_ignore][was_det_other].astype('str'),     fmt = '.'   , alpha = ignore_alpha, ms = 0)
    plt.scatter (all_phases[detect_ignore][was_det_ZTF]  , all_magnitudes[detect_ignore][was_det_ZTF]  ,                                           color  = all_colors[detect_ignore][was_det_ZTF].astype('str')  ,  marker = 'd'   , alpha = ignore_alpha, s = 90)
    plt.scatter (all_phases[detect_ignore][was_det_Local] , all_magnitudes[detect_ignore][was_det_Local] ,                                           color  = all_colors[detect_ignore][was_det_Local].astype('str') ,  marker = '*'   , alpha = ignore_alpha, s = 90)
    plt.scatter (all_phases[detect_ignore][was_det_other], all_magnitudes[detect_ignore][was_det_other],                                           color  = all_colors[detect_ignore][was_det_other].astype('str'),  marker = '.'   , alpha = ignore_alpha, s = 90)
    # (Upper limits)
    plt.scatter (all_phases[upper_ignore], all_magnitudes[upper_ignore], color = all_colors[upper_ignore].astype('str'), marker = 'v', alpha = ignore_alpha, s = 90)

    #plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    #plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    if full_range:
        #plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        #plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        plt.tick_params(axis='both', top=False, bottom=True, labeltop=False, labelbottom=True)
        plt.xlabel('MJD', fontsize = 11)
    else:
        plt.tick_params(axis='both', top=False, bottom=True, labeltop=False, labelbottom=True)
        plt.xlabel('Days Since Brightest Point', fontsize = 11)

    plt.ylabel('Magnitude', fontsize = 11)
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)

    # Plot fits to the data
    if plot_model:
        model_time    = np.linspace(-80, 200, 1000)
        model_red     = linex(model_time, red_amplitude  , red_amplitude2  , red_offset  , red_magnitude  )
        model_green   = linex(model_time, green_amplitude, green_amplitude2, green_offset, green_magnitude)

        plt.plot(model_time, model_red    , color = 'r', linestyle = ':', linewidth = 0.5)
        plt.plot(model_time, model_green  , color = 'g', linestyle = ':', linewidth = 0.5)

    # Plot comparison TDE, Ia, and SLSN
    if plot_comparison & (full_range == False):
        brightest_mag = np.nanmin(all_magnitudes[detection])
        max_Ia = np.nanmin(magnitude_1a_model_r + brightest_mag)
        plt.plot(time_1a_model_r, magnitude_1a_model_r + brightest_mag, color = 'm', linestyle = '--', linewidth = 1)

    used_colors, used_sources = np.unique(all_filters), np.unique(all_sources)
    return used_colors, used_sources

def plot_legend(sub_y, sub_x, sub_n, used_colors, used_sources):
    '''
    Include a legend

    Parameters
    ---------------
    sub_y, sub_x, sub_n  : Subplot parameters (rows, columns, plot number)
    used_colors   : Colors from plot_lightcurve() used
    used_sources  : Sources from plot_lightcurve() used
    '''

    # Select Plot Colors
    all_colors = plot_colors(used_colors)

    # Select groups of Data (detectiond and upper limits)
    is_det_ZTF   = np.where(used_sources == 'ZTF' )[0]
    is_det_FLWO  = np.where((used_sources == 'FLWO') | (used_sources == 'Local'))[0]
    is_det_other = np.where([i not in ['ZTF', 'FLWO', 'Local'] for i in used_sources])[0]

    plt.subplot(sub_y, sub_x, sub_n)
    if len(is_det_ZTF)   > 0: plt.scatter([], [], marker = 'd', alpha = 1.0, s = 90, color = 'k', label = 'ZTF'  )
    if len(is_det_FLWO)  > 0: plt.scatter([], [], marker = '*', alpha = 1.0, s = 90, color = 'k', label = 'Local')
    if len(is_det_other) > 0: plt.scatter([], [], marker = 'o', alpha = 1.0, s = 90, color = 'k', label = 'OSC'  )

    for i in range(len(used_colors)):
        plt.scatter([], [], marker = 'o', alpha = 1.0, s = 90, color = all_colors[i], label = used_colors[i])

    plt.legend(loc = 'center', frameon = False, fontsize = 11)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    plt.axis('off')

def query_PS1_image(ra_deg, dec_deg, image_color, wcs_size = 90, autoscale = 75):
    '''
    Query the 3pi website and download an image in the 
    specified filter. If no image was found return '--'.
    If the image_color is not found, return it in 'r' band.

    Parameters
    ---------------
    ra_deg, dec_deg : Coordinates of the object in degrees
    image_color     : Filter to search for, either 'g', 'r', 'i', 'z', or 'y'
    wcs_size        : Image size in arcsec 
    autoscale       : scaling of the image

    Output
    ---------------
    img : Image file
    '''

    image_size = wcs_size * 4

    # Request data from PS1 website
    if image_color in 'grizy':
        if dec_deg > 0:
            requests_link = "http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos=" + str(np.around(ra_deg, decimals = 6)) + "+" + str(np.around(dec_deg, decimals = 6)) + "&filter=" + image_color + "&size=" + str(image_size) + "&autoscale=" + str(autoscale)
        else:
            requests_link = "http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos=" + str(np.around(ra_deg, decimals = 6))       + str(np.around(dec_deg, decimals = 6)) + "&filter=" + image_color + "&size=" + str(image_size) + "&autoscale=" + str(autoscale)

    # If no available filter, use r
    else:
        if dec_deg > 0:
            requests_link = "http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos=" + str(np.around(ra_deg, decimals = 6)) + "+" + str(np.around(dec_deg, decimals = 6)) + "&filter=" + 'r' + "&size=" + str(image_size) + "&autoscale=" + str(autoscale)
        else:
            requests_link = "http://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos=" + str(np.around(ra_deg, decimals = 6))       + str(np.around(dec_deg, decimals = 6)) + "&filter=" + 'r' + "&size=" + str(image_size) + "&autoscale=" + str(autoscale)

    python_version = sys.version_info[0]
    try:
        if python_version == 3:
            # Extract image
            image_req  = requests.get(requests_link)
            image_data = image_req.text
            soup       = BeautifulSoup(image_data, "html5lib")
            URL        = 'http:' + soup.find_all('img')[1].get('src')
            file       = BytesIO(urllib.request.urlopen(URL).read())
        elif python_version == 2:
            import cStringIO
            # Extract image
            image_req  = requests.get(requests_link)
            image_data = image_req.text
            soup       = BeautifulSoup(image_data, "lxml")
            URL        = 'http:' + soup.find_all('img')[1].get('src')
            file       = cStringIO.StringIO(urllib.urlopen(URL).read())
        img = Image.open(file)
        return img
    except:
        return '--'

def plot_field_image(sub_y, sub_x, sub_n, ra_deg, dec_deg, object_name, image_color = 'r', search_radius = 1.0, autoscale = 75):
    '''
    Plot the field image at the given RA and DEC from 3PI

    Parameters
    ---------------
    sub_y, sub_x, sub_n : Subplot parameters (rows, columns, plot number)
    ra_deg, dec_deg     : Coordiantes of the star in degrees
    object_name         : Title with transient type
    image_color         : Filter name
    search_radius       : Radius of the image in arcminutes
    autoscale           : PS1 scaling (lower is more contrast)
    '''

    # Create folder to store images, if it doesn't exist
    if len(glob.glob('images')) == 0:
        os.system("mkdir images")

    # If the image exists, open it
    wcs_size = int(search_radius * 2 * 60)
    exists = glob.glob('images/%s.jpeg'%object_name)
    if len(exists) == 0:
        img = query_PS1_image(float(ra_deg), float(dec_deg), image_color, wcs_size, autoscale)
        try:
            img.save('images/%s.jpeg'%object_name, 'jpeg')
        except:
            pass
    else:
        img = Image.open('images/%s.jpeg'%object_name)        

    plt.subplot(sub_y, sub_x, sub_n)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    if img == '--':
        plt.annotate('Outside PS1 footprint', xy=(0,0))
        plt.set_xlim(-0.5, 1)
        plt.set_ylim(-1, 1)
    else:
        plt.imshow(np.array(img), cmap = 'viridis')

    plt.scatter(wcs_size*2, wcs_size*2, marker = '+', color = 'r')

def extinct(output_table, g_correct = 0, r_correct = 0):
    '''
    Correct table for extinction, only g and r bands 
    for plotting purposes

    Parameters
    ---------------
    output_table     : Data with photometry
    g_correct        : extinction value in g band
    r_correct        : extinction value in r band

    Output
    ---------------
    output_table_correct
    '''

    output_table_correct = table.Table(output_table)

    # Separate into green and red filters
    green = np.where(((output_table['Filter'] == 'g') | (output_table['Filter'] == "g'")))
    red   = np.where(((output_table['Filter'] == 'r') | (output_table['Filter'] == 'R') | (output_table['Filter'] == "r'")))

    output_table_correct['Mag'][green] = output_table['Mag'][green] - g_correct
    output_table_correct['Mag'][red]   = output_table['Mag'][red]   - r_correct

    return output_table_correct

def plot_nature_mag_distance(sub_y, sub_x, sub_n, separation, hosts_magnitudes, output_nature, transient_magnitude, host_separation, host_magnitude, host_nature, closest_separation, closest_magnitude, closest_nature, search_radius):
    '''
    Plot the host magnitude as a function of distance from transient
    And make the size equal to the probability of being a galaxy

    Parameters
    ---------------
    sub_y, sub_x, sub_n : Subplot parameters (rows, columns, plot number)
    separation          : Separation between transient and objects in arcsec
    hosts_magnitudes    : Magnitude of objects
    output_nature       : Probability of object being a galaxy
    transient_magnitude : Brightest transient magnitude
    host_separation     : Separation of the host and transient
    host_magnitude      : Magnitude of the host
    host_nature         : Probability of host being a galaxy
    search_radius       : For plot limit purposes
    '''

    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_yaxis()
    plt.title("Probability of being a Galaxy")
    plt.scatter(separation, hosts_magnitudes, s = (output_nature + 0.1)*1000, c = (output_nature + 0.1)*1000, vmin=0, vmax=1000, alpha = 0.75)
    plt.scatter(host_separation, host_magnitude, marker='+', alpha = 1.0, color = 'b', s = 1000)
    plt.axhline(float(transient_magnitude), color = 'b', linestyle = '--', linewidth = 1, label = "Brightest Mag = %s"%np.around(float(transient_magnitude), decimals = 2))
    plt.axhline(float(host_magnitude), color = 'k', linestyle = '-.', linewidth = 1)
    plt.annotate(str(np.around(host_nature, decimals = 2)), xy=(host_separation, host_magnitude))
    if host_separation != closest_separation:
        plt.annotate(str(np.around(closest_nature, decimals = 2)), xy=(closest_separation, closest_magnitude))
    plt.legend(loc = 'best', fancybox = True)
    plt.xlim(0, search_radius * 60 + 2)
    plt.xlabel("Distance from Transient [Arcsec]")
    plt.ylabel("Magnitude")

def plot_host_mag_distance(sub_y, sub_x, sub_n, separation_total, brightest_host, output_nature, average_coincidence, transient_magnitude, host_separation, host_magnitude, host_Pcc, closest_separation, closest_magnitude, closest_coincidence, search_radius, star_cut):
    '''
    Plot the host magnitude as a function of distance from transient
    And make the size equal to the probability of being the host

    Parameters
    ---------------
    sub_y, sub_x, sub_n : Subplot parameters (rows, columns, plot number)
    separation_total    : Separation between transient and objects in arcsec
    brightest_host      : Magnitude of objects
    output_nature       : Probability of object being a galaxy
    average_coincidence : Probability of chance coincidence for objects in field
    transient_magnitude : Brightest transient magnitude
    host_separation     : Separation of the host and transient
    host_magnitude      : Magnitude of the host
    host_Pcc            : Probability of chance coincidece of the host
    maybe_galaxies      : Which objects from the total list are galaxies
    search_radius       : For plot limit purposes
    star_cut            : which of the objects are likely galaxies
    closest_separation  : Separation of the closest object
    closest_magnitude   : Magnitude of the closest object
    closest_coincidence : Probability of chance coincidece of the closest object
    '''

    maybe_galaxies = np.where(output_nature >  star_cut)[0]
    stars          = np.where(output_nature <= star_cut)[0]

    average_probability = 1 - average_coincidence
    host_probability    = 1 - host_Pcc
    closest_probability = 1 - closest_coincidence
    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_yaxis()
    plt.title("Probability of being the Host")
    plt.scatter(separation_total[maybe_galaxies], brightest_host[maybe_galaxies], s = (average_probability[maybe_galaxies] + 0.1)*1000, c = (average_probability[maybe_galaxies] + 0.1)*1000, vmin=0, vmax=1000, alpha = 0.75)
    plt.scatter(separation_total[stars], brightest_host[stars],                   s = 100                            , c = 'r'                            , marker = 'x', alpha = 0.75, label = 'Stars')
    plt.scatter(host_separation, host_magnitude, marker='+', alpha = 1.0, color = 'b', s = 1000)
    plt.axhline(float(transient_magnitude), color = 'b', linestyle = '--', linewidth = 1)
    plt.axhline(float(host_magnitude), color = 'k', linestyle = '-.', linewidth = 1)
    if host_separation != closest_separation:
        plt.annotate(str(np.around(closest_probability, decimals = 2)), xy=(closest_separation, closest_magnitude))
    plt.annotate(str(np.around(host_probability, decimals = 2)), xy=(host_separation, host_magnitude))
    plt.legend(loc = 'best', fancybox = True)
    plt.xlim(0, search_radius * 60 + 2)
    plt.xlabel("Distance from Transient [Arcsec]")
    plt.ylabel("Magnitude")

#def plot_host_information(sub_y, sub_x, sub_n, data_catalog, best_host, output_table, TNS_name, ZTF_name, ra_deg, dec_deg, host_separation, host_radius, star_host, magnitudes_array, magnitudes_colors, redshift, z_ned, z_flag_ned, z_simbad, z_sdss, zerr_sdss, TNS_class, TNS_redshift):
def plot_host_information(sub_y, sub_x, sub_n, ra_deg, dec_deg, info_table, host_radius, host_separation, host_magnitude, transient_magnitude):
    '''
    Make a plot with information about the transient and the host

    Parameters
    ---------------
    sub_y, sub_x, sub_n : Subplot parameters (rows, columns, plot number)
    ra_deg, dec_deg     : Coordiantes of the target in degrees
    info_table          : Table with all the output FLEET information
    host_radius         : The half-light radius of the best host in arcsec
    host_separation     : The transient-host separation in arcsec
    host_magnitude      : The magnitude of the best host
    transient_magnitude : Peak Magnitude of the transient
    '''
    # Empty array
    emptys = ['',' ','None','--', '-', b'',b' ',b'None',b'--', b'-', None, np.nan, 'nan', b'nan', '0', np.inf, 'inf']

    # Name
    object_name = info_table['object_name'][0]
    ZTF_name    = info_table['ztf_name'][0]
    TNS_name    = info_table['tns_name'][0]
    if object_name == ZTF_name: ZTF_name = ''
    if object_name == TNS_name: TNS_name = ''
    if ZTF_name in emptys: ZTF_name = ''
    if TNS_name in emptys: TNS_name = ''

    # Classification
    object_class = info_table['object_class'][0]

    # Coordinates Info
    coords    = SkyCoord(ra_deg, dec_deg, unit = 'deg')
    coord_str = coords.to_string('hmsdms', sep = ':')
    RA_hms    = coord_str[:coord_str.find(' ')]
    DEC_dms   = coord_str[coord_str.find(' ')+1:]
    l_deg     = coords.galactic.l.value
    b_deg     = coords.galactic.b.value

    # Redsfhit Info
    redshift = info_table['use_redshift']
    redshift_label = info_table['redshift_label']

    # If there's additional photo-z info, add it
    if (redshift_label == 'specz'):
        try:
            photoz     = float(info_table['photoz'])
            photoz_err = float(info_table['photoz_err'])
        except:
            photoz = photoz_err = '--'
    else:
        photoz = photoz_err = '--'

    # Classifier Info
    P_quick_Nuclear         = float(info_table['P_quick_Nuclear'])
    P_quick_SLSNI           = float(info_table['P_quick_SLSNI'])
    P_quick_SLSNII          = float(info_table['P_quick_SLSNII'])
    P_quick_SNII            = float(info_table['P_quick_SNII']) + float(info_table['P_quick_SNIIb']) + float(info_table['P_quick_SNIIn'])
    P_quick_SNI             = float(info_table['P_quick_SNIa']) + float(info_table['P_quick_SNIbc'])
    P_quick_Star            = float(info_table['P_quick_Star'])
    P_quick_Nuclear_std     = float(info_table['P_quick_Nuclear_std'])
    P_quick_SLSNI_std       = float(info_table['P_quick_SLSNI_std'])
    P_quick_SLSNII_std      = float(info_table['P_quick_SLSNII_std'])
    P_quick_SNII_std        = np.sqrt(float(info_table['P_quick_SNII_std']) ** 2 + float(info_table['P_quick_SNIIb_std']) ** 2 + float(info_table['P_quick_SNIIn_std']) ** 2)
    P_quick_SNI_std         = np.sqrt(float(info_table['P_quick_SNIa_std']) ** 2 + float(info_table['P_quick_SNIbc_std']) ** 2)
    P_quick_Star_std        = float(info_table['P_quick_Star_std'])

    P_late_Nuclear          = float(info_table['P_late_Nuclear'])
    P_late_SLSNI            = float(info_table['P_late_SLSNI'])
    P_late_SLSNII           = float(info_table['P_late_SLSNII'])
    P_late_SNII             = float(info_table['P_late_SNII']) + float(info_table['P_late_SNIIb']) + float(info_table['P_late_SNIIn'])
    P_late_SNI              = float(info_table['P_late_SNIa']) + float(info_table['P_late_SNIbc'])
    P_late_Star             = float(info_table['P_late_Star'])
    P_late_Nuclear_std      = float(info_table['P_late_Nuclear_std'])
    P_late_SLSNI_std        = float(info_table['P_late_SLSNI_std'])
    P_late_SLSNII_std       = float(info_table['P_late_SLSNII_std'])
    P_late_SNII_std         = np.sqrt(float(info_table['P_late_SNII_std']) ** 2 + float(info_table['P_late_SNIIb_std']) ** 2 + float(info_table['P_late_SNIIn_std']) ** 2)
    P_late_SNI_std          = np.sqrt(float(info_table['P_late_SNIa_std']) ** 2 + float(info_table['P_late_SNIbc_std']) ** 2)
    P_late_Star_std         = float(info_table['P_late_Star_std'])

    P_redshift_Nuclear      = float(info_table['P_redshift_Nuclear'])
    P_redshift_SLSNI        = float(info_table['P_redshift_SLSNI'])
    P_redshift_SLSNII       = float(info_table['P_redshift_SLSNII'])
    P_redshift_SNII         = float(info_table['P_redshift_SNII']) + float(info_table['P_redshift_SNIIb']) + float(info_table['P_redshift_SNIIn'])
    P_redshift_SNI          = float(info_table['P_redshift_SNIa']) + float(info_table['P_redshift_SNIbc'])
    P_redshift_Star         = float(info_table['P_redshift_Star'])
    P_redshift_Nuclear_std  = float(info_table['P_redshift_Nuclear_std'])
    P_redshift_SLSNI_std    = float(info_table['P_redshift_SLSNI_std'])
    P_redshift_SLSNII_std   = float(info_table['P_redshift_SLSNII_std'])
    P_redshift_SNII_std     = np.sqrt(float(info_table['P_redshift_SNII_std']) ** 2 + float(info_table['P_redshift_SNIIb_std']) ** 2 + float(info_table['P_redshift_SNIIn_std']) ** 2)
    P_redshift_SNI_std      = np.sqrt(float(info_table['P_redshift_SNIa_std']) ** 2 + float(info_table['P_redshift_SNIbc_std']) ** 2)
    P_redshift_Star_std     = float(info_table['P_redshift_Star_std'])

    P_host_Nuclear          = float(info_table['P_host_Nuclear'])
    P_host_SLSNI            = float(info_table['P_host_SLSNI'])
    P_host_SLSNII           = float(info_table['P_host_SLSNII'])
    P_host_SNII             = float(info_table['P_host_SNII']) + float(info_table['P_host_SNIIb']) + float(info_table['P_host_SNIIn'])
    P_host_SNI              = float(info_table['P_host_SNIa']) + float(info_table['P_host_SNIbc'])
    P_host_Star             = float(info_table['P_host_Star'])
    P_host_Nuclear_std      = float(info_table['P_host_Nuclear_std'])
    P_host_SLSNI_std        = float(info_table['P_host_SLSNI_std'])
    P_host_SLSNII_std       = float(info_table['P_host_SLSNII_std'])
    P_host_SNII_std         = np.sqrt(float(info_table['P_host_SNII_std']) ** 2 + float(info_table['P_host_SNIIb_std']) ** 2 + float(info_table['P_host_SNIIn_std']) ** 2)
    P_host_SNI_std          = np.sqrt(float(info_table['P_host_SNIa_std']) ** 2 + float(info_table['P_host_SNIbc_std']) ** 2)
    P_host_Star_std         = float(info_table['P_host_Star_std'])

    ############## Generate Legend ##############

    # Names
    plt.subplot(sub_y, sub_x, sub_n)
    plt.title('%s     %s    %s'%(object_name, TNS_name, ZTF_name))

    # Coordinates
    legend_name  = 'RA = %s  DEC = %s\n'%(RA_hms, DEC_dms)
    legend_name += '         %s                     %s\n'%(round(ra_deg,5), round(dec_deg,5))
    legend_name += ' l = %s               b = %s\n\n'%(round(l_deg,5) , round(b_deg,5))

    # Magnitudes
    if object_class not in emptys : legend_name += object_class + '\n'
    legend_name += r'$\Delta$M = %s $- %s = %s$'%(round(host_magnitude, 2), round(transient_magnitude, 2), round(host_magnitude - transient_magnitude, 2)) + '\n'

    # Size
    legend_name += 'Size = %s'%(round(host_radius, 2)) + "''\n"
    legend_name += 'Separation = %s'%(round(host_separation, 2)) + "''\n"
    legend_name += r'Offset = %s $R_e$'%(round(host_separation/host_radius, 2)) + '\n\n'

    # Redshift And Magnitude    
    absolute_magnitude = '--'
    if redshift not in emptys:
        if np.isfinite(float(redshift)):
            legend_name += r'z = $%s$ (%s)'%(round(float(redshift), 3), redshift_label[0]) + '\n'
            absolute_magnitude, _ = redshift_magnitude(transient_magnitude, float(redshift))
    if photoz not in emptys:
        if np.isfinite(float(photoz)):
            legend_name += r'z = $%s\pm%s$ (%s)'%(round(float(photoz), 3), round(photoz_err, 3), 'photoz') + '\n'
    if str(absolute_magnitude) not in emptys : legend_name += 'Abs. Mag = %s\n\n'%(round(absolute_magnitude, 2))

    # ML Classification
    legend_name += '           Nuclear  SLSN-I  SLSN-II  SNII  SNI  Star\n'
    if np.isfinite(P_quick_Nuclear):
        if np.isfinite(P_quick_Nuclear_std):
            legend_name += r'early       %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f'%(P_quick_Nuclear,P_quick_Nuclear_std,P_quick_SLSNI,P_quick_SLSNI_std,P_quick_SLSNII,P_quick_SLSNII_std,P_quick_SNII,P_quick_SNII_std,P_quick_SNI,P_quick_SNI_std,P_quick_Star,P_quick_Star_std) + '\n'
        else:
            legend_name += r'early       %.0f        %.0f        %.0f        %.0f        %.0f        %.0f     '%(P_quick_Nuclear,P_quick_SLSNI,P_quick_SLSNII,P_quick_SNII,P_quick_SNI,P_quick_Star) + '\n'
    if np.isfinite(P_late_Nuclear):
        if np.isfinite(P_late_Nuclear_std):
            legend_name += r'late        %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f'%(P_late_Nuclear,P_late_Nuclear_std,P_late_SLSNI,P_late_SLSNI_std,P_late_SLSNII,P_late_SLSNII_std,P_late_SNII,P_late_SNII_std,P_late_SNI,P_late_SNI_std,P_late_Star,P_late_Star_std) + '\n'
        else:
            legend_name += r'late        %.0f        %.0f        %.0f        %.0f        %.0f        %.0f     '%(P_late_Nuclear,P_late_SLSNI,P_late_SLSNII,P_late_SNII,P_late_SNI,P_late_Star) + '\n'
    if np.isfinite(P_redshift_Nuclear):
        if np.isfinite(P_redshift_Nuclear_std):
            legend_name += r'redshift  %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f'%(P_redshift_Nuclear,P_redshift_Nuclear_std,P_redshift_SLSNI,P_redshift_SLSNI_std,P_redshift_SLSNII,P_redshift_SLSNII_std,P_redshift_SNII,P_redshift_SNII_std,P_redshift_SNI,P_redshift_SNI_std,P_redshift_Star,P_redshift_Star_std) + '\n'
        else:
            legend_name += r'redshift  %.0f        %.0f        %.0f        %.0f        %.0f        %.0f     '%(P_redshift_Nuclear,P_redshift_SLSNI,P_redshift_SLSNII,P_redshift_SNII,P_redshift_SNI,P_redshift_Star) + '\n'
    if np.isfinite(P_host_Nuclear):
        if np.isfinite(P_host_Nuclear_std):
            legend_name += r'host        %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f   %.0f$\pm$%.0f'%(P_host_Nuclear,P_host_Nuclear_std,P_host_SLSNI,P_host_SLSNI_std,P_host_SLSNII,P_host_SLSNII_std,P_host_SNII,P_host_SNII_std,P_host_SNI,P_host_SNI_std,P_host_Star,P_host_Star_std)
        else:
            legend_name += r'host        %.0f        %.0f        %.0f        %.0f        %.0f        %.0f     '%(P_host_Nuclear,P_host_SLSNI,P_host_SLSNII,P_host_SNII,P_host_SNI,P_host_Star)

    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    plt.annotate(legend_name, xy = (0.02, 0.96), fontsize = 10, va = 'top')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

def plot_observability(sub_y, sub_x, sub_n, ra_deg, dec_deg, Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan):
    '''
    Plot airmass graphs of a given object for MMT and Magellan 

    Parameters
    ---------------
    sub_y, sub_x, sub_n                     : Subplot parameters (rows, columns, plot number)
    ra_deg, dec_deg                         : Coordiantes of the star in degrees
    Dates_MMT, Dates_Magellan               : Time array for MMT and Magellan airmass graph
    Airmass_MMT, Airmass_Magellan           : Airmass array of MMT and Magellan
    SunElevation_MMT, SunElevation_Magellan : Sun Elevantion for each Time, to calculate twilight
    '''

    # Airmass plots
    axes = plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_yaxis()
    plt.plot(Dates_MMT, Airmass_MMT, color = 'g', linewidth = 4, label = 'MMT Airmass')
    plt.fill_between(Dates_MMT, 0, 3.0, where=SunElevation_MMT <= -18.0, facecolor = 'g', alpha = 0.25, label = 'MMT Night')
    plt.plot(Dates_Magellan, Airmass_Magellan, color = 'r', linewidth = 4, label = 'Magellan Airmass')
    plt.fill_between(Dates_Magellan, 0.0, 3.0, where=SunElevation_Magellan <= -18.0, facecolor = 'r', alpha = 0.25, label = 'Magellan Night')
    plt.legend(loc = 'upper left')
    plt.ylabel("Airmass")
    plt.title("UT = %s"%datetime.datetime.now().strftime('%Y-%m-%d') + "     MJD = %s"%np.around(Time(datetime.datetime.now()).mjd, decimals = 1))
    plt.ylim(3, 0.5)
    plt.xlim(min(Dates_MMT), max(Dates_MMT))

    # Change Axes Labels
    plt.xticks(Dates_MMT[::round(len(Dates_MMT) / 10)])
    xfmt = md.DateFormatter('%H:%M')
    axes.xaxis.set_major_formatter(xfmt)
    plt.xlabel('UT')

def plot_ra_dec_size(sub_y, sub_x, sub_n, objects_ra, objects_dec, ra_deg, dec_deg, output_nature, host_ra, host_dec, search_radius, halflight_radius, transient_magnitude):
    '''
    Plot the field objects in RA and DEC, where the size
    is the size of the objects

    Parameters
    ---------------
    sub_y, sub_x, sub_n : Subplot parameters (rows, columns, plot number)
    objects_ra          : RA in degrees for objects in field
    objects_dec         : DEC in degrees for objects in field
    ra_deg              : Target RA
    dec_deg             : Target DEC
    output_nature       : Probabiliy of objects being a galaxy
    host_ra             : Host RA
    host_dec            : Host DEC
    search_radius       : For plot limit purposes
    halflight_radius    : Size of objects (half light radius)
    transient_magnitude : Magnitude of the transient
    '''

    Lum_Transient = 10 ** 9 * (10 ** (-0.4 * float(transient_magnitude)))

    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_xaxis()
    plt.title("Probability of being a Galaxy (Size)")
    plt.scatter(flot(objects_ra), flot(objects_dec), s = (output_nature + 0.1)*1000, c = (output_nature + 0.1)*1000, vmin=0, vmax=1000, alpha = 0.75)
    plt.scatter(ra_deg, dec_deg, marker='*', color = 'r', s = Lum_Transient)
    plt.scatter(host_ra, host_dec, marker='+', alpha = 1.0, color = 'b', s = 1000)
    plt.xlabel("RA")
    plt.ylabel("DEC")
    plt.xlim(ra_deg + search_radius / 60, ra_deg - search_radius / 60)
    plt.ylim(dec_deg - search_radius / 60, dec_deg + search_radius / 60)
    #ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    #ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.ticklabel_format(useOffset=False)
    for j in range(len(objects_ra)):
        if np.isfinite(halflight_radius[j]):
            plt.annotate(str(np.around(halflight_radius[j], decimals = 2)), xy=(objects_ra[j], objects_dec[j]))

def plot_ra_dec_magnitude(sub_y, sub_x, sub_n, objects_ra, objects_dec, ra_deg, dec_deg, hosts_magnitudes, host_ra, host_dec, search_radius, transient_magnitude):
    '''
    Plot the field objects in RA and DEC, where the size
    is the size of the objects

    Parameters
    ---------------
    sub_y, sub_x, sub_n : Subplot parameters (rows, columns, plot number)
    objects_ra          : RA in degrees for objects in field
    objects_dec         : DEC in degrees for objects in field
    ra_deg              : Target RA
    dec_deg             : Target DEC
    hosts_magnitudes    : Magnitudes of the objects in the field
    host_ra             : Host RA
    host_dec            : Host DEC
    search_radius       : For plot limit purposes
    transient_magnitude : Magnitude of the transient
    '''

    Lum_Transient = 10 ** 9 * (10 ** (-0.4 * float(transient_magnitude)))
    Lum = 10 ** 9 * (10 ** (-0.4 * hosts_magnitudes))

    plt.subplot(sub_y, sub_x, sub_n)
    plt.gca().invert_xaxis()
    plt.title("Magnitude")
    plt.scatter(flot(objects_ra), flot(objects_dec), s = 40 * Lum, c = Lum, alpha = 0.75)
    plt.scatter(ra_deg, dec_deg, marker='*', color = 'r', s = Lum_Transient)
    plt.scatter(host_ra, host_dec, marker='+', alpha = 1.0, color = 'b', s = 1000, label = 'Best Host')
    plt.legend(loc = 'best')
    plt.xlabel("RA")
    plt.xlim(ra_deg + search_radius / 60, ra_deg - search_radius / 60)
    plt.ylim(dec_deg - search_radius / 60, dec_deg + search_radius / 60)
    plt.ticklabel_format(useOffset=False)
    #ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    #ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

def calc_airmass(RA, DEC, Telescope, do_observability = True):
    '''
    Calculate the Airmass of a star from a specified telescope

    Parameters
    ---------------
    RA, DEC   : Coordiantes of the star in 00:00:00 format
    Telescope : Telescope for which to calculate the airmass

    Return
    ---------------
    Dates        : Array of dates to calculate the airmass
    Airmass      : Airmass value at each date
    SunElevation : Sun's elevation at each date
    '''

    # Specify the coordinates based on the observatory
    if Telescope == 'MMT'          : Lat, Lon, Alt =  31.6883, -110.885, 2608
    if Telescope == 'Magellan'     : Lat, Lon, Alt = -29.0182, -70.6915, 2516
    if Telescope == 'FLWO'         : Lat, Lon, Alt =  31.6816, -110.876, 2344
    if Telescope == 'Kitt Peak'    : Lat, Lon, Alt =  31.9633, -111.600, 2120
    if Telescope == 'McDonald'     : Lat, Lon, Alt =  30.6716, -104.021, 2075
    if Telescope == 'CTIO'         : Lat, Lon, Alt = -30.1650, -70.8150, 2215
    if Telescope == 'Harvard'      : Lat, Lon, Alt =  42.3770, -71.1167,   54
    if Telescope == 'Las Campanas' : Lat, Lon, Alt = -29.0182, -70.6915, 2380
    if Telescope == 'AAT'          : Lat, Lon, Alt = -31.2754, 149.0672, 1164
    if Telescope == 'HWT'          : Lat, Lon, Alt =  28.7610, -17.8820, 2332
    if Telescope == 'Gemini_North' : Lat, Lon, Alt =  19.8200, -155.468, 4213
    if Telescope == 'Gemini_South' : Lat, Lon, Alt = -30.2280, -70.7230, 2725

    # Create the observatory
    Observatory = ephem.Observer()
    Observatory.lat = str(int(Lat)) + ":" + str(np.around((Lat - int(Lat))*60, 1))
    Observatory.lon = str(int(Lon)) + ":" + str(np.around((Lon - int(Lon))*60, 1))
    Observatory.elevation = Alt

    # Choose start of date counter as right now with a stepsize in minutes
    # But set the start hour to 4PM
    Date0 = datetime.datetime.now()
    Date0 = Date0.replace(hour = 16)
    Stepsize = 30

    # Empty Variables to modify
    Dates = np.array([])

    # Define the start time of the observations, and the subsequent timesteps
    # For which to do the calculations.
    Step  = datetime.timedelta(0, 60*Stepsize)
    Hours = np.arange(24*60/Stepsize)
    for i in range(len(Hours)):
        Dates = np.append(Dates, Date0 + Step * i)

    # Don't actually run the function if requested
    if do_observability == False:
        return Dates, np.nan * np.ones(len(Dates)), np.nan * np.ones(len(Dates))

    # Calculate the elevation of the sun
    SunElevation = np.zeros(len(Dates))
    for i in range(len(Dates)):
        SunElevation[i] = Astral.solar_elevation(Astral(), Dates[i], Lat, Lon)

    # Create the star
    Star = ephem.readdb("Star,f|M|A0,%s,%s,0,2000"%(RA, DEC))

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

def is_it_observable(Airmass_telescope, SunElevation_telescope, SunElevation = -18.0, max_airmass = 2.0):
    '''
    Determine if at any point an object is observable 
    given its airmass and the Sun's elevation.

    Parameters
    ---------------
    Airmass_telescope      : Array of airmasses
    SunElevation_telescope : Array of Sun's elevation
    SunElevation           : Minimum Sun's elevation for astornomical twilight
    max_airmass            : Maximum acceptable airmass for the object to be observable

    Returns
    -------------
    Is the object observable? True or False
    '''

    # Is the airmass a real number
    good_telescope = np.isfinite(Airmass_telescope)
    # Does it satisfy the conditions
    observable_telescope = np.where((SunElevation_telescope[good_telescope] <= SunElevation)&(Airmass_telescope[good_telescope] < max_airmass))[0]

    if len(observable_telescope) > 0:
        return True
    else:
        return False

def calculate_observability(ra_deg, dec_deg, do_observability = True, SunElevation = -18.0, max_airmass = 2.0):
    '''
    Calculate the airmass for MMT and Magellan for an object
    given an RA and DEC.

    Parameters
    ---------------
    ra_deg, dec_deg : Coordiantes of the star in degrees
    SunElevation    : When does twilight start
    max_airmass     : What's the maximum airmass to consider

    Return
    ---------------
    Dates_MMT, Dates_Magellan               : Time array for MMT and Magellan airmass graph
    Airmass_MMT, Airmass_Magellan           : Airmass array of MMT and Magellan
    SunElevation_MMT, SunElevation_Magellan : Sun Elevantion for each Time, to calculate twilight
    MMT_observable, Magellan_observable     : Is the object visible from MMT and Magellan (True or False)
    '''

    # Convert coordinates to string format
    coord     = SkyCoord(ra_deg, dec_deg, unit="deg")
    coord_str = coord.to_string('hmsdms', sep = ':')
    RA_str    = coord_str[:coord_str.find(' ')]
    DEC_str   = coord_str[coord_str.find(' ')+1:]

    # Calculate airmass for MMT and Magellan
    Dates_MMT,      Airmass_MMT,      SunElevation_MMT      = calc_airmass(RA_str, DEC_str, 'MMT'     , do_observability)
    Dates_Magellan, Airmass_Magellan, SunElevation_Magellan = calc_airmass(RA_str, DEC_str, 'Magellan', do_observability)

    # Calculate if the source will be observable from MMT and Magellan
    if do_observability:
        MMT_observable      = is_it_observable(Airmass_MMT     , SunElevation_MMT     , SunElevation, max_airmass)
        Magellan_observable = is_it_observable(Airmass_Magellan, SunElevation_Magellan, SunElevation, max_airmass)
    else:
        MMT_observable      = '--'
        Magellan_observable = '--'

    return Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, MMT_observable, Magellan_observable

def plot_blackbody(sub_y, sub_x, sub_n, data_catalog, best_host):
    '''
    The the SED of the best host and a blackbody fit

    Parameters
    ---------------
    sub_y, sub_x, sub_n : Subplot parameters (rows, columns, plot number)
    data_catalog        : Astropy table with catalog data 
    best_host           : Index of best host
    '''

    # Data to search for
    filters_list     = np.array(['g'  , 'r'   , 'i'   , 'z'   , 'y'  , 'u'   , 'g'   , 'r'   , 'i'   , 'z'   ])
    surveys_list     = np.array(['3pi', '3pi' , '3pi' , '3pi' , '3pi', 'sdss', 'sdss', 'sdss', 'sdss', 'sdss'])
    zeropoints_list  = np.array([4.773e-9,2.897e-9,1.943e-9,1.452e-9,1.180e-9,8.423e-9,5.055e-9,2.904e-9,1.967e-9,1.375e-9])
    wavelengths_list = np.array([4775.62,6129.53,7484.60,8657.82,9603.10,3594.90,4640.40,6122.30,7439.50,8897.10])
    magnitudes_list  = np.array([float(get_kron_and_psf(i, j, data_catalog[best_host:best_host+1]              )[0]) for i, j in zip(filters_list, surveys_list)])
    errors_list      = np.array([float(get_kron_and_psf(i, j, data_catalog[best_host:best_host+1], error = True)[0]) for i, j in zip(filters_list, surveys_list)])
    flux_list        = 4*np.pi*zeropoints_list*10**(-0.4*magnitudes_list)
    error_flux       = 2.5/np.log(10) * flux_list * errors_list

    # Select Surveys
    PI3   = np.where([surveys_list == '3pi'  ])[0]
    SDSS  = np.where([surveys_list == 'sdss' ])[0]
    MASS2 = np.where([surveys_list == '2mass'])[0]
    WISE  = np.where([surveys_list == 'wise' ])[0]

    # Check if there is data
    PI3_exists   = np.nansum(magnitudes_list[PI3  ]) > 0
    SDSS_exists  = np.nansum(magnitudes_list[SDSS ]) > 0
    MASS2_exists = np.nansum(magnitudes_list[MASS2]) > 0
    WISE_exists  = np.nansum(magnitudes_list[WISE ]) > 0

    # Plot
    plt.subplot(sub_y, sub_x, sub_n)
    if PI3_exists   : plt.errorbar(wavelengths_list[PI3  ], flux_list[PI3  ], error_flux[PI3  ], fmt = 'o', color = 'g',      alpha = 0.7, label = '3PI'  )
    if SDSS_exists  : plt.errorbar(wavelengths_list[SDSS ], flux_list[SDSS ], error_flux[SDSS ], fmt = 'o', color = 'r',      alpha = 0.7, label = 'SDSS' )
    if MASS2_exists : plt.errorbar(wavelengths_list[MASS2], flux_list[MASS2], error_flux[MASS2], fmt = 'o', color = 'orange', alpha = 0.7, label = '2MASS')
    if WISE_exists  : plt.errorbar(wavelengths_list[WISE ], flux_list[WISE ], error_flux[WISE ], fmt = 'o', color = 'gray',   alpha = 0.7, label = 'WISE' )

    plt.legend(loc = 'upper right')
    plt.title('Best Host SED')
    plt.xlim(2900, 260000)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Wavelength [\u212b]')
    plt.ylabel('Flux')

def plot_coordinates(sub_y, sub_x, sub_n, ra_deg, dec_deg, data_catalog, closest, acceptance_boxsize = 1.5):
    '''
    Plot the closest area and the coordinates of all the catalogs of the closest object.

    Parameters
    ---------------
    sub_y, sub_x, sub_n : Subplot parameters (rows, columns, plot number)
    ra_deg, dec_deg     : Coordiantes of the star in degrees
    data_catalog        : Catalog of objects in field
    closest             : Index of the best host
    acceptance_boxsize  : Objects inside this box were matched in catalog (arcsec)
    '''

    # Get RA and DEC from the target
    target   = data_catalog[closest]
    all_ras  = flot(target['ra_matched'])
    all_decs = flot(target['dec_matched'])

    # If Gaia coordinates exist, use those above all others
    if 'RA_ICRS_gaia' in target.colnames:
        ra_gaia  = flot(target['RA_ICRS_gaia'])
        dec_gaia = flot(target['DE_ICRS_gaia'])

        if np.isfinite(ra_gaia):
            # Replace values
            all_ras  = ra_gaia 
            all_decs = dec_gaia

    # Plot data
    plt.subplot(sub_y, sub_x, sub_n)
    # Offset
    o = acceptance_boxsize / 3600 / 2
    # plot square region
    x_box = np.array([ all_ras - o,  all_ras + o,  all_ras + o,  all_ras - o,  all_ras - o])
    y_box = np.array([all_decs - o, all_decs - o, all_decs + o, all_decs + o, all_decs - o])
    plt.plot((x_box - all_ras) * 3600, (y_box - all_decs) * 3600, color = 'b', linewidth = 1, alpha = 0.5)

    plt.title('Closest Host')
    plt.scatter((all_ras - all_ras) * 3600, (all_decs - all_decs) * 3600, marker = '+', color = 'b', s = 90 , alpha = 0.5, zorder = 10, label = 'Joint' )
    plt.scatter((ra_deg  - all_ras) * 3600, (dec_deg  - all_decs) * 3600, marker = '*', color = 'b', s = 200, alpha = 0.7,              label = 'Target')
    # Calculate Errorbars
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if 'errMaj_2mass'  in data_catalog.colnames: err_2mass = np.nanmean([np.array(target['errMaj_2mass']).astype(float), np.array(target['errMin_2mass']).astype(float)], axis = 0) / 3600
        if 'eeMaj_wise'    in data_catalog.colnames: err_wise  = np.nanmean([np.array(target['eeMaj_wise']).astype(float), np.array(target['eeMin_wise']).astype(float)], axis = 0) / 3600
    # Plot Locations
    if 'raStack_3pi'   in data_catalog.colnames: plt.errorbar((np.array(target   ['raStack_3pi'   ]).astype(float) - all_ras) * 3600, (np.array(target  ['decStack_3pi'  ]).astype(float) - all_decs) * 3600, xerr = np.array(target['raStackErr_3pi']).astype(float)      , yerr = np.array(target['decStackErr_3pi']).astype(float)    , alpha = 0.5, color = 'g',      fmt = 'o', label = '3PI'   )
    if 'ra_sdss'       in data_catalog.colnames: plt.errorbar((np.array(target   ['ra_sdss'       ]).astype(float) - all_ras) * 3600, (np.array(target  ['dec_sdss'      ]).astype(float) - all_decs) * 3600, xerr = np.array(target['raErr_sdss'    ]).astype(float)      , yerr = np.array(target['decErr_sdss'    ]).astype(float)    , alpha = 0.5, color = 'r',      fmt = 'o', label = 'SDSS'  )
    if 'RAJ2000_2mass' in data_catalog.colnames: plt.errorbar((np.array(target   ['RAJ2000_2mass' ]).astype(float) - all_ras) * 3600, (np.array(target  ['DEJ2000_2mass' ]).astype(float) - all_decs) * 3600, xerr = err_2mass * 3600                                      , yerr = err_2mass * 3600                                     , alpha = 0.5, color = 'orange', fmt = 'o', label = '2MASS' )
    if 'RAJ2000_wise'  in data_catalog.colnames: plt.errorbar((np.array(target   ['RAJ2000_wise'  ]).astype(float) - all_ras) * 3600, (np.array(target  ['DEJ2000_wise'  ]).astype(float) - all_decs) * 3600, xerr = err_wise  * 3600                                      , yerr = err_wise  * 3600                                     , alpha = 0.5, color = 'gray',   fmt = 'o', label = 'WISE'  )
    if 'RA_ICRS_gaia'  in data_catalog.colnames: plt.errorbar((np.array(target   ['RA_ICRS_gaia'  ]).astype(float) - all_ras) * 3600, (np.array(target  ['DE_ICRS_gaia'  ]).astype(float) - all_decs) * 3600, xerr = np.array(target['e_RA_ICRS_gaia']).astype(float)/1000 , yerr = np.array(target['e_DE_ICRS_gaia']).astype(float)/1000, alpha = 0.5, color = 'C0',     fmt = 'o', label = 'Gaia'  )
    plt.legend(loc = 'best')
    plt.xlim(- 1, + 1)
    plt.ylim(- 1, + 1)
    plt.gca().invert_xaxis()
    plt.xlabel('RA [arcsec]')
    plt.ylabel('DEC [arcsec]')

def make_plot(object_name, ra_deg, dec_deg, output_table, data_catalog, info_table, best_host, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, first_mjd, bright_mjd, search_radius, star_cut, Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, red_amplitude = np.nan, red_amplitude2 = np.nan, red_offset = np.nan, red_magnitude = np.nan, green_amplitude = np.nan, green_amplitude2 = np.nan, green_offset = np.nan, green_magnitude = np.nan, g_correct = 0, r_correct = 0):
    '''
    Create a diagnosis plot with the light curve and field image of a transient

    Parameters
    ---------------
    object_name           : Name of the transient
    ra_deg, dec_deg       : Coordiantes of the transient in degrees
    output_table          : Data with photometry
    data_catalog          : Astropy table with catalog data 
    info_table            : Table with all the output FLEET information
    best_host             : Index of best host
    host_radius           : The half-light radius of the best host in arcsec
    host_separation       : The transient-host separation in arcsec
    host_ra, host_dec     : Host coordinates
    host_Pcc              : The probability of chance coincidence for the best host
    host_magnitude        : The magnitude of the best host
    host_nature           : Probability of object being a galaxy
    first_mjd             : First MJD in either g or r from fit_linex()
    bright_mjd            : Brightest MJD in either g or r from fit_linex()
    search_radius         : For plot limit purposes
    star_cut              : maximum allowed probability of an object to be a star
    Dates_MMT             : Datetime format of hours
    Airmass_MMT           : Airmass of target at MMT
    SunElevation_MMT      : Sun's elevation at MMT 
    Dates_Magellan        : Datetime format of hours
    Airmass_Magellan      : Airmass of target at Magellan
    SunElevation_Magellan : Sun's elevation at Magellan 
    red_amplitude         : Parameters from fit_linex()
    red_amplitude2        : 
    red_offset            : 
    red_magnitude         : 
    green_amplitude       : 
    green_amplitude2      : 
    green_offset          : 
    green_magnitude       : 
    g_correct             : extinction value in g band
    r_correct             : extinction value in r band

    Output
    ---------------
    Saves an output to plots/ directory
    '''

    # Correct for extinction
    output_table_correct = extinct(output_table, g_correct, r_correct)

    # Create folder to store images, if it doesn't exist
    if len(glob.glob('plots')) == 0:
        os.system("mkdir plots")

    # Properties of All Objects
    separation         = flot(data_catalog['separation'         ])
    hosts_magnitudes   = flot(data_catalog['effective_magnitude'])
    output_nature      = flot(data_catalog['object_nature'      ])
    chance_coincidence = flot(data_catalog['chance_coincidence' ])
    objects_ra         = flot(data_catalog['ra_matched'         ])
    objects_dec        = flot(data_catalog['dec_matched'        ])
    halflight_radius   = flot(data_catalog['halflight_radius'   ])

    # Properties of closest object
    closest = np.nanargmin(separation)
    closest_separation  = separation[closest]
    closest_magnitude   = hosts_magnitudes[closest]
    closest_nature      = output_nature[closest]
    closest_coincidence = chance_coincidence[closest]

    # Peak Transient magnitude
    transient_magnitude = np.nanmin(output_table_correct['Mag'][(output_table['UL'] == 'False') & (output_table['Ignore'] == 'False')])

    # Zoom in if possible
    if (host_separation < search_radius * 30):
        search_radius_close = search_radius / 2

    plt.close('all')
    fig = plt.figure(figsize = (22,16))

    plot_nature_mag_distance(3, 4,  1, separation, hosts_magnitudes, output_nature, transient_magnitude, host_separation, host_magnitude, host_nature, closest_separation, closest_magnitude, closest_nature, search_radius_close)
    plot_host_mag_distance  (3, 4,  2, separation, hosts_magnitudes, output_nature, chance_coincidence, transient_magnitude, host_separation, host_magnitude, host_Pcc, closest_separation, closest_magnitude, closest_coincidence, search_radius_close, star_cut)
    plot_host_information   (3, 4,  3, ra_deg, dec_deg, info_table, host_radius, host_separation, host_magnitude, transient_magnitude)
    plot_observability      (3, 4,  4, ra_deg, dec_deg, Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan)
    plot_ra_dec_size        (3, 4,  5, objects_ra, objects_dec, ra_deg, dec_deg, output_nature, host_ra, host_dec, search_radius, halflight_radius, transient_magnitude)
    plot_ra_dec_magnitude   (3, 4,  6, objects_ra, objects_dec, ra_deg, dec_deg, hosts_magnitudes, host_ra, host_dec, search_radius, transient_magnitude)
    plot_field_image        (3, 4,  7, ra_deg, dec_deg, object_name, search_radius = search_radius)
    used_colors, used_sources = plot_lightcurve(3,4,8, output_table_correct, first_mjd, bright_mjd, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, full_range = False)
    plot_blackbody          (3, 4,  9, data_catalog, best_host)
    plot_coordinates        (3, 4, 10, ra_deg,dec_deg, data_catalog, closest)
    plot_legend             (3, 4, 11, used_colors, used_sources)
    plot_lightcurve         (3, 4, 12, output_table_correct, first_mjd, bright_mjd, full_range = True)

    plt.savefig('plots/%s_output.pdf'%object_name, bbox_inches = 'tight')
    plt.clf(); plt.close('all')


#data_catalog = catalog_operations('2018fd', table.Table.read('catalogs/2018fd.cat', format = 'ascii'), 137.651484362, 35.7217682668)
