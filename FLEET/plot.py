from FLEET.lightcurve import linex
import matplotlib.pyplot as plt
from matplotlib import gridspec
from bs4 import BeautifulSoup
from astropy import table
from io import BytesIO
from PIL import Image
import numpy as np
import requests
import urllib
import glob
import sys
import os

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

def plot_lightcurve(ax, output_table, first_mjd, bright_mjd, red_amplitude = np.nan, red_amplitude2 = np.nan, red_offset = np.nan, red_magnitude = np.nan, green_amplitude = np.nan, green_amplitude2 = np.nan, green_offset = np.nan, green_magnitude = np.nan, subtract_phase = 0, add_phase = 0, plot_model = True, full_range = False):
    '''
    Plot the light curve and model for a given transient

    Parameters
    ---------------
    ax            : axes on which to plot the light curve
    output_table  : Data with photometry
    bright_mjd    : Brightest MJD in either g or r from fit_linex()
    first_mjd     : First MJD in either g or r from fit_linex()

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
    ax.invert_yaxis()
    ax.set_ylim(np.nanmax(real_magnitudes) + 0.5, np.nanmin(real_magnitudes) - 0.5)

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
        ax.set_xlim(np.nanmin(all_times), np.nanmax(all_times))
    else:
        ax.set_xlim(phase_minimum - subtract_phase, phase_maximum + add_phase)

    # Plot Data (detections)
    ax.errorbar(all_phases[detection][is_det_ZTF]  , all_magnitudes[detection][is_det_ZTF]  , all_sigmas[detection][is_det_ZTF]  , ecolor = all_colors[detection][is_det_ZTF].astype('str')  ,     fmt = 'd'   , alpha = 0.8, ms = 0)
    ax.errorbar(all_phases[detection][is_det_Local] , all_magnitudes[detection][is_det_Local] , all_sigmas[detection][is_det_Local] , ecolor = all_colors[detection][is_det_Local].astype('str') ,     fmt = '*'   , alpha = 0.8, ms = 0)
    ax.errorbar(all_phases[detection][is_det_other], all_magnitudes[detection][is_det_other], all_sigmas[detection][is_det_other], ecolor = all_colors[detection][is_det_other].astype('str'),     fmt = '.'   , alpha = 0.8, ms = 0)
    ax.scatter (all_phases[detection][is_det_ZTF]  , all_magnitudes[detection][is_det_ZTF]  ,                                       color = all_colors[detection][is_det_ZTF].astype('str')  ,  marker = 'd'   , alpha = 0.8, s = 90)
    ax.scatter (all_phases[detection][is_det_Local] , all_magnitudes[detection][is_det_Local] ,                                       color = all_colors[detection][is_det_Local].astype('str') ,  marker = '*'   , alpha = 0.8, s = 90)
    ax.scatter (all_phases[detection][is_det_other], all_magnitudes[detection][is_det_other],                                       color = all_colors[detection][is_det_other].astype('str'),  marker = '.'   , alpha = 0.8, s = 90)
    # Plot Data (Upper limits)
    ax.scatter (all_phases[upper_limit], all_magnitudes[upper_limit], color = all_colors[upper_limit].astype('str'), marker = 'v', alpha = 0.5, s = 90)

    ### Plot Ignored Data
    ignore_alpha = 0.15
    # Select groups of Data (detectiond and upper limits)
    was_det_ZTF   = np.where(all_sources[detect_ignore] == 'ZTF')[0]
    was_det_Local  = np.where(all_sources[detect_ignore] == 'Local')[0]
    was_det_other = np.where([i not in ['ZTF', 'Local'] for i in all_sources[detect_ignore]])[0]
    # (detections)
    ax.errorbar(all_phases[detect_ignore][was_det_ZTF]  , all_magnitudes[detect_ignore][was_det_ZTF]  , all_sigmas[detect_ignore][was_det_ZTF]  , ecolor = all_colors[detect_ignore][was_det_ZTF].astype('str')  ,     fmt = 'd'   , alpha = ignore_alpha, ms = 0)
    ax.errorbar(all_phases[detect_ignore][was_det_Local] , all_magnitudes[detect_ignore][was_det_Local] , all_sigmas[detect_ignore][was_det_Local] , ecolor = all_colors[detect_ignore][was_det_Local].astype('str') ,     fmt = '*'   , alpha = ignore_alpha, ms = 0)
    ax.errorbar(all_phases[detect_ignore][was_det_other], all_magnitudes[detect_ignore][was_det_other], all_sigmas[detect_ignore][was_det_other], ecolor = all_colors[detect_ignore][was_det_other].astype('str'),     fmt = '.'   , alpha = ignore_alpha, ms = 0)
    ax.scatter (all_phases[detect_ignore][was_det_ZTF]  , all_magnitudes[detect_ignore][was_det_ZTF]  ,                                           color  = all_colors[detect_ignore][was_det_ZTF].astype('str')  ,  marker = 'd'   , alpha = ignore_alpha, s = 90)
    ax.scatter (all_phases[detect_ignore][was_det_Local] , all_magnitudes[detect_ignore][was_det_Local] ,                                           color  = all_colors[detect_ignore][was_det_Local].astype('str') ,  marker = '*'   , alpha = ignore_alpha, s = 90)
    ax.scatter (all_phases[detect_ignore][was_det_other], all_magnitudes[detect_ignore][was_det_other],                                           color  = all_colors[detect_ignore][was_det_other].astype('str'),  marker = '.'   , alpha = ignore_alpha, s = 90)
    # (Upper limits)
    ax.scatter (all_phases[upper_ignore], all_magnitudes[upper_ignore], color = all_colors[upper_ignore].astype('str'), marker = 'v', alpha = ignore_alpha, s = 90)

    #plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    #plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    if full_range:
        #plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        #plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        ax.tick_params(axis='both', top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.set_title('MJD', fontsize = 11)
    else:
        ax.set_xlabel('Days Since Brightest Point', fontsize = 11)
        ax.tick_params(axis='both', top=False, bottom=True, labeltop=False, labelbottom=True)

    ax.set_ylabel('Magnitude', fontsize = 11)
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)

    # Plot fits to the data
    if plot_model:
        model_time    = np.linspace(-80, 200, 1000)
        model_red     = linex(model_time, red_amplitude  , red_amplitude2  , red_offset  , red_magnitude  )
        model_green   = linex(model_time, green_amplitude, green_amplitude2, green_offset, green_magnitude)

        ax.plot(model_time, model_red    , color = 'r', linestyle = ':', linewidth = 0.5)
        ax.plot(model_time, model_green  , color = 'g', linestyle = ':', linewidth = 0.5)

    used_colors, used_sources = np.unique(all_filters), np.unique(all_sources)
    return used_colors, used_sources

def plot_legend(ax, used_colors, used_sources):
    '''
    Include a legend

    Parameters
    ---------------
    ax            : axes on which to plot the light curve
    used_colors   : Colors from plot_lightcurve() used
    used_sources  : Sources from plot_lightcurve() used
    '''

    # Select Plot Colors
    all_colors = plot_colors(used_colors)

    # Select groups of Data (detectiond and upper limits)
    is_det_ZTF   = np.where(used_sources == 'ZTF' )[0]
    is_det_FLWO  = np.where(used_sources == 'FLWO')[0]
    is_det_other = np.where([i not in ['ZTF', 'FLWO'] for i in used_sources])[0]

    if len(is_det_ZTF)   > 0: ax.scatter([], [], marker = 'd', alpha = 1.0, s = 90, color = 'k', label = 'ZTF'  )
    if len(is_det_FLWO)  > 0: ax.scatter([], [], marker = '*', alpha = 1.0, s = 90, color = 'k', label = 'Local')
    if len(is_det_other) > 0: ax.scatter([], [], marker = 'o', alpha = 1.0, s = 90, color = 'k', label = 'OSC'  )

    for i in range(len(used_colors)):
        ax.scatter([], [], marker = 'o', alpha = 1.0, s = 90, color = all_colors[i], label = used_colors[i])

    ax.legend(loc = 'center', frameon = False, fontsize = 11)
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    ax.axis('off')

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

def plot_field_image(ax, ra_deg, dec_deg, object_name, image_color = 'r', search_radius = 1.0, autoscale = 75):
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

    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    if img == '--':
        ax.annotate('Outside PS1 footprint', xy=(0,0))
        ax.set_xlim(-0.5, 1)
        ax.set_ylim(-1, 1)
    else:
        ax.imshow(np.array(img), cmap = 'viridis')

    ax.scatter(wcs_size*2, wcs_size*2, marker = '+', color = 'r')

def make_plot(object_name, ra_deg, dec_deg, output_table, first_mjd, bright_mjd, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, g_correct = 0, r_correct = 0):
    '''
    Create a diagnosis plot with the light curve and field image of a transient

    Parameters
    ---------------
    object_name      : Name of the transient
    ra_deg, dec_deg  : Coordiantes of the transient in degrees
    output_table     : Data with photometry
    first_mjd        : First MJD in either g or r from fit_linex()
    bright_mjd       : Brightest MJD in either g or r from fit_linex()
    red_amplitude    : Parameters from fit_linex()
    red_amplitude2   : 
    red_offset       : 
    red_magnitude    : 
    green_amplitude  : 
    green_amplitude2 : 
    green_offset     : 
    green_magnitude  : 
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

    # Plot Axes and Grid
    fig = plt.figure(figsize = (8, 5))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[2, 1], hspace = 0, wspace = 0) 

    ax0 = plt.subplot(gs[0])
    plot_lightcurve(ax0, output_table_correct, first_mjd, bright_mjd, full_range = True)

    ax2 = plt.subplot(gs[2])
    used_colors, used_sources = plot_lightcurve(ax2, output_table_correct, first_mjd, bright_mjd, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, full_range = False)

    ax3 = plt.subplot(gs[3])
    plot_legend(ax3, used_colors, used_sources)

    ax1 = plt.subplot(gs[1])
    plot_field_image(ax1, ra_deg, dec_deg, object_name)

    plt.savefig('plots/%s.pdf'%object_name, bbox_inches = 'tight')
    plt.clf(); plt.close('all')

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





