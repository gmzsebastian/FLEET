from FLEET.plot import make_plot, calculate_observability, redshift_magnitude, quick_plot
from FLEET.catalog import get_catalog, catalog_operations, get_best_host, get_extinction
from FLEET.transient import get_transient_info, generate_lightcurve, ignore_data
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from FLEET.lightcurve import fit_linex
from astropy import table
import pkg_resources
import numpy as np
import glob
import os

def create_features(object_name, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, late_color, bright_mjd, first_mjd, green_brightest, red_brightest, host_radius, host_separation, host_Pcc, host_magnitude, chi2, hostless_cut = 0.1, redshift = np.nan):
    '''
    Create an astropy table with all the relevant features to feed into the classifier

    Parameters
    ---------------
    red_amplitude  , red_amplitude2  , red_offset  , red_magnitude   : Best fit parameters for r-band light curve
    green_amplitude, green_amplitude2, green_offset, green_magnitude : Best fit parameters for g-band light curve
    model_color     : g-r color from the model light curves at the time of brightest measured magnitude
    late_color      : g-r color at late_phase (40 default) days after peak
    bright_mjd      : Brightest MJD in either g or r
    first_mjd       : First MJD in either g or r
    green_brightest : Brightest measured g-band magnitude
    red_brightest   : Brightest measured r-band magnitude
    hostless_cut    : Only consider hosts with a Pcc lower than this
    redshift        : Provided redshift
    object_name     : name of the object
    host_radius     : The half-light radius of the best host in arcsec
    host_separation : The transient-host separation in arcsec
    host_Pcc        : The probability of chance coincidence for the best host
    host_magnitude  : The magnitude of the best host
    chi2            : Chi squared value

    Return
    ---------------
    An astropy table with all the relevant features

    '''

    # Calculate time from discovery to peak
    delta_time = bright_mjd - first_mjd

    # Obtain Features
    if host_Pcc <= hostless_cut:
        input_separation  = host_separation
        input_size        = host_radius
        input_magnitude   = host_magnitude
        normal_separation = input_separation / input_size
    else:
        input_separation  = 0
        input_size        = 0
        normal_separation = 0
        input_magnitude   = 23.2 # r-band limit from 3PI

    # Calculate deltamag (Host - Transient)
    deltamag_red   = input_magnitude - red_brightest
    deltamag_green = input_magnitude - green_brightest

    # Redshift and Absolute Magnitude
    if np.isfinite(red_brightest):
        absmag, _ = redshift_magnitude(red_brightest, np.max([redshift, 0]))
        if np.isinf(absmag): absmag = 0
    else:
        absmag = np.nan

    features       = np.array(                     [object_name  , red_amplitude  , red_amplitude2  , red_offset  , red_magnitude  , green_amplitude  , green_amplitude2  , green_offset  , green_magnitude  , delta_time  , input_separation  , input_size  ,  normal_separation  , deltamag_red  ,  deltamag_green   , model_color ,  late_color , host_Pcc ,   redshift,   absmag ,     chi2 ])
    features_table = table.Table(features, names = ['object_name', 'red_amplitude', 'red_amplitude2', 'red_offset', 'red_magnitude', 'green_amplitude', 'green_amplitude2', 'green_offset', 'green_magnitude', 'delta_time', 'input_separation', 'input_size',  'normal_separation', 'deltamag_red', 'deltamag_green',  'model_color', 'late_color', 'Pcc'    , 'redshift',  'absmag',    'chi2'],
                                           dtype = ['S25'        , 'float64'      ,        'float64',    'float64',       'float64',         'float64',          'float64',      'float64',         'float64',    'float64',          'float64',    'float64',            'float64',      'float64',        'float64',      'float64',    'float64', 'float64',  'float64', 'float64', 'float64'])

    return features_table

def create_training_testing(object_name, features_table, training_days = 20, model = 'single', clean = 0, feature_set = 13, sorting_state = 42, SMOTE_state = 42, clf_state = 42, n_estimators = 100, max_depth = 7, hostless_cut = 0.1):
    '''
    Import the training set and modify the features_table according to the model
    parameters specified.

    Parameters
    -------------
    object_name    : Name of the object to exclude from training set
    features_table : Astropy table with all the features of the new transient
    training_days  : What data set to use for training
    model          : Which model to use for training, single or double
    clean          : Clean hostless transients?
    feature_set    : Which feature set to use
    sorting_state  : Seed number for list sorter
    SMOTE_state    : Seed number for SMOTE
    clf_state      : Seed number for classifier
    n_estimators   : Number of trees
    max_depth      : Depth of trees
    hostless_cut   : Only consider hosts with a Pcc lower than this

    Return
    ---------------
    Predicted Probability to be ['AGN','SLSN-I','SLSN-II','SNII','SNIIb','SNIIn','SNIa','SNIbc','Star','TDE']
    '''

    # Import Data
    table_name = pkg_resources.resource_filename(__name__, 'training_set/center_table_%s_%s.txt'%(training_days, model))
    training_table_in = table.Table.read(table_name, format = 'ascii')

    # Remove bad objects from training sample
    bad = ['2020cui','2019lwy','2019cvi','2018jsc','2005bf' ,'2005gi' ,'2007ib' ,'2008aq' ,
           '2008ax' ,'2009N'  ,'2010id' ,'2012aw' ,'2013ai' ,'2013am' ,'2013bu' ,'2013ej' ,
           '2013fs' ,'2016X'  ,'2018cyg','2018epm','2018fjw','2018fii','2018fuw','2018gvt',
           '2018imj','2018lcd','2019B'  ,'2019bvq','2019cda','2019ci' ,'2019dok','2019gaf',
           '2019gqk','2019hau','2019iex','2019keo','2019lkw','2019oa' ,'2019otb','2019pjs',
           '2019sjx','2019tqb','2019wbg','2020ekk', object_name]
    good = [i not in bad for i in training_table_in['object_name']]
    training_table_in = training_table_in[good]

    # Shuffle Order of Table
    order = np.arange(len(training_table_in))
    np.random.seed(sorting_state)
    np.random.shuffle(order)
    training_table = training_table_in[order]

    # Select Only Clean Data
    if clean == 0:
        clean_training = training_table[np.isfinite(training_table['red_amplitude']) & np.isfinite(training_table['Pcc'])]
    if clean == 1:
        clean_training = training_table[np.isfinite(training_table['red_amplitude']) & np.isfinite(training_table['Pcc']) & (training_table['Pcc'] <= hostless_cut)]

    # Select Features
    if feature_set == 0 :  use_features = ['red_amplitude',                   'green_amplitude',                                      'normal_separation' , 'deltamag_red', 'deltamag_green'                              ]
    if feature_set == 1 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green'                              ]
    if feature_set == 2 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   ,                       'deltamag_red', 'deltamag_green'                              ]
    if feature_set == 3 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                                 ]
    if feature_set == 4 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   ,                                                                                     ]
    if feature_set == 5 :  use_features = ['red_amplitude',                   'green_amplitude',                                                                                                                          ]
    if feature_set == 6 :  use_features = [                                                                                           'normal_separation' , 'deltamag_red', 'deltamag_green'                              ]
    if feature_set == 7 :  use_features = ['red_amplitude',                   'green_amplitude',                                      'normal_separation'                                                                 ]
    if feature_set == 8 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'color'                    ]
    if feature_set == 9 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'color'                    ]
    if feature_set == 10:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'Pcc'       ]
    if feature_set == 11:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'redshift'  ]
    if feature_set == 12:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'absmag'    ]
    if feature_set == 13:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'model_color'              ]
    if feature_set == 14:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'              ]
    if feature_set == 15:  use_features = ['red_amplitude',                   'green_amplitude',                                      'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'              ]
    if feature_set == 16:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'model_color', 'redshift'  ]
    if feature_set == 17:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green'                , 'redshift'  ]
    if feature_set == 18:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'input_separation'  , 'deltamag_red', 'deltamag_green'                , 'input_size']
    if feature_set == 19:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'input_separation'  , 'deltamag_red', 'deltamag_green' , 'model_color', 'input_size']
    if feature_set == 20:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'input_separation'  , 'deltamag_red', 'deltamag_green'                , 'input_size']
    if feature_set == 21:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'input_separation'  , 'deltamag_red', 'deltamag_green' , 'model_color', 'input_size']
    if feature_set == 22:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color', 'redshift'  ]
    if feature_set == 23:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'              , 'chi2']
    if feature_set == 24:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color', 'redshift'  , 'chi2']
    if feature_set == 25:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'              , 'late_color']
    if feature_set == 26:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color', 'redshift'  , 'late_color']

    # If using the 'double' model, add the W2 parameter
    if model == 'double':
        use_features += ['red_amplitude2','green_amplitude2']

    # Create array with Training and Testing data
    training_data = np.array(clean_training[use_features].to_pandas())
    testing_data  = np.array(features_table[use_features].to_pandas())

    # Get names of objects and classes
    training_class_in = np.array(clean_training['class'])

    # Group Transients into groups
    training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
    training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
    training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'

    training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'SNIbc'
    training_class_in[np.where(training_class_in == 'SNIb'   )] = 'SNIbc'
    training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'SNIbc'
    training_class_in[np.where(training_class_in == 'SNIc'   )] = 'SNIbc'
    training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'SNIbc'

    training_class_in[np.where(training_class_in == 'SNII'   )] = 'SNII'
    training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'SNII'

    classes_names = {'AGN'     : 0, 'SLSN-I' : 1, 'SLSN-II' : 2, 'SNII'    : 3, 'SNIIb'   : 4,
                     'SNIIn'   : 5, 'SNIa'   : 6, 'SNIbc'   : 7, 'Star'    : 8, 'TDE'     : 9}
    training_class = np.array([classes_names[i] for i in training_class_in]).astype(int)

    # SMOTE the data
    sampler = SMOTE(random_state=SMOTE_state)
    data_train_smote, class_train_smote = sampler.fit_resample(training_data, training_class)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=clf_state)
    clf.fit(data_train_smote, class_train_smote)

    # Predict Excluded Object
    predicted_probability = 100 * clf.predict_proba(testing_data)

    return predicted_probability

def predict_SLSN(object_name_in = '', ra_in = '', dec_in = '', redshift = np.nan, acceptance_radius = 3, import_ZTF = True, import_OSC = True, import_local = True, import_lightcurve = True, reimport_catalog = False, search_radius = 1.0, dust_map = 'SFD', Pcc_filter = 'i', Pcc_filter_alternative = 'r', star_separation = 1, star_cut = 0.1, date_range = np.inf, late_phase = 40, n_walkers = 50, n_steps = 500, n_cores = 1, model = 'single', training_days = 20, hostless_cut = 0.1, sorting_state = 42, clean = 0, SMOTE_state = 42, clf_state = 42, n_estimators = 100, max_depth = 7, feature_set = 13, neighbors = 20, recalculate_nature = False, classifier = '', n_samples = 3, object_class = '', plot_lightcurve = False, do_observability = False, save_features = False, overwrite_features = False):
    '''
    Main Function to predict the probability of an object to be a Superluminous Supernovae
    using the training set provided and a random forest algorithim. The function will query
    the TNS, ZTF, and the OSC for data for the transient. And SDSS and 3PI for catalog data.

    Parameters
    -------------
    object_name_in          : name of the transient (either ZTF, TNS, or other)
    ra_in                   : R.A. in degrees or hms format
    dec_in                  : Dec. in degrees or dms format
    redshift                : redshift, only required for redshift classifier
    acceptance_radius       : match objects for catalog cross-matching; in arcseconds
    import_ZTF              : Import ZTF data (True) or read existing file (False)
    import_OSC              : Import OSC data (True) or read existing file (False)
    import_local            : Import local data from ./photometry directory
    import_lightcurve       : Regenerate existing lightcurve file (True) or read
                              the existing out from ./lightcurves (False)
    reimport_catalog        : Overwrite the existing 3PI/SDSS catalog
    search_radius           : Search radius in arcminutes for the 3PI/SDSS catalog
    dust_map                : 'SF' or 'SFD', to query Schlafy and Finkbeiner 2011
                              or Schlafy, Finkbeiner and Davis 1998.
                              set to 'none' to not correct for extinction
    Pcc_filter              : The effective magnitude, radius, and Pcc
                              are calculated in this filter.
    Pcc_filter_alternative  : If Pcc_filter is not found, use this one
                              as an acceptable alternative.
    star_separation         : A star needs to be this close to be matched
                              to a transient [in Arcsec]
    star_cut                : maximum allowed probability of an object to be a star
    date_range              : Maximum number of light curve days from the first 
                              detection to use in fitting the light curve
    late_phase              : Phase at which to calculate the late color
    n_walkers               : Number of walkers for MCMC
    n_steps                 : Number of steps for MCMC
    n_cores                 : Number of cores for MCMC
    model                   : 'single' or 'double' for the function to fit to the 
                              lightcurve. Use later one for full light curves.
    training_days           : Which training set to use for classification
    hostless_cut            : Only consider hosts with a Pcc lower than this
    sorting_state           : Seed number for list sorter
    clean                   : 0 keeps all objects, and 1 removed objects that do not
                              have a detected host
    SMOTE_state             : Seed number for SMOTE
    clf_state               : Seed number for classifier
    n_estimators            : Number of trees for random forest
    max_depth               : Depth of trees for random forest
    feature_set             : Set of features to use
    neighbors               : neighbors to use for star/galaxy separator
    recalculate_nature      : Overwrite existing Nature column?
    classifier              : Pick the classifier to use based on the available information
                              either 'quick', 'redshift', 'host', 'late', or 'all'. If empty
                              default (or specified) values will be used.
    n_samples               : Number of random seeds to use, only for the 'all' classifier
    object_class            : Transient type, to overwrite any existing classes
    plot_lightcurve         : Save an output plot with the light curve and PS1 image?
    do_observability        : Calculate Observavility from Magellan and MMT?
    save_features           : Save the features table to a file
    overwrite_features      : Overwrite features?

    Returns
    ---------------
    Predicted Probability to be ['AGN','SLSN-I','SLSN-II','SNII','SNIIb','SNIIn','SNIa','SNIbc','Star','TDE']

    '''
    print('\n################# FLEET #################')

    # Empty Features for if search failed
    if save_features:
        filename   = '%s_%s/center_table_%s_%s_%s.txt'%(int(float(date_range)), model, int(float(date_range)), model, object_name_in)
    features       = np.array(                     [object_name_in,          np.nan,           np.nan,       np.nan,          np.nan,            np.nan,             np.nan,         np.nan,            np.nan,       np.nan,             np.nan,       np.nan,               np.nan,         np.nan,           np.nan,         np.nan,        np.nan,    np.nan,     np.nan,    np.nan,    np.nan])
    features_table = table.Table(features, names = ['object_name' , 'red_amplitude', 'red_amplitude2', 'red_offset', 'red_magnitude', 'green_amplitude', 'green_amplitude2', 'green_offset', 'green_magnitude', 'delta_time', 'input_separation', 'input_size',  'normal_separation', 'deltamag_red', 'deltamag_green',  'model_color',  'late_color',     'Pcc', 'redshift',  'absmag',    'chi2'],
                                           dtype = ['S25'         , 'float64'      ,        'float64',    'float64',       'float64',         'float64',          'float64',      'float64',         'float64',    'float64',          'float64',    'float64',            'float64',      'float64',        'float64',      'float64',     'float64', 'float64',  'float64', 'float64', 'float64'])

    # If the features file already exists, don't overwrite it
    if save_features:
        if not overwrite_features :
            if len(glob.glob(filename)) > 0:
                print('exists')
                return table.Table()

    ##### Basic transient info #####
    ra_deg, dec_deg, transient_source, object_name, ztf_data, ztf_name, tns_name, object_class, osc_data = get_transient_info(object_name_in, ra_in, dec_in, object_class, acceptance_radius, import_ZTF, import_OSC, import_lightcurve)
    if ra_deg == '--': return table.Table()
    print('%s %s %s'%(object_name, ra_deg, dec_deg))
    if dec_deg <= -32:
        print('dec = %s, too low for SDSS or 3PI'%dec_deg)
        if save_features : features_table.write(filename, format = 'ascii', overwrite = True)
        return table.Table()

    ##### Lightcurve data #####
    output_table = generate_lightcurve(ztf_data, osc_data, object_name, ztf_name, tns_name, import_lightcurve, import_local)
    # Ignore selected patches
    output_table = ignore_data(object_name, output_table)

    if len(output_table) == 0:
        print('No data in lightcurve')
        if save_features : features_table.write(filename, format = 'ascii', overwrite = True)
        return table.Table()
    if np.sum((output_table['UL'] == 'False') & (output_table['Ignore'] == 'False')) == 0:
        print('No useable data in lightcurve')
        if save_features : features_table.write(filename, format = 'ascii', overwrite = True)
        return table.Table()

    # Extinction
    g_correct, r_correct = get_extinction(ra_deg, dec_deg, dust_map)

    ##### Fit Lightcurve #####
    red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, late_color, bright_mjd, first_mjd, green_brightest, red_brightest, chi2 = fit_linex(output_table, date_range, n_walkers, n_steps, n_cores, model, g_correct, r_correct, late_phase)
    if np.isnan(red_amplitude):
        if plot_lightcurve:
            first_mjd  = np.nanmin(np.array(output_table['MJD']).astype(float))
            bright_mjd = output_table['MJD'][np.nanargmin(output_table['Mag'])]
            quick_plot(object_name, ra_deg, dec_deg, output_table, first_mjd, bright_mjd, full_range = True)
        # If running normal FLEET, stop here
        if save_features == False:
            return table.Table()

    ##### Catalog data #####
    data_catalog_out = get_catalog(object_name, ra_deg, dec_deg, search_radius, dust_map, reimport_catalog)
    if len(data_catalog_out) == 0:
        print('No data found in SDSS or 3PI')
        # If running normal FLEET, stop here
        if save_features == False:
            return table.Table()

    ##### Catalog Operations #####
    print('Creating Catalog ...')
    data_catalog = catalog_operations(object_name, data_catalog_out, ra_deg, dec_deg, Pcc_filter, Pcc_filter_alternative, neighbors, recalculate_nature)

    ##### Find the Best host #####
    host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, photoz, photoz_err, specz, specz_err, best_host = get_best_host(data_catalog, star_separation, star_cut)

    ##### Use Appropriate Redshift #####
    if np.isfinite(float(redshift)):
        # User specified redshift
        use_redshift   = float(redshift)
        redshift_label = 'specz'
    elif np.isfinite(float(specz)):
        # Spectroscopic Redshift
        use_redshift   = float(specz)
        redshift_label = 'specz'
    elif np.isfinite(float(photoz)):
        # Photometric Redshift
        use_redshift   = float(photoz)
        redshift_label = 'photoz'
    else:
        # No Redshift
        use_redshift   = np.nan
        redshift_label = 'none'

    ##### Get Features #####
    features_table = create_features(object_name, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, late_color, bright_mjd, first_mjd, green_brightest, red_brightest, host_radius, host_separation, host_Pcc, host_magnitude, chi2, hostless_cut, use_redshift)

    ##### Save Features for training #####
    if save_features:
        # Save output
        foldername = '%s_%s'%(int(float(date_range)), model)
        if len(glob.glob(foldername)) == 0:
            os.system('mkdir %s'%foldername)
        features_table.write(filename, format = 'ascii', overwrite = True)
        if plot_lightcurve:
            quick_plot(object_name, ra_deg, dec_deg, output_table, first_mjd, bright_mjd, g_correct, r_correct, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, full_range = False)
        return features_table

    # Empty variables (9 is the number of classes)
    quick_probability_average    = np.nan * np.ones(9)
    quick_probability_std        = np.nan * np.ones(9)
    late_probability_average     = np.nan * np.ones(9)
    late_probability_std         = np.nan * np.ones(9)
    redshift_probability_average = np.nan * np.ones(9)
    redshift_probability_std     = np.nan * np.ones(9)
    host_probability_average     = np.nan * np.ones(9)
    host_probability_std         = np.nan * np.ones(9)

    # Classifier Parameters
    if (classifier == '') or ('slsn' in classifier):
        training_days_quick    = 20
        model_quick            = 'single'
        clean_quick            = 0
        feature_set_quick      = 14
        max_depth_quick        = 11

        training_days_late     = 70
        model_late             = 'double'
        clean_late             = 0
        feature_set_late       = 14
        max_depth_late         = 11

        training_days_redshift = 20
        model_redshift         = 'single'
        clean_redshift         = 0
        feature_set_redshift   = 22
        max_depth_redshift     = 11

        training_days_host     = 70
        model_host             = 'double'
        clean_host             = 1
        feature_set_host       = 14
        max_depth_host         = 11

    elif ('tde' in classifier):
        training_days_quick    = 20
        model_quick            = 'single'
        clean_quick            = 0
        feature_set_quick      = 14
        max_depth_quick        = 14

        training_days_late     = 40
        model_late             = 'double'
        clean_late             = 0
        feature_set_late       = 14
        max_depth_late         = 14

        training_days_redshift = 20
        model_redshift         = 'single'
        clean_redshift         = 0
        feature_set_redshift   = 22
        max_depth_redshift     = 14

        training_days_host     = 40
        model_host             = 'double'
        clean_host             = 1
        feature_set_host       = 14
        max_depth_host         = 14

    ##### Run Classifier #####
    print('Classifying ...')
    if classifier == '':
        quick_probability_average    = create_training_testing(object_name, features_table, training_days, model, clean, feature_set, sorting_state, SMOTE_state, clf_state, n_estimators, max_depth, hostless_cut)[0]
    elif ('quick' in classifier):
        quick_probability_average    = create_training_testing(object_name, features_table, training_days = training_days_quick, model = model_quick, clean = clean_quick, feature_set = feature_set_quick, max_depth = max_depth_quick)[0]
    elif ('late' in classifier):
        late_probability_average     = create_training_testing(object_name, features_table, training_days = training_days_late, model = model_late, clean = clean_late, feature_set = feature_set_late, max_depth = max_depth_late)[0]
    elif ('redshift' in classifier):
        redshift_probability_average = create_training_testing(object_name, features_table, training_days = training_days_redshift, model = model_redshift, clean = clean_redshift, feature_set = feature_set_redshift, max_depth = max_depth_redshift)[0]
    elif ('host' in classifier):
        host_probability_average     = create_training_testing(object_name, features_table, training_days = training_days_host, model = model_host, clean = clean_host, feature_set = feature_set_host, max_depth = max_depth_host)[0]

    ##### All in one classifier for SLSN #####
    elif 'all' in classifier:
        # Quick Classifier
        for i in range(n_samples):
            quick_probability_n = create_training_testing(object_name, features_table, training_days = training_days_quick, model = model_quick, clean = clean_quick, feature_set = feature_set_quick, max_depth = max_depth_quick, clf_state = int(39 + i))
            if i == 0:
                quick_probability = quick_probability_n
            else:
                quick_probability = np.vstack([quick_probability, quick_probability_n])
        quick_probability_average = np.average(quick_probability, axis = 0)
        quick_probability_std     = np.std    (quick_probability, axis = 0)

        # Late Classifier
        for i in range(n_samples):
            late_probability_n = create_training_testing(object_name, features_table, training_days = training_days_late, model = model_late, clean = clean_late, feature_set = feature_set_late, max_depth = max_depth_late, clf_state = int(39 + i))
            if i == 0:
                late_probability = late_probability_n
            else:
                late_probability = np.vstack([late_probability, late_probability_n])
        late_probability_average = np.average(late_probability, axis = 0)
        late_probability_std     = np.std    (late_probability, axis = 0)

        # Redsfhit Classifier
        if np.isfinite(np.float(use_redshift)) & (host_Pcc <= hostless_cut):
            for i in range(n_samples):
                redshift_probability_n = create_training_testing(object_name, features_table, training_days = training_days_redshift, model = model_redshift, clean = clean_redshift, feature_set = feature_set_redshift, max_depth = max_depth_redshift, clf_state = int(39 + i))
                if i == 0:
                    redshift_probability = redshift_probability_n
                else:
                    redshift_probability = np.vstack([redshift_probability, redshift_probability_n])
            redshift_probability_average = np.average(redshift_probability, axis = 0)
            redshift_probability_std     = np.std    (redshift_probability, axis = 0)
        else:
            redshift_probability_average = np.nan * np.ones(len(quick_probability_average))
            redshift_probability_std     = np.nan * np.ones(len(quick_probability_average))

        # Host Classifier
        if (host_Pcc <= hostless_cut):
            for i in range(n_samples):
                host_probability_n = create_training_testing(object_name, features_table, training_days = training_days_host, model = model_host, clean = clean_host, feature_set = feature_set_host, max_depth = max_depth_host, clf_state = int(39 + i))
                if i == 0:
                    host_probability = host_probability_n
                else:
                    host_probability = np.vstack([host_probability, host_probability_n])
            host_probability_average = np.average(host_probability, axis = 0)
            host_probability_std     = np.std    (host_probability, axis = 0)
        else:
            host_probability_average = np.nan * np.ones(len(quick_probability_average))
            host_probability_std     = np.nan * np.ones(len(quick_probability_average))

    # Calculate Time Span
    detections = (np.array(output_table['UL']) == b'False') & (np.array(output_table['Ignore']) == b'False') & ((output_table['Filter'] == 'r') | (output_table['Filter'] == 'g'))
    time_span  = np.nanmax(output_table['MJD'][detections]) - np.nanmin(output_table['MJD'][detections])

    Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, MMT_observable, Magellan_observable = calculate_observability(ra_deg, dec_deg, do_observability)

    # Output Array
    info_data  = np.array([object_name, ztf_name, tns_name, object_class, ra_deg, dec_deg, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, photoz, photoz_err, specz, specz_err, use_redshift, redshift_label, classifier, time_span, chi2, MMT_observable, Magellan_observable, *quick_probability_average, *quick_probability_std, *late_probability_average, *late_probability_std, *redshift_probability_average, *redshift_probability_std, *host_probability_average, *host_probability_std])
    info_names = ['object_name'       ,'ztf_name'             ,'tns_name'             ,'object_class'         ,'ra_deg'               ,'dec_deg'              ,'host_radius'          ,'host_separation'      ,'host_ra'              ,'host_dec'          ,'host_Pcc'           ,'host_magnitude',
                  'host_nature'       ,'photoz'               ,'photoz_err'           ,'specz'                ,'specz_err'            ,'use_redshift'         ,'redshift_label'       ,'classifier'           ,'time_span'            ,'chi2'              ,'MMT_observable'     ,'Magellan_observable',
                  'P_quick_AGN'       ,'P_quick_SLSNI'        ,'P_quick_SLSNII'       ,'P_quick_SNII'         ,'P_quick_SNIIb'        ,'P_quick_SNIIn'        ,'P_quick_SNIa'         ,'P_quick_SNIbc'        ,'P_quick_Star'         ,'P_quick_TDE'       ,
                  'P_quick_AGN_std'   ,'P_quick_SLSNI_std'    ,'P_quick_SLSNII_std'   ,'P_quick_SNII_std'     ,'P_quick_SNIIb_std'    ,'P_quick_SNIIn_std'    ,'P_quick_SNIa_std'     ,'P_quick_SNIbc_std'    ,'P_quick_Star_std'     ,'P_quick_TDE_std'   ,
                  'P_late_AGN'        ,'P_late_SLSNI'         ,'P_late_SLSNII'        ,'P_late_SNII'          ,'P_late_SNIIb'         ,'P_late_SNIIn'         ,'P_late_SNIa'          ,'P_late_SNIbc'         ,'P_late_Star'          ,'P_late_TDE'        ,
                  'P_late_AGN_std'    ,'P_late_SLSNI_std'     ,'P_late_SLSNII_std'    ,'P_late_SNII_std'      ,'P_late_SNIIb_std'     ,'P_late_SNIIn_std'     ,'P_late_SNIa_std'      ,'P_late_SNIbc_std'     ,'P_late_Star_std'      ,'P_late_TDE_std'    ,
                  'P_redshift_AGN'    ,'P_redshift_SLSNI'     ,'P_redshift_SLSNII'    ,'P_redshift_SNII'      ,'P_redshift_SNIIb'     ,'P_redshift_SNIIn'     ,'P_redshift_SNIa'      ,'P_redshift_SNIbc'     ,'P_redshift_Star'      ,'P_redshift_TDE'    ,
                  'P_redshift_AGN_std','P_redshift_SLSNI_std' ,'P_redshift_SLSNII_std','P_redshift_SNII_std'  ,'P_redshift_SNIIb_std' ,'P_redshift_SNIIn_std' ,'P_redshift_SNIa_std'  ,'P_redshift_SNIbc_std' ,'P_redshift_Star_std'  ,'P_redshift_TDE_std',
                  'P_host_AGN'        ,'P_host_SLSNI'         ,'P_host_SLSNII'        ,'P_host_SNII'          ,'P_host_SNIIb'         ,'P_host_SNIIn'         ,'P_host_SNIa'          ,'P_host_SNIbc'         ,'P_host_Star'          ,'P_host_TDE'        ,
                  'P_host_AGN_std'    ,'P_host_SLSNI_std'     ,'P_host_SLSNII_std'    ,'P_host_SNII_std'      ,'P_host_SNIIb_std'     ,'P_host_SNIIn_std'     ,'P_host_SNIa_std'      ,'P_host_SNIbc_std'     ,'P_host_Star_std'      ,'P_host_TDE_std'    ]
    info_table = table.Table(info_data, names = info_names)

    print('Plotting ...')
    if plot_lightcurve:
        make_plot(object_name, ra_deg, dec_deg, output_table, data_catalog, info_table, best_host, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, first_mjd, bright_mjd, search_radius, star_cut, Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, chi2, g_correct, r_correct)
    
    return info_table

def predict_Host(object_name_in = '', ra_in = '', dec_in = '', redshift = np.nan, acceptance_radius = 3, import_ZTF = False, import_OSC = False, import_local = False, import_lightcurve = True, reimport_catalog = False,
                 search_radius = 1.0, dust_map = 'SFD', Pcc_filter = 'i', Pcc_filter_alternative = 'r', star_separation = 1, star_cut = 0.1, hostless_cut = 0.1,
                 neighbors = 20, recalculate_nature = False, object_class = '', plot_lightcurve = False, do_observability = False):
    '''
    Main Function to predict the probability of an object to be a Superluminous Supernovae
    using the training set provided and a random forest algorithim. The function will query
    the TNS, ZTF, and the OSC for data for the transient. And SDSS and 3PI for catalog data.

    Parameters
    -------------
    object_name_in          : name of the transient (either ZTF, TNS, or other)
    ra_in                   : R.A. in degrees or hms format
    dec_in                  : Dec. in degrees or dms format
    redshift                : redshift, only required for redshift classifier
    acceptance_radius       : match objects for catalog cross-matching; in arcseconds
    import_ZTF              : Import ZTF data (True) or read existing file (False)
    import_OSC              : Import OSC data (True) or read existing file (False)
    import_local            : Import local data from ./photometry directory
    import_lightcurve       : Regenerate existing lightcurve file (True) or read
                              the existing out from ./lightcurves (False)
    reimport_catalog        : Overwrite the existing 3PI/SDSS catalog
    search_radius           : Search radius in arcminutes for the 3PI/SDSS catalog
    dust_map                : 'SF' or 'SFD', to query Schlafy and Finkbeiner 2011
                              or Schlafy, Finkbeiner and Davis 1998.
                              set to 'none' to not correct for extinction
    Pcc_filter              : The effective magnitude, radius, and Pcc
                              are calculated in this filter.
    Pcc_filter_alternative  : If Pcc_filter is not found, use this one
                              as an acceptable alternative.
    star_separation         : A star needs to be this close to be matched
                              to a transient [in Arcsec]
    star_cut                : maximum allowed probability of an object to be a star
    hostless_cut            : Only consider hosts with a Pcc lower than this
    neighbors               : neighbors to use for star/galaxy separator
    recalculate_nature      : Overwrite existing Nature column?
    object_class            : Transient type, to overwrite any existing classes
    plot_lightcurve         : Save an output plot with the light curve and PS1 image?
    do_observability        : Calculate Observavility from Magellan and MMT?

    Returns
    ---------------
    Predicted Probability to be ['AGN','SLSN-I','SLSN-II','SNII','SNIIb','SNIIn','SNIa','SNIbc','Star','TDE']

    '''
    print('\n################# FLEET #################')

    ##### Basic transient info #####
    ra_deg, dec_deg, transient_source, object_name, ztf_data, ztf_name, tns_name, object_class, osc_data = get_transient_info(object_name_in, ra_in, dec_in, object_class, acceptance_radius, import_ZTF, import_OSC, import_lightcurve)
    if ra_deg == '--': return table.Table()
    print('%s %s %s'%(object_name, ra_deg, dec_deg))
    if dec_deg <= -32:
        print('dec = %s, too low for SDSS or 3PI'%dec_deg)
        return table.Table()

    ##### Catalog data #####
    data_catalog_out = get_catalog(object_name, ra_deg, dec_deg, search_radius, dust_map, reimport_catalog)
    if len(data_catalog_out) == 0:
        print('No data found in SDSS or 3PI')
        return table.Table()

    ##### Catalog Operations #####
    print('Creating Catalog ...')
    data_catalog = catalog_operations(object_name, data_catalog_out, ra_deg, dec_deg, Pcc_filter, Pcc_filter_alternative, neighbors, recalculate_nature)

    ##### Find the Best host #####
    host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, photoz, photoz_err, specz, specz_err, best_host = get_best_host(data_catalog, star_separation, star_cut)

    ##### Use Appropriate Redshift #####
    if np.isfinite(float(redshift)):
        # User specified redshift
        use_redshift   = float(redshift)
        redshift_label = 'specz'
    elif np.isfinite(float(specz)):
        # Spectroscopic Redshift
        use_redshift   = float(specz)
        redshift_label = 'specz'
    elif np.isfinite(float(photoz)):
        # Photometric Redshift
        use_redshift   = float(photoz)
        redshift_label = 'photoz'
    else:
        # No Redshift
        use_redshift   = np.nan
        redshift_label = 'none'

    ##### Get Features #####
    features_table = create_features(object_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, host_radius, host_separation, host_Pcc, host_magnitude, np.nan, hostless_cut, use_redshift)

    # Calculate Observability
    Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, MMT_observable, Magellan_observable = calculate_observability(ra_deg, dec_deg, do_observability)

    # Output Array
    info_data  = np.array([object_name, ztf_name, tns_name, object_class, ra_deg, dec_deg, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, photoz, photoz_err, specz, specz_err, use_redshift, redshift_label])
    info_names = ['object_name'       ,'ztf_name'             ,'tns_name'             ,'object_class'         ,'ra_deg'               ,'dec_deg'              ,'host_radius'          ,'host_separation'      ,'host_ra'              ,'host_dec'          ,'host_Pcc'           ,'host_magnitude',
                  'host_nature'       ,'photoz'               ,'photoz_err'           ,'specz'                ,'specz_err'            ,'use_redshift'         ,'redshift_label']
    info_table = table.Table(info_data, names = info_names)

    # Dummy Light Curve Table
    output_names = ['MJD'    , 'Mag'    , 'MagErr' , 'Telescope', 'Filter', 'Source', 'UL'   , 'Ignore']
    output_types = ['float64', 'float64', 'float64', 'S25'      , 'S25'   , 'S25'   , 'S25'  , 'S25'   ]
    output_data  = [50000    , 20.0     , 0.1      , 'FLWO'     , 'r'     , 'FLWO'  , 'False', 'False' ]
    output_table = table.Table(data = np.array(output_data).T, names = output_names, dtype = output_types)

    print('Plotting ...')
    if plot_lightcurve:
        make_plot(object_name + '_host', ra_deg, dec_deg, output_table, data_catalog, info_table, best_host, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, 50000, 50000, search_radius, star_cut, Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan)
    
    return info_table
