from FLEET.transient import get_transient_info, generate_lightcurve, ignore_data
from FLEET.catalog import get_catalog, catalog_operations, get_best_host, get_extinction
from sklearn.ensemble import RandomForestClassifier
from astropy.coordinates import Distance
from imblearn.over_sampling import SMOTE
from FLEET.lightcurve import fit_linex
from FLEET.plot import make_plot
from astropy import table
import pkg_resources
import numpy as np

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

def create_features(object_name, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, bright_mjd, first_mjd, green_brightest, red_brightest, host_radius, host_separation, host_Pcc, host_magnitude, hostless_cut = 0.1, redshift = np.nan):
    '''
    Create an astropy table with all the relevant features to feed into the classifier

    Parameters
    ---------------
    red_amplitude  , red_amplitude2  , red_offset  , red_magnitude   : Best fit parameters for r-band light curve
    green_amplitude, green_amplitude2, green_offset, green_magnitude : Best fit parameters for g-band light curve
    model_color     : g-r color from the model light curves at the time of brightest measured magnitude
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
        absmag, _ = redshift_magnitude(red_brightest, redshift)
        if np.isinf(absmag): absmag = 0
    else:
        absmag = np.nan

    features       = np.array(                     [object_name  , red_amplitude  , red_amplitude2  , red_offset  , red_magnitude  , green_amplitude  , green_amplitude2  , green_offset  , green_magnitude  , delta_time  , input_separation  , input_size  ,  normal_separation   , deltamag_red   , deltamag_green, model_color   , host_Pcc   , redshift   , absmag])
    features_table = table.Table(features, names = ['object_name', 'red_amplitude', 'red_amplitude2', 'red_offset', 'red_magnitude', 'green_amplitude', 'green_amplitude2', 'green_offset', 'green_magnitude', 'delta_time', 'input_separation', 'input_size',  'normal_separation', 'deltamag_red', 'deltamag_green',  'model_color', 'Pcc'    , 'redshift',  'absmag'],
                                           dtype = ['S25'        , 'float64'      ,        'float64',    'float64',       'float64',         'float64',          'float64',      'float64',         'float64',    'float64',          'float64',    'float64',            'float64',      'float64',        'float64',      'float64', 'float64',  'float64', 'float64'])

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
    Predicted Probability to be ['Nuclear','SLSN-I','SLSN-II','SNII','SNIIb','SNIIn','SNIa','SNIbc','Star']
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
    good = [i not in bad for i in training_table_in['mod_object_name']]
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
    if feature_set == 0 :  use_features = ['red_amplitude',                   'green_amplitude',                                      'normal_separation' , 'deltamag_red', 'deltamag_green'                            ]
    if feature_set == 1 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green'                            ]
    if feature_set == 2 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   ,                       'deltamag_red', 'deltamag_green'                            ]
    if feature_set == 3 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                               ]
    if feature_set == 4 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   ,                                                                                   ]
    if feature_set == 5 :  use_features = ['red_amplitude',                   'green_amplitude',                                                                                                                        ]
    if feature_set == 6 :  use_features = [                                                                                           'normal_separation' , 'deltamag_red', 'deltamag_green'                            ]
    if feature_set == 7 :  use_features = ['red_amplitude',                   'green_amplitude',                                      'normal_separation'                                                               ]
    if feature_set == 8 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'color'                  ]
    if feature_set == 9 :  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'color'                  ]
    if feature_set == 10:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'Pcc'     ]
    if feature_set == 11:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'redshift']
    if feature_set == 12:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'absmag'  ]
    if feature_set == 13:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'model_color'            ]
    if feature_set == 14:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ]
    if feature_set == 15:  use_features = ['red_amplitude',                   'green_amplitude',                                      'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ]
    if feature_set == 16:  use_features = ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'model_color', 'redshift']

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

    training_class_in[np.where(training_class_in == 'TDE'    )] = 'Nuclear'
    training_class_in[np.where(training_class_in == 'AGN'    )] = 'Nuclear'

    classes_names = {'Nuclear' : 0, 'SLSN-I' : 1, 'SLSN-II' : 2, 'SNII'    : 3, 'SNIIb'   : 4,
                     'SNIIn'   : 5, 'SNIa'   : 6, 'SNIbc'   : 7, 'Star' : 8}
    training_class = np.array([classes_names[i] for i in training_class_in]).astype(int)

    # SMOTE the data
    sampler = SMOTE(random_state=SMOTE_state)
    data_train_smote, class_train_smote = sampler.fit_resample(training_data, training_class)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=clf_state)
    clf.fit(data_train_smote, class_train_smote)

    # Predict Excluded Object
    predicted_probability = clf.predict_proba(testing_data)

    return predicted_probability

def predict_SLSN(object_name_in = '', ra_in = '', dec_in = '', redshift = np.nan, acceptance_radius = 3, import_ZTF = True, import_OSC = True, import_local = True, import_lightcurve = True, reimport_catalog = False, search_radius = 1.0, dust_map = 'SFD', Pcc_filter = 'i', Pcc_filter_alternative = 'r', star_separation = 1, star_cut = 0.1, date_range = np.inf, n_walkers = 50, n_steps = 500, n_cores = 1, model = 'single', training_days = 20, hostless_cut = 0.1, sorting_state = 42, clean = 0, SMOTE_state = 42, clf_state = 42, n_estimators = 100, max_depth = 7, feature_set = 13, neighbors = 20, classifier = '', plot_lightcurve = False):
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
    classifier              : Pick the classifier to use based on the available information
                              either 'quick', 'redshift', 'host', or 'late'. 
    plot_lightcurve         : Save an output plot with the light curve and PS1 image?

    Returns
    ---------------
    Predicted Probability to be ['Nuclear','SLSN-I','SLSN-II','SNII','SNIIb','SNIIn','SNIa','SNIbc','Star']

    '''
    print('\n################# FLEET #################')
    ##### Basic transient info #####
    ra_deg, dec_deg, transient_source, object_name, ztf_data, ztf_name, tns_name, osc_data = get_transient_info(object_name_in, ra_in, dec_in, acceptance_radius, import_ZTF, import_OSC, import_lightcurve)
    if ra_deg == '--': return
    print('%s %s %s'%(object_name, ra_deg, dec_deg))
    if dec_deg <= -32:
        print('dec = %s, too low for SDSS or 3PI'%dec_deg)
        return 

    ##### Lightcurve data #####
    output_table = generate_lightcurve(ztf_data, osc_data, object_name, ztf_name, tns_name, import_lightcurve, import_local)
    # Ignore selected patches
    output_table = ignore_data(object_name, output_table)

    if len(output_table) == 0:
        print('No data in lightcurve')
        return
    if np.sum((output_table['UL'] == 'False') & (output_table['Ignore'] == 'False')) == 0:
        print('No useable data in lightcurve')
        return

    # Extinction
    g_correct, r_correct = get_extinction(ra_deg, dec_deg, dust_map)

    ##### Fit Lightcurve #####
    red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, bright_mjd, first_mjd, green_brightest, red_brightest = fit_linex(output_table, date_range, n_walkers, n_steps, n_cores, model, g_correct, r_correct)
    if np.isnan(red_amplitude):
        return

    ##### Catalog data #####
    data_catalog_out = get_catalog(object_name, ra_deg, dec_deg, search_radius, dust_map, reimport_catalog)
    if len(data_catalog_out) == 0:
        print('No data found in SDSS or 3PI')
        return

    ##### Catalog Operations #####
    data_catalog = catalog_operations(data_catalog_out, ra_deg, dec_deg, Pcc_filter, Pcc_filter_alternative, neighbors)

    ##### Find the Best host #####
    host_radius, host_separation, host_Pcc, host_magnitude, host_nature, photoz, photoz_err, specz, specz_err = get_best_host(data_catalog, star_separation, star_cut)

    ##### Get Features #####
    features_table = create_features(object_name, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, bright_mjd, first_mjd, green_brightest, red_brightest, host_radius, host_separation, host_Pcc, host_magnitude, hostless_cut, redshift)

    ##### Run Classifier #####
    if classifier == '':
        predicted_probability = create_training_testing(object_name, features_table, training_days, model, clean, feature_set, sorting_state, SMOTE_state, clf_state, n_estimators, max_depth, hostless_cut)
    elif classifier == 'quick':
        predicted_probability = create_training_testing(object_name, features_table, training_days = 20, model = 'single', clean = 0, feature_set = 13, max_depth = 7)
    elif classifier == 'redshift':
        predicted_probability = create_training_testing(object_name, features_table, training_days = 20, model = 'single', clean = 0, feature_set = 16, max_depth = 7)
    elif classifier == 'late':
        predicted_probability = create_training_testing(object_name, features_table, training_days = 70, model = 'double', clean = 0, feature_set = 7 , max_depth = 9)
    elif classifier == 'host':
        predicted_probability = create_training_testing(object_name, features_table, training_days = 70, model = 'double', clean = 1, feature_set = 7 , max_depth = 9)

    # Predicted Probability to be ['Nuclear','SLSN-I','SLSN-II','SNII','SNIIb','SNIIn','SNIa','SNIbc','Star']
    P_SLSNI = predicted_probability[0][1]
    # Return Probability that the object is a SLSN-I

    if plot_lightcurve:
        make_plot(object_name, ra_deg, dec_deg, output_table, first_mjd, bright_mjd, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, g_correct, r_correct)

    return P_SLSNI