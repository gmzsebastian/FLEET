from FLEET.plot import make_plot, calculate_observability, redshift_magnitude, quick_plot
from FLEET.catalog import get_catalog, catalog_operations, get_best_host, get_extinction, overwrite_glade
from FLEET.transient import get_transient_info, generate_lightcurve, ignore_data
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from FLEET.lightcurve import fit_linex
from astropy import table
import pkg_resources
import numpy as np
import pickle
import glob
import os

def create_features(acceptance_radius,import_ZTF,import_OSC,import_local,import_lightcurve,reimport_catalog,search_radius,Pcc_filter,Pcc_filter_alternative,star_separation,star_cut,
                    date_range,late_phase,n_walkers,n_steps,n_cores,model,training_days,sorting_state,clean,SMOTE_state,clf_state,n_estimators,max_depth,feature_set,neighbors,
                    recalculate_nature,classifier,n_samples,plot_lightcurve,do_observability,save_features,ra_deg, dec_deg, transient_source, object_name, object_name_in,
                    ztf_name, tns_name, object_class, g_correct, r_correct, dust_map, red_amplitude, red_amplitude2,
                    red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, late_color, late_color10, late_color20, late_color40,
                    late_color60, bright_mjd, first_mjd, green_brightest, red_brightest, chi2, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_magnitude_g,
                    host_magnitude_r, host_nature, photoz, photoz_err, specz, specz_err, best_host, force_detection, use_redshift, redshift_label, hostless_cut = 0.1):
    '''
    Create an astropy table with all the relevant features to feed into the classifier

    Parameters
    ---------------
    acceptance_radius      : Threshold in arcseconds for catalog cross-matching
    import_ZTF             : Import ZTF data?
    import_OSC             : Import OSC data?
    import_local           : Import local photometry data?
    import_lightcurve      : Regenerate existing lightcurve file?
    reimport_catalog       : Overwrite the existing 3PI/SDSS catalog?
    search_radius          : Search radius in arcminutes for the 3PI/SDSS catalog
    Pcc_filter             : Filter for the host magnitude, radius, and Pcc
    Pcc_filter_alternative : Alternative filter for the host magnitude, radius, and Pcc
    star_separation        : Maximum star-transient separation to determine association
    star_cut               : Maximum allowed probability of an object to be a star
    date_range             : Maximum number of light curve days from the first detection to fit
    late_phase             : Phase at which to calculate the late color
    n_walkers              : Number of walkers for MCMC
    n_steps                : Number of steps for MCMC
    n_cores                : Number of cores for MCMC
    model                  : 'single' or 'double', the latter fits late-time data
    training_days          : Which training set to use for classification, days
    sorting_state          : Seed number for list sorter
    clean                  : 0 keeps all objects, and 1 removes orphan transients
    SMOTE_state            : Seed number for SMOTE
    clf_state              : Seed number for classifier
    n_estimators           : Number of trees for random forest
    max_depth              : Depth of trees for random forest
    feature_set            : Set of features to use
    neighbors              : Neighbors to use for star/galaxy separator
    recalculate_nature     : Overwrite existing Nature column?
    classifier             : Classifier name used for classification
    n_samples              : Number of random seeds to use
    plot_lightcurve        : Save an output plot with the light curve and PS1 image?
    do_observability       : Calculate Observavility from Magellan and MMT?
    save_features          : Save the features table to a file?
    ra_deg                 : Transient RA
    dec_deg                : Transient DEC
    g_correct              : Extinction in g-band in SFD map
    r_correct              : Extinction in r-band in SFD map
    red_amplitude          : Light curve width in r-band, extinction corrected
    red_amplitude2         : Light curve Amplitude (W2 / A) in r-band, extinction corrected, default 0.37
    red_offset             : Time offset in r-band
    red_magnitude          : Peak r-band model magnitude, extinction corrected
    green_amplitude        : Light curve width in g-band, extinction corrected
    green_amplitude2       : Light curve Amplitude (W2 / A) in g-band, extinction corrected, default 0.55
    green_offset           : Time offset in g-band
    green_magnitude        : Peak g-band model magnitude, extinction corrected
    model_color            : Model g-r measured at bright_mjd
    late_color             : Model g-r measured at specified phase
    late_color10           : Model g-r measured at bright_mjd + 10
    late_color20           : Model g-r measured at bright_mjd + 20
    late_color40           : Model g-r measured at bright_mjd + 40
    late_color60           : Model g-r measured at bright_mjd + 60
    bright_mjd             : MJD of brightest magnitude, except UL and Ignore, any band
    first_mjd              : MJD of first detection, except UL and Ignore, any band
    green_brightest        : Peak measured g-band magnitude, extinction corrected
    red_brightest          : Peak measured r-band magnitude, extinction corrected
    chi2                   : Combined reduced chi^2 of r and g band light curve models
    host_radius            : Half-light radius in i-band, or r-band if i is not detected
    host_separation        : Host-transient separation, not band-dependent
    host_ra                : Host RA
    host_dec               : Host DEC
    host_Pcc               : Probability of chance coincidence in i-band, or r-band if i is not detected, not extinction corrected
    host_magnitude         : Host magnitude in i-band, or r-band if i is not detected, not extinction corrected
    host_magnitude_g       : Host magnitude in g-band, not extinction corrected
    host_magnitude_r       : Host magnitude in r-band, not extinction corrected
    host_nature            : 0 means star, 1 means galaxy, averaged among all available bands
    photoz                 : SDSS photometric redshift
    photoz_err             : SDSS photometric redshift error
    specz                  : SDSS spectroscopic redshift
    specz_err              : SDSS spectroscopic redshift error
    best_host              : Index of best host from catalog
    redshift               : Redshift used or specified by used
    delta_time             : bright_mjd - first_mjd
    deltamag_red           : Host - Transient mag, using default "host_magnitude", extinction ignored. Or 3PI 5-sigma limits if undeteced.
    deltamag_green         : Host - Transient mag, using default "host_magnitude", extinction ignored. Or 3PI 5-sigma limits if undeteced.
    absmag                 : Absolute magnitude using "red_brightest" and "redshift"
    input_separation       : Same as "host_separation", but 0 if "host_Pcc" > "hostless_cut"
    input_size             : Same as "host_radius", but 0 if "host_Pcc" > "hostless_cut"
    input_magnitude        : Same as "host_magnitude", but 0 if "host_Pcc" > "hostless_cut"
    normal_separation      : "input_separation" / "input_size"
    true_deltamag_red      : Host - Transient mag, using default "host_magnitude_r", extinction ignored. Or 3PI 5-sigma limits if undeteced.
    true_deltamag_green    : Host - Transient mag, using default "host_magnitude_g", extinction ignored. Or 3PI 5-sigma limits if undeteced.
    hostless_cut           : Only consider host galaxies with a with a probability of chance coincidence lower than this

    Return
    ---------------
    An astropy table with all the relevant features

    '''

    # Calculate time from discovery to peak
    delta_time = bright_mjd - first_mjd

    # Obtain Features
    if (host_Pcc <= 0.02) | ((host_Pcc <= 0.07) & (host_separation <= 8)) | force_detection:
        input_separation  = host_separation
        input_size        = host_radius
        input_magnitude   = host_magnitude
        normal_separation = input_separation / input_size

        # Using the right host magnitude
        true_deltamag_red   = host_magnitude_r - red_brightest
        true_deltamag_green = host_magnitude_g - green_brightest
        input_host_g_r      = host_magnitude_g - host_magnitude_r

    else:
        input_separation  = 0
        input_size        = 0
        normal_separation = 0
        input_magnitude   = 23.1 # i-band limit from 3PI

        # Using the right host magnitude
        true_deltamag_red   = 23.2 - red_brightest    # r-band limit from 3PI
        true_deltamag_green = 23.3 - green_brightest  # g-band limit from 3PI
        input_host_g_r      = 0

    # Calculate deltamag (Host - Transient)
    deltamag_red   = input_magnitude - red_brightest
    deltamag_green = input_magnitude - green_brightest

    # Redshift and Absolute Magnitude
    if np.isfinite(red_brightest):
        absmag, _ = redshift_magnitude(red_brightest, np.max([use_redshift, 0]))
        if np.isinf(absmag): absmag = 0
    else:
        absmag = np.nan

    data = [acceptance_radius,import_ZTF,import_OSC,import_local,import_lightcurve,reimport_catalog,search_radius,Pcc_filter,Pcc_filter_alternative,star_separation,star_cut,
            date_range,late_phase,n_walkers,n_steps,n_cores,model,training_days,sorting_state,clean,SMOTE_state,clf_state,n_estimators,max_depth,feature_set,neighbors,
            recalculate_nature,classifier,n_samples,plot_lightcurve,do_observability,save_features,ra_deg, dec_deg, transient_source, object_name,
            object_name_in, ztf_name, tns_name, object_class, g_correct, r_correct, dust_map, red_amplitude, red_amplitude2,
            red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, late_color, late_color10, late_color20, late_color40,
            late_color60, bright_mjd, first_mjd, green_brightest, red_brightest, chi2, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_magnitude_g,
            host_magnitude_r, input_host_g_r, host_nature, photoz, photoz_err, specz, specz_err, best_host, use_redshift, redshift_label, delta_time, deltamag_red, deltamag_green, absmag,
            input_separation, input_size, input_magnitude, normal_separation, true_deltamag_red, true_deltamag_green, hostless_cut]
    names = ['acceptance_radius','import_ZTF','import_OSC','import_local','import_lightcurve','reimport_catalog','search_radius','Pcc_filter','Pcc_filter_alternative','star_separation','star_cut',
             'date_range','late_phase','n_walkers','n_steps','n_cores','model','training_days','sorting_state','clean','SMOTE_state','clf_state','n_estimators','max_depth','feature_set','neighbors',
             'recalculate_nature','classifier','n_samples','plot_lightcurve','do_observability','save_features','ra_deg', 'dec_deg', 'transient_source', 'object_name',
             'object_name_in', 'ztf_name', 'tns_name', 'object_class', 'g_correct', 'r_correct', 'dust_map', 'red_amplitude', 'red_amplitude2',
            'red_offset', 'red_magnitude', 'green_amplitude', 'green_amplitude2', 'green_offset', 'green_magnitude', 'model_color', 'late_color', 'late_color10', 'late_color20', 'late_color40',
            'late_color60', 'bright_mjd', 'first_mjd', 'green_brightest', 'red_brightest', 'chi2', 'host_radius', 'host_separation', 'host_ra', 'host_dec', 'host_Pcc', 'host_magnitude', 'host_magnitude_g',
            'host_magnitude_r', 'input_host_g_r', 'host_nature', 'photoz', 'photoz_err', 'specz', 'specz_err', 'best_host', 'redshift', 'redshift_label', 'delta_time', 'deltamag_red', 'deltamag_green', 'absmag',
            'input_separation', 'input_size', 'input_magnitude', 'normal_separation', 'true_deltamag_red', 'true_deltamag_green', 'hostless_cut']
    types = ['float64','S25','S25','S25','S25','S25','float64','S25','S25','float64','float64','float64','float64','float64','float64','float64','S25','float64','float64',
            'float64','float64','float64','float64','float64','float64','float64','S25','S25','float64','S25','S25','S25','float64', 'float64', 'S25', 'S25', 'S25', 'S25', 'S25', 'S25', 'float64', 'float64', 'S25', 'float64', 'float64',
            'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
            'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
            'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'S25', 'float64', 'float64', 'float64', 'float64',
            'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64']

    features_table = table.Table(data = np.array(data), names = names, dtype = types)

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
    bad  = ['2019vzf','2019wbg','2020acwp','2020eqv','2019fer','2020aafm','2020krl','2016hvm']
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
    if feature_set == 1 : use_features = ['red_amplitude','green_amplitude','host_Pcc','host_nature','input_host_g_r','normal_separation','model_color','late_color20','chi2','delta_time','deltamag_red','deltamag_green'] # Every Feature

    # For SLSNe - Late time
    if feature_set == 2 : use_features = ['red_amplitude','green_amplitude','host_Pcc'              ,'input_host_g_r','normal_separation','model_color','late_color20'                    ,'deltamag_red','deltamag_green'] # Optimal Features for SLSNe
    if feature_set == 3 : use_features = [                                  'host_Pcc'              ,'input_host_g_r','normal_separation'                                                 ,'deltamag_red','deltamag_green'] # Optimal Host-only Features for SLSNe
    if feature_set == 4 : use_features = ['red_amplitude','green_amplitude'                                                              ,'model_color','late_color20','chi2'                                             ] # Optimal Lightcurve-only Features for SLSNe

    # For SLSNe - Rapid
    if feature_set == 5 : use_features = ['red_amplitude','green_amplitude','host_Pcc'              ,'input_host_g_r','normal_separation','model_color'                      ,'delta_time','deltamag_red','deltamag_green'] # Optimal Features for SLSNe - rapid
    if feature_set == 6 : use_features = [                                  'host_Pcc'              ,'input_host_g_r','normal_separation'                                    ,'delta_time','deltamag_red','deltamag_green'] # Optimal Host-only Features for SLSNe - rapid
    if feature_set == 7 : use_features = ['red_amplitude','green_amplitude'                                                              ,'model_color'               ,'chi2','delta_time'                                ] # Optimal Lightcurve-only Features for SLSNe - rapid

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

def pickled_training_set(filename, features_table, feature_set, model):
    # Select Features
    if feature_set == 1 : use_features = ['red_amplitude','green_amplitude','host_Pcc','host_nature','input_host_g_r','normal_separation','model_color','late_color20','chi2','delta_time','deltamag_red','deltamag_green'] # Every Feature

    # For SLSNe - Late time
    if feature_set == 2 : use_features = ['red_amplitude','green_amplitude','host_Pcc'              ,'input_host_g_r','normal_separation','model_color','late_color20'                    ,'deltamag_red','deltamag_green'] # Optimal Features for SLSNe
    if feature_set == 3 : use_features = [                                  'host_Pcc'              ,'input_host_g_r','normal_separation'                                                 ,'deltamag_red','deltamag_green'] # Optimal Host-only Features for SLSNe
    if feature_set == 4 : use_features = ['red_amplitude','green_amplitude'                                                              ,'model_color','late_color20','chi2'                                             ] # Optimal Lightcurve-only Features for SLSNe

    # For SLSNe - Rapid
    if feature_set == 5 : use_features = ['red_amplitude','green_amplitude','host_Pcc'              ,'input_host_g_r','normal_separation','model_color'                      ,'delta_time','deltamag_red','deltamag_green'] # Optimal Features for SLSNe - rapid
    if feature_set == 6 : use_features = [                                  'host_Pcc'              ,'input_host_g_r','normal_separation'                                    ,'delta_time','deltamag_red','deltamag_green'] # Optimal Host-only Features for SLSNe - rapid
    if feature_set == 7 : use_features = ['red_amplitude','green_amplitude'                                                              ,'model_color'               ,'chi2','delta_time'                                ] # Optimal Lightcurve-only Features for SLSNe - rapid

    # If using the 'double' model, add the W2 parameter
    if model == 'double':
        use_features += ['red_amplitude2','green_amplitude2']

    # Create array with Training and Testing data
    testing_data  = np.array(features_table[use_features].to_pandas())

    with open(filename, 'rb') as f:
        clf = pickle.load(f)

    # Predict Excluded Object
    predicted_probability = 100 * clf.predict_proba(testing_data)

    return predicted_probability

def predict_SLSN(object_name_in = '', ra_in = '', dec_in = '', redshift = np.nan, acceptance_radius = 3, import_ZTF = True, import_OSC = True, import_local = True, import_lightcurve = True, reimport_catalog = False,
                 search_radius = 1.0, dust_map = 'SFD', Pcc_filter = 'i', Pcc_filter_alternative = 'r', star_separation = 1, star_cut = 0.1, date_range = 75, late_phase = 40, n_walkers = 50, n_steps = 1000,
                 n_cores = 1, model = 'single', training_days = 20, hostless_cut = 0.1, sorting_state = 42, clean = 0, SMOTE_state = 42, clf_state = 42, n_estimators = 100, max_depth = 7, feature_set = 13,
                 neighbors = 20, recalculate_nature = False, classifier = 'all', n_samples = 3, object_class = '', plot_lightcurve = False, do_observability = False, save_features = False, use_glade = False):
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
    use_glade               : Use the GLADE catalog to overwrite the best host

    Returns
    ---------------
    Predicted Probability to be ['AGN','SLSN-I','SLSN-II','SNII','SNIIb','SNIIn','SNIa','SNIbc','Star','TDE']

    '''
    print('\n################# FLEET #################')



    ########## Basic transient info ##########
    ra_deg, dec_deg, transient_source, object_name, ztf_data, ztf_name, tns_name, object_class, osc_data = get_transient_info(object_name_in, ra_in, dec_in, object_class, acceptance_radius, import_ZTF, import_OSC, import_lightcurve)

    # If transient info failed, return empty
    if ra_deg == '--':
        print('Coordinates cannot be empty')
        return table.Table()
    # If transient is too south, return empty
    if dec_deg <= -32:
        print(f'dec = {dec_deg}, too low for SDSS or 3PI')
        return table.Table()
    # Else, print correctly
    print('%s %s %s'%(object_name, ra_deg, dec_deg))



    ########## Lightcurve data ##########
    output_table = generate_lightcurve(ztf_data, osc_data, object_name, ztf_name, tns_name, import_lightcurve, import_local)
    # Ignore selected patches
    output_table = ignore_data(object_name, output_table)

    # If there's no data
    if len(output_table) == 0:
        print('No data in lightcurve')
        return table.Table()
    # If all data is ignored or non-detections
    if np.sum((output_table['UL'] == 'False') & (output_table['Ignore'] == 'False')) == 0:
        print('No useable data in lightcurve')
        return table.Table()

    # Get extinction values
    g_correct, r_correct = get_extinction(ra_deg, dec_deg, dust_map)



    ########## Fit Lightcurve ##########
    red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, late_color, late_color10, late_color20, late_color40, late_color60, bright_mjd, first_mjd, green_brightest, red_brightest, chi2 = fit_linex(output_table, date_range, n_walkers, n_steps, n_cores, model, g_correct, r_correct, late_phase)

    # If fit failed there's probably not enough data
    if np.isnan(red_amplitude):
        if plot_lightcurve:
            # MJD including UL and Ignores
            #absolute_first_mjd = np.nanmin(np.array(output_table['MJD']).astype(float))
            # = output_table['MJD'][np.nanargmin(output_table['Mag'])]
            quick_plot(object_name, object_class, ra_deg, dec_deg, output_table, first_mjd, bright_mjd, full_range = True)
        return table.Table()



    ########## Catalog Operations ##########
    data_catalog_out, catalog_exists = get_catalog(object_name, ra_deg, dec_deg, search_radius, dust_map, reimport_catalog)
    # If there's no catalog, return empty
    if len(data_catalog_out) == 0:
        print('No data found in SDSS or 3PI')
        return table.Table()

    # Processing catalog data
    print('Creating Catalog ...')
    data_catalog = catalog_operations(object_name, data_catalog_out, ra_deg, dec_deg, Pcc_filter, Pcc_filter_alternative, neighbors, recalculate_nature, catalog_exists)

    # Find the best host
    if use_glade:
        best_host_glade = overwrite_glade(ra_deg, dec_deg, object_name, data_catalog)
    else:
        best_host_glade = np.nan

    host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_magnitude_g, host_magnitude_r, host_nature, photoz, photoz_err, specz, specz_err, best_host, force_detection = get_best_host(data_catalog, star_separation, star_cut, best_host_glade)

    # Define Appropriate Redshift
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

    ########## Generate Features ##########
    features_table = create_features(acceptance_radius,import_ZTF,import_OSC,import_local,import_lightcurve,reimport_catalog,search_radius,Pcc_filter,Pcc_filter_alternative,star_separation,star_cut,
                                     date_range,late_phase,n_walkers,n_steps,n_cores,model,training_days,sorting_state,clean,SMOTE_state,clf_state,n_estimators,max_depth,feature_set,neighbors,
                                     recalculate_nature,classifier,n_samples,plot_lightcurve,do_observability,save_features,ra_deg, dec_deg, transient_source, object_name,
                                     object_name_in, ztf_name, tns_name, object_class, g_correct, r_correct, dust_map, red_amplitude, red_amplitude2,
                                     red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, late_color, late_color10, late_color20, late_color40,
                                     late_color60, bright_mjd, first_mjd, green_brightest, red_brightest, chi2, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_magnitude_g,
                                     host_magnitude_r, host_nature, photoz, photoz_err, specz, specz_err, best_host, force_detection, use_redshift, redshift_label, hostless_cut)
    # Return table here if this is the end
    if save_features:
        if plot_lightcurve:
            # MJD including UL and Ignores
            #absolute_first_mjd = np.nanmin(np.array(output_table['MJD']).astype(float))
            #bright_mjd = output_table['MJD'][np.nanargmin(output_table['Mag'])]
            quick_plot(object_name, object_class, ra_deg, dec_deg, output_table, first_mjd, bright_mjd, g_correct, r_correct, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, full_range = False)
        return features_table

    print('Classifying ...')
    if classifier == 'all':
        # Late-time classifier
        filenames_main = f'pickles/main_late_*.pkl'
        full_filenames_main = glob.glob(pkg_resources.resource_filename(__name__, filenames_main))

        for i in range(len(full_filenames_main)):
            filename = full_filenames_main[i]
            late_probability_n = pickled_training_set(filename, features_table, feature_set = 2, model = 'double')
            if i == 0:
                late_probability = late_probability_n
            else:
                late_probability = np.vstack([late_probability, late_probability_n])
        late_probability_average = np.average(late_probability, axis = 0)
        late_probability_std     = np.std    (late_probability, axis = 0)

        # Rapid SLSN classifier
        filenames_slsn = f'pickles/slsn_rapid_*.pkl'
        full_filenames_slsn = glob.glob(pkg_resources.resource_filename(__name__, filenames_slsn))

        for i in range(len(full_filenames_slsn)):
            filename = full_filenames_slsn[i]
            quick_probability_n = pickled_training_set(filename, features_table, feature_set = 5, model = 'single')
            if i == 0:
                quick_probability = quick_probability_n
            else:
                quick_probability = np.vstack([quick_probability, quick_probability_n])
        quick_probability_average = np.average(quick_probability, axis = 0)
        quick_probability_std     = np.std    (quick_probability, axis = 0)

        # Rapid TDE classifier
        filenames_tde = f'pickles/tde_rapid_*.pkl'
        full_filenames_tde = glob.glob(pkg_resources.resource_filename(__name__, filenames_tde))

        for i in range(len(full_filenames_tde)):
            filename = full_filenames_tde[i]
            host_probability_n = pickled_training_set(filename, features_table, feature_set = 5, model = 'single')
            if i == 0:
                host_probability = host_probability_n
            else:
                host_probability = np.vstack([host_probability, host_probability_n])
        host_probability_average = np.average(host_probability, axis = 0)
        host_probability_std     = np.std    (host_probability, axis = 0)

        # Empty right now because I never use it
        redshift_probability_average = np.zeros_like(quick_probability_average)
        redshift_probability_std     = np.zeros_like(quick_probability_average)
    else:
        print("Other classifiers not implemented yet")
        # quick_probability_n = create_training_testing(object_name, features_table, training_days = training_days_quick, model = model_quick, clean = clean_quick, feature_set = feature_set_quick, max_depth = max_depth_quick, clf_state = int(39 + i))

    # Calculate Time Span
    detections = (np.array(output_table['UL']) == b'False') & (np.array(output_table['Ignore']) == b'False') & ((output_table['Filter'] == 'r') | (output_table['Filter'] == 'g'))
    time_span  = np.nanmax(output_table['MJD'][detections]) - np.nanmin(output_table['MJD'][detections])
    # Calculate Observability
    Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, MMT_observable, Magellan_observable = calculate_observability(ra_deg, dec_deg, do_observability)

    # Output Array
    info_data  = np.array([object_name, ztf_name, tns_name, object_class, ra_deg, dec_deg, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, photoz, photoz_err, specz, specz_err, use_redshift, redshift_label, classifier, time_span, MMT_observable, Magellan_observable, *quick_probability_average, *quick_probability_std, *late_probability_average, *late_probability_std, *redshift_probability_average, *redshift_probability_std, *host_probability_average, *host_probability_std])
    info_names = ['object_name'       ,'ztf_name'             ,'tns_name'             ,'object_class'         ,'ra_deg'               ,'dec_deg'              ,'host_radius'          ,'host_separation'      ,'host_ra'              ,'host_dec'          ,'host_Pcc'           ,'host_magnitude',
                  'host_nature'       ,'photoz'               ,'photoz_err'           ,'specz'                ,'specz_err'            ,'use_redshift'         ,'redshift_label'       ,'classifier'           ,'time_span'            ,'MMT_observable'    ,'Magellan_observable',
                  'P_quick_AGN'       ,'P_quick_SLSNI'        ,'P_quick_SLSNII'       ,'P_quick_SNII'         ,'P_quick_SNIIb'        ,'P_quick_SNIIn'        ,'P_quick_SNIa'         ,'P_quick_SNIbc'        ,'P_quick_Star'         ,'P_quick_TDE'       ,
                  'P_quick_AGN_std'   ,'P_quick_SLSNI_std'    ,'P_quick_SLSNII_std'   ,'P_quick_SNII_std'     ,'P_quick_SNIIb_std'    ,'P_quick_SNIIn_std'    ,'P_quick_SNIa_std'     ,'P_quick_SNIbc_std'    ,'P_quick_Star_std'     ,'P_quick_TDE_std'   ,
                  'P_late_AGN'        ,'P_late_SLSNI'         ,'P_late_SLSNII'        ,'P_late_SNII'          ,'P_late_SNIIb'         ,'P_late_SNIIn'         ,'P_late_SNIa'          ,'P_late_SNIbc'         ,'P_late_Star'          ,'P_late_TDE'        ,
                  'P_late_AGN_std'    ,'P_late_SLSNI_std'     ,'P_late_SLSNII_std'    ,'P_late_SNII_std'      ,'P_late_SNIIb_std'     ,'P_late_SNIIn_std'     ,'P_late_SNIa_std'      ,'P_late_SNIbc_std'     ,'P_late_Star_std'      ,'P_late_TDE_std'    ,
                  'P_redshift_AGN'    ,'P_redshift_SLSNI'     ,'P_redshift_SLSNII'    ,'P_redshift_SNII'      ,'P_redshift_SNIIb'     ,'P_redshift_SNIIn'     ,'P_redshift_SNIa'      ,'P_redshift_SNIbc'     ,'P_redshift_Star'      ,'P_redshift_TDE'    ,
                  'P_redshift_AGN_std','P_redshift_SLSNI_std' ,'P_redshift_SLSNII_std','P_redshift_SNII_std'  ,'P_redshift_SNIIb_std' ,'P_redshift_SNIIn_std' ,'P_redshift_SNIa_std'  ,'P_redshift_SNIbc_std' ,'P_redshift_Star_std'  ,'P_redshift_TDE_std',
                  'P_host_AGN'        ,'P_host_SLSNI'         ,'P_host_SLSNII'        ,'P_host_SNII'          ,'P_host_SNIIb'         ,'P_host_SNIIn'         ,'P_host_SNIa'          ,'P_host_SNIbc'         ,'P_host_Star'          ,'P_host_TDE'        ,
                  'P_host_AGN_std'    ,'P_host_SLSNI_std'     ,'P_host_SLSNII_std'    ,'P_host_SNII_std'      ,'P_host_SNIIb_std'     ,'P_host_SNIIn_std'     ,'P_host_SNIa_std'      ,'P_host_SNIbc_std'     ,'P_host_Star_std'      ,'P_host_TDE_std'    ]
    data_table = table.Table(info_data, names = info_names)

    # Join Features and Information
    use_names  = list(set(data_table.colnames) - set(features_table.colnames))
    info_table = table.hstack([features_table, data_table[use_names]])

    print('Plotting ...')
    if plot_lightcurve:
        make_plot(object_name, ra_deg, dec_deg, output_table, data_catalog, info_table, best_host, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, first_mjd, bright_mjd, search_radius, star_cut, Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, do_observability, red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, chi2, g_correct, r_correct)

    return info_table

def predict_Host(object_name_in = '', ra_in = '', dec_in = '', redshift = np.nan, acceptance_radius = 3, import_ZTF = False, import_OSC = False, import_local = False, import_lightcurve = False, reimport_catalog = False,
                 search_radius = 1.0, dust_map = 'SFD', Pcc_filter = 'i', Pcc_filter_alternative = 'r', star_separation = 1, star_cut = 0.1, date_range = np.inf, late_phase = 40, n_walkers = 50, n_steps = 1000,
                 n_cores = 1, model = 'single', training_days = 20, hostless_cut = 0.1, sorting_state = 42, clean = 0, SMOTE_state = 42, clf_state = 42, n_estimators = 100, max_depth = 7, feature_set = 13,
                 neighbors = 20, recalculate_nature = False, classifier = '', n_samples = 3, object_class = '', plot_lightcurve = True, do_observability = False, save_features = False, use_glade = False):
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
    use_glade               : Use the GLADE catalog to overwrite the best host

    Returns
    ---------------
    Predicted Probability to be ['AGN','SLSN-I','SLSN-II','SNII','SNIIb','SNIIn','SNIa','SNIbc','Star','TDE']

    '''
    print('\n################# FLEET #################')

    ########## Basic transient info ##########
    ra_deg, dec_deg, transient_source, object_name, ztf_data, ztf_name, tns_name, object_class, osc_data = get_transient_info(object_name_in, ra_in, dec_in, object_class, acceptance_radius, import_ZTF, import_OSC, import_lightcurve)

    # If transient info failed, return empty
    if ra_deg == '--':
        print('Coordinates cannot be empty')
        return table.Table()
    # If transient is too south, return empty
    if dec_deg <= -32:
        print(f'dec = {dec_deg}, too low for SDSS or 3PI')
        return table.Table()
    # Else, print correctly
    print('%s %s %s'%(object_name, ra_deg, dec_deg))

    ########## Catalog Operations ##########
    data_catalog_out, catalog_exists = get_catalog(object_name, ra_deg, dec_deg, search_radius, dust_map, reimport_catalog)
    # If there's no catalog, return empty
    if len(data_catalog_out) == 0:
        print('No data found in SDSS or 3PI')
        return table.Table()

    # Processing catalog data
    print('Creating Catalog ...')
    data_catalog = catalog_operations(object_name, data_catalog_out, ra_deg, dec_deg, Pcc_filter, Pcc_filter_alternative, neighbors, recalculate_nature, catalog_exists)

    if use_glade:
        best_host_glade = overwrite_glade(ra_deg, dec_deg, object_name, data_catalog)
    else:
        best_host_glade = np.nan

    ##### Find the Best host #####
    host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_magnitude_g, host_magnitude_r, host_nature, photoz, photoz_err, specz, specz_err, best_host, force_detection = get_best_host(data_catalog, star_separation, star_cut, best_host_glade)

    # Define Appropriate Redshift
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

    ########## Generate Features ##########
    features_table = create_features(acceptance_radius,import_ZTF,import_OSC,import_local,import_lightcurve,reimport_catalog,search_radius,Pcc_filter,Pcc_filter_alternative,star_separation,star_cut,
                                     date_range,late_phase,n_walkers,n_steps,n_cores,model,training_days,sorting_state,clean,SMOTE_state,clf_state,n_estimators,max_depth,feature_set,neighbors,
                                     recalculate_nature,classifier,n_samples,plot_lightcurve,do_observability,save_features,ra_deg, dec_deg, transient_source, object_name,
                                     object_name_in, ztf_name, tns_name, object_class, np.nan, np.nan, dust_map, np.nan, np.nan,
                                     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_magnitude_g,
                                     host_magnitude_r, host_nature, photoz, photoz_err, specz, specz_err, best_host, force_detection, use_redshift, redshift_label, hostless_cut)

    # Dummy Light Curve Table
    output_names = ['MJD'    , 'Mag'    , 'MagErr' , 'Telescope', 'Filter', 'Source', 'UL'   , 'Ignore']
    output_types = ['float64', 'float64', 'float64', 'S25'      , 'S25'   , 'S25'   , 'S25'  , 'S25'   ]
    output_data  = [50000    , 20.0     , 0.1      , 'FLWO'     , 'r'     , 'FLWO'  , 'False', 'False' ]
    output_table = table.Table(data = np.array(output_data).T, names = output_names, dtype = output_types)

    # Return table here if this is the end
    if save_features:
        return features_table

    # Calculate Observability
    if do_observability:
        Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, MMT_observable, Magellan_observable = calculate_observability(ra_deg, dec_deg, do_observability)
    else:
        Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    print('Plotting ...')
    if plot_lightcurve:
        first_mjd  = 50000
        bright_mjd = 50000
        make_plot(object_name + '_host', ra_deg, dec_deg, output_table, data_catalog, features_table, best_host, host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude, host_nature, first_mjd, bright_mjd, search_radius, star_cut, Dates_MMT, Airmass_MMT, SunElevation_MMT, Dates_Magellan, Airmass_Magellan, SunElevation_Magellan, do_observability)

    return features_table
