from .transient import get_transient_info, process_lightcurve
from .model import fit_data
from .catalog import get_catalog, catalog_operations, overwrite_with_glade, get_best_host, host_limit
from .plot import make_plot, calculate_observability, calc_absmag, quick_plot
import pkg_resources
import multiprocessing
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from astropy import table
import numpy as np
import os
import glob
import pickle

try:
    fleet_data = os.environ['fleet_data']
except KeyError:
    fleet_data = os.path.join(os.path.dirname(__file__), 'data')

feature_set = {2: ['lc_width_r', 'lc_width_g', 'host_Pcc', 'input_host_g_r',
                   'normal_separation', 'color_peak', 'late_color20',
                   'deltamag_red', 'deltamag_green'],
               5: ['lc_width_r', 'lc_width_g', 'host_Pcc', 'input_host_g_r',
                   'normal_separation', 'color_peak', 'late_color20',
                   'delta_time', 'deltamag_red', 'deltamag_green'],
               17: ['lc_width', 'lc_decline', 'initial_temp', 'cooling_rate',
                    'host_Pcc', 'input_host_g_r', 'normal_separation', 'color_peak',
                    'late_color20', 'delta_time', 'deltamag_red', 'deltamag_green']
               }


def format_training(training_days, grouping, sorting_state, features, model, clean, remove_bad=True):
    """
    Function that will format the training set for needed for the classifier.
    Can be used to either create a new pickle file, or to validate the classifier.

    Parameters
    ----------
    training_days : float
        The number of days of photometry to use for training.
    grouping : str
        The grouping of transient types to use for training.
    sorting_state : bool
        The random seed for the sorting algorithm.
    features : int
        The index of the feature list to use for training.
    model : str, either 'single' or 'double'
        The model to use for training.
    clean : bool
        Whether to require objects to have a host galaxy.
    remove_bad : bool
        Whether to remove bad objects from the training set. Default is True.

    Returns
    -------
    training_data : np.ndarray
        The training data for the classifier.
    training_class : np.ndarray
        The training classes for the classifier.
    training_names : np.ndarray
        The AT names of each object in the table
    classes_names : dict
        The mapping of class names to indices.
    """

    # Read in table with input parameters
    table_name = pkg_resources.resource_filename(__name__, f'training_set/table_{model}_{training_days}.txt')
    training_table_in = table.Table.read(table_name, format='ascii')

    # Remove objects deemed to be bad
    # Some either have the wrong host, really poor light curves, should be hostless when they are not, etc.
    bad = ['2016hvm',  '2017fro',  '2018ebt',  '2018eub',  '2018ffj',  '2018hhr',
           '2018hna',  '2018jsc',  '2019afa',  '2019baj',  '2019bjp',  '2019dnz',
           '2019elm',  '2019gaf',  '2019gwl',  '2019gzd',  '2019hib',  '2019ief',
           '2019lnz',  '2019pdx',  '2019pjs',  '2019qo',   '2019rom',  '2019spk',
           '2019upq',  '2019wup',  '2019xdx',  '2020abyx', '2020acat', '2020aceu',
           '2020ackf', '2020acwp', '2020adgg', '2020axk',  '2020bpi',  '2020cxe',
           '2020dic',  '2020gar',  '2020gc',   '2020iji',  '2020kq',   '2020lmd',
           '2020mjm',  '2020mos',  '2020mrf',  '2020nps',  '2020nze',  '2020rue',
           '2020rxv',  '2020vba',  '2020xpy',  '2020ykb',  '2020ykr',  '2021cjy',
           '2021csf',  '2021dpw]', '2021hbl',  '2021hiw',  '2021hmb',  '2021ojn',
           '2021pie',  '2021sbc',  '2021scb',  '2021ued',  '2021uij',  '2021yte',
           '2021zu']
    if remove_bad:
        training_table_in = training_table_in[~np.isin(training_table_in['object_name'], bad)]

    # Shuffle the table rows
    np.random.seed(sorting_state)
    training_table = training_table_in[np.random.permutation(len(training_table_in))]

    # Clean the table by removing objects without a host galaxy
    if clean:
        clean_training = training_table[np.isfinite(training_table['lc_width_r']) &
                                        np.isfinite(training_table['host_Pcc']) & (training_table['input_separation'] > 0)]
    else:
        if model == 'full':
            clean_training = training_table[np.isfinite(training_table['lc_width']) & np.isfinite(training_table['host_Pcc'])]
        elif (model == 'single') or (model == 'double'):
            clean_training = training_table[np.isfinite(training_table['lc_width_r']) & np.isfinite(training_table['host_Pcc'])]

    # Select the feature set
    in_features = feature_set[features]

    # Add the light curve decline for the double model
    if model == 'double':
        use_features = in_features + ['lc_decline_r', 'lc_decline_g']
    else:
        use_features = in_features

    # Create the training data
    training_data = np.array(clean_training[use_features].to_pandas())
    training_classes = np.array(clean_training['object_class'])
    training_names = np.array(clean_training['object_name'])

    # Group transient types
    if grouping == 11:
        training_classes[np.where(training_classes == 'LBV')] = 'Star'
        training_classes[np.where(training_classes == 'Varstar')] = 'Star'
        training_classes[np.where(training_classes == 'CV')] = 'Star'
        training_classes[np.where(training_classes == 'SNIbn')] = 'SNIbc'
        training_classes[np.where(training_classes == 'SNIb')] = 'SNIbc'
        training_classes[np.where(training_classes == 'SNIbc')] = 'SNIbc'
        training_classes[np.where(training_classes == 'SNIc')] = 'SNIbc'
        training_classes[np.where(training_classes == 'SNIc-BL')] = 'SNIbc'
        training_classes[np.where(training_classes == 'SNII')] = 'SNII'
        training_classes[np.where(training_classes == 'SNIIP')] = 'SNII'

        classes_names = {'AGN': 0, 'SLSN-I': 1, 'SLSN-II': 2, 'SNII': 3, 'SNIIb': 4,
                         'SNIIn': 5, 'SNIa': 6, 'SNIbc': 7, 'Star': 8, 'TDE': 9}
    else:
        print(f'Grouping {grouping} not implemented')
        return None, None, None

    # Transform classes into numbers
    training_class = np.array([classes_names[i] for i in training_classes]).astype(int)

    return training_data, training_class, training_names, classes_names


def create_pickle(training_days, grouping, smote_state, n_estimators, max_depth, clf_state, sorting_state, features,
                  model, clean, prefix, overwrite=False, remove_bad=True):
    """
    Create pickle file for the classifier based on the optimal parameters
    of the random forest.

    Parameters
    ----------
    training_days : float
        The number of days of photometry to use for training.
    grouping : str
        The grouping of transient types to use for training.
    smote_state : bool
        The random seed for the SMOTE algorithm.
    n_estimators : int
        The number of trees in the random forest.
    max_depth : int
        The maximum depth of the trees in the random forest.
    clf_state : int
        The random seed for the random forest classifier.
    sorting_state : bool
        The random seed for the sorting algorithm.
    features : int
        The index of the feature list to use for training.
    model : str, either 'single' or 'double'
        The model to use for training.
    clean : bool
        Whether to require objects to have a host galaxy.
    prefix : str
        The prefix for the pickle file name.
    overwrite : bool
        Whether to overwrite existing pickle files. Default is False.
    remove_bad : bool
        Whether to remove bad objects from the training set. Default is True.
    """
    # Create pickle file
    pickle_name = f'{prefix}_{training_days}_{grouping}_{smote_state}_{n_estimators}_{max_depth}_{clf_state}_{sorting_state}_{features}_{model}_{clean}.pkl'
    filename = os.path.join(fleet_data, pickle_name)

    if os.path.exists(filename) and not overwrite:
        print('Pickle file already exists:', filename)
        return
    else:
        print('Creating Pickle for:', training_days, grouping, smote_state, n_estimators, max_depth, clf_state, sorting_state, features, model, clean, prefix)

    # Format the training data
    training_data, training_class, _, _ = format_training(training_days, grouping, sorting_state, features, model, clean=clean, remove_bad=remove_bad)

    # SMOTE the data
    sampler = SMOTE(random_state=smote_state)
    data_train_smote, class_train_smote = sampler.fit_resample(training_data, training_class)

    # Train Random Forest Classifier
    print('Training Classifier...')
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=clf_state)
    clf.fit(data_train_smote, class_train_smote)

    # Save the pickle
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)


def leave_one_out(remove_ind, training_data, training_class, training_names, testing_data, testing_class, testing_names,
                  smote_state, n_estimators, max_depth, clf_state):
    """
    This function will predict the class probabilities of objects in the training set,
    by removing one or more objects at a time and training the classifier on the rest of the data.

    Parameters
    ----------
    remove_ind : int or list/array
        The index or indices of the objects to remove from the training set.
    training_data : np.ndarray
        The training data for the classifier.
    training_class : np.ndarray
        The training classes for the classifier.
    training_names : np.ndarray
        The AT names of each object in the table
    testing_data : np.ndarray
        The testing data for the classifier.
    testing_class : np.ndarray
        The testing classes for the classifier.
    testing_names : np.ndarray
        The AT names of each object in the table
    smote_state : bool
        The random seed for the SMOTE algorithm.
    n_estimators : int
        The number of trees in the random forest.
    max_depth : int
        The maximum depth of the trees in the random forest.
    clf_state : int
        The random seed for the random forest classifier.

    Returns
    -------
    predicted_probs : numpy.ndarray or list of numpy.ndarray
        Array of class probabilities for each removed object
    """
    # Convert remove_ind to array if it's a single index
    if isinstance(remove_ind, (int, np.integer)):
        remove_ind = [remove_ind]

    # Select the objects to be tested
    use_testing = testing_data[remove_ind]
    use_names = testing_names[remove_ind]

    # Remove objects with the same names from the training set
    use_idx = np.where(np.isin(training_names, use_names))[0]
    use_training = np.delete(training_data, use_idx, axis=0)
    use_class = np.delete(training_class, use_idx, axis=0)

    # SMOTE the data
    sampler = SMOTE(random_state=smote_state)
    data_train_smote, class_train_smote = sampler.fit_resample(use_training, use_class)

    # Train Random Forest Classifier
    print(f'Training Classifier without indices {[i+1 for i in remove_ind]}...')
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=clf_state)
    clf.fit(data_train_smote, class_train_smote)

    # Get the class probabilities for the removed objects
    predicted_probs = clf.predict_proba(use_testing)

    # If only one object was removed, return a single array, otherwise return a list of arrays
    return predicted_probs[0] if len(predicted_probs) == 1 else predicted_probs


def leave_one_out_parallel(training_data, training_class, training_names, testing_data, testing_class, testing_names,
                           smote_state, n_estimators, max_depth, clf_state, num_processes=1, chunk_size=1):
    """
    Run leave-one-out cross-validation in parallel, removing chunk_size objects at a time
    and training the classifier on the rest of the data.

    Parameters
    ----------
    training_data : np.ndarray
        The training data for the classifier.
    training_class : np.ndarray
        The training classes for the classifier.
    training_names : np.ndarray
        The AT names of each object in the table
    testing_data : np.ndarray
        The testing data for the classifier.
    testing_class : np.ndarray
        The testing classes for the classifier.
    testing_names : np.ndarray
        The AT names of each object in the table
    smote_state : bool
        The random seed for the SMOTE algorithm.
    n_estimators : int
        The number of trees in the random forest.
    max_depth : int
        The maximum depth of the trees in the random forest.
    clf_state : int
        The random seed for the random forest classifier.
    num_processes : int
        Number of processes to use for parallel processing. Default is 1.
    chunk_size : int
        Number of items to remove at a time during leave-out. Default is 1.

    Returns
    -------
    predicted_probs : list of numpy.ndarray
        List of arrays containing class probabilities for each object in the training set.
    """
    # Create a partial function with fixed arguments
    process_func = partial(leave_one_out,
                           training_data=training_data,
                           training_class=training_class,
                           training_names=training_names,
                           testing_data=testing_data,
                           testing_class=testing_class,
                           testing_names=testing_names,
                           smote_state=smote_state,
                           n_estimators=n_estimators,
                           max_depth=max_depth,
                           clf_state=clf_state)

    # Create indices for all samples in the training data
    indices = list(range(len(testing_data)))

    # Create chunks of indices of size chunk_size
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    # List to store the probability results for each sample
    predicted_probs = [None] * len(testing_data)

    # If single process, use simple loop for better debugging
    if num_processes == 1:
        for chunk in chunks:
            chunk_results = process_func(chunk)
            if len(chunk) == 1:
                # Single result
                predicted_probs[chunk[0]] = chunk_results
            else:
                # Multiple results
                for i, idx in enumerate(chunk):
                    predicted_probs[idx] = chunk_results[i]
    else:
        # Otherwise use Pool for parallel processing
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Map the function to all chunks
            chunk_results = pool.map(process_func, chunks)

            # Process the results
            for chunk, results in zip(chunks, chunk_results):
                if len(chunk) == 1:
                    # Single result
                    predicted_probs[chunk[0]] = results
                else:
                    # Multiple results
                    for i, idx in enumerate(chunk):
                        predicted_probs[idx] = results[i]

    return predicted_probs


def save_pickles(overwrite=False):
    """
    Function that creates and saves the optimal set of pickle files
    for the classifiers. The optimal parameters are based on
    the results of the random forest classifier.

    Parameters
    ----------
    overwrite : bool
        Whether to overwrite existing pickle files. Default is False.
    """

    # Optimal parameters shared among all classifiers
    sorting_states = np.array([38, 39, 40, 41, 42])
    clean = False
    grouping = 11
    n_estimators = 100

    # Optimal parameters for the main late-time classifier
    training_days = 75
    max_depth = 15
    features = 17
    model = 'full'

    # Create the pickle files
    for state in sorting_states:
        create_pickle(training_days, grouping, state, n_estimators, max_depth, state, state, features, model, clean,
                      'main_late', overwrite=overwrite)

    # Optimal parameters for the SLSN rapid classifier
    training_days = 15
    max_depth = 20
    features = 17
    model = 'full'

    # Create the pickle files
    for state in sorting_states:
        create_pickle(training_days, grouping, state, n_estimators, max_depth, state, state, features, model, clean,
                      'slsn_rapid', overwrite=overwrite)

    # Optimal parameters for the TDE rapid classifier
    training_days = 20
    max_depth = 20
    features = 17
    model = 'full'

    # Create the pickle files
    for state in sorting_states:
        create_pickle(training_days, grouping, state, n_estimators, max_depth, state, state, features, model, clean,
                      'tde_rapid', overwrite=overwrite)


def create_validation_table(training_days, testing_days, grouping, smote_state, n_estimators, max_depth, clf_state, sorting_state, features,
                            model, clean, num_processes=1, chunk_size=1, output_dir='validations', overwrite=False, remove_bad=True):
    """
    Run the leave-one-out cross validation method on a testing set, and save or
    return the output diagnosis table.

    Parameters
    ----------
    training_days : float
        The number of days of photometry to use for training.
    testing_days : float
        The number of days of photometry to use for testing.
    grouping : str
        The grouping of transient types to use for training.
    smote_state : bool
        The random seed for the SMOTE algorithm.
    n_estimators : int
        The number of trees in the random forest.
    max_depth : int
        The maximum depth of the trees in the random forest.
    clf_state : int
        The random seed for the random forest classifier.
    sorting_state : bool
        The random seed for the sorting algorithm.
    features : int
        The index of the feature list to use for training.
    model : str, either 'single' or 'double'
        The model to use for training.
    clean : bool
        Whether to require objects to have a host galaxy.
    num_processes : int
        Number of processes to use for parallel processing. Default is 1.
    chunk_size : int
        Number of items to remove at a time during leave-out. Default is 1.
    output_dir : str
        The name of the directory where to save validation tables.
    overwrite : bool
        Whether to overwrite existing pickle files. Default is False.
    remove_bad : bool
        Whether to remove bad objects from the training set. Default is True.
    """

    # Format the output name
    name_format = f'{training_days}_{testing_days}_{grouping}_{smote_state}_{n_estimators}_{max_depth}_{clf_state}' + \
                  f'_{sorting_state}_{features}_{model}_{clean}_{chunk_size}.txt'
    output_name = os.path.join(output_dir, name_format)

    if os.path.exists(output_name) and not overwrite:
        print('Validation table already exists:', output_name)
        return

    # Format the training table
    training_data, training_class, training_names, classes_names = format_training(training_days, grouping, sorting_state, features, model,
                                                                                   clean=clean, remove_bad=remove_bad)

    # Format the testing table
    testing_data, testing_class, testing_names, classes_names = format_training(testing_days, grouping, sorting_state, features, model,
                                                                                clean=clean, remove_bad=False)

    # Predict classes using leave-one-out cross-validation
    predicted_probs = leave_one_out_parallel(training_data, training_class, training_names,
                                             testing_data, testing_class, testing_names,
                                             smote_state=smote_state, n_estimators=n_estimators,
                                             max_depth=max_depth, clf_state=clf_state, num_processes=num_processes,
                                             chunk_size=chunk_size)

    # Get the most likely predicted class of each object
    predicted_classes = np.array([np.argmax(i) for i in predicted_probs])

    # Save output table
    used_classes = list(classes_names.keys())
    output_table = table.Table(np.array(predicted_probs), names=used_classes)
    testing_class_name = np.array(list(classes_names.keys()))[testing_class]
    name_col = table.Table.Column(testing_names, name='object_name')
    true_col = table.Table.Column(testing_class, name='true_class')
    true_name_col = table.Table.Column(testing_class_name, name='true_class_name')
    predicted_col = table.Table.Column(predicted_classes, name='predicted_class')
    output_table.add_columns([name_col, true_col, true_name_col, predicted_col])

    os.makedirs(output_dir, exist_ok=True)
    output_table.write(output_name, format='ascii', overwrite=overwrite)


def predict_probability(info_table, prefix, features, model):
    """
    Predict the probability of each class for a given object.

    Parameters
    ----------
    info_table : astropy.table.Table
        The table containing the information about the object.
    prefix : str
        The prefix for the pickle file name.
    features : list
        The index of the feature list to use for training.
    model : str, either 'single' or 'double'
        The model to use for training.

    Returns
    -------
    probability_avearge : numpy.ndarray
        The average probability of each class.
    probability_std : numpy.ndarray
        The standard deviation of the probability of each class.
    """

    # Select the feature set
    in_features = feature_set[features]

    # Add the light curve decline for the double model
    if model == 'double':
        use_features = in_features + ['lc_decline_r', 'lc_decline_g']
    else:
        use_features = in_features

    # Create the testing data
    testing_data = np.array(info_table[use_features].to_pandas())

    # Find pickle files
    file_directory = os.path.join(fleet_data, f'{prefix}_*.pkl')
    filenames = glob.glob(file_directory)

    predicted_list = []  # Initialize an empty list

    for i in range(len(filenames)):
        filename = filenames[i]
        with open(filename, 'rb') as f:
            clf = pickle.load(f)
        predicted = 100 * clf.predict_proba(testing_data)
        predicted_list.append(predicted[0])

    # Convert list to numpy array after the loop completes
    predicted_array = np.array(predicted_list)

    # Calculate the average and standard deviation
    probability_avearge = np.average(predicted_array, axis=0)
    probability_std = np.std(predicted_array, axis=0)

    return probability_avearge, probability_std


def create_info_table(parameters, output_table, data_catalog, **kwargs):
    """
    Creates a uniform table with all the information about the transient, host, light curve, fit, and input parameters.

    Parameters
    ----------
    parameters : dict
        The parameters from the fit_data function.
    output_table : astropy.table.Table
        The output light curve table from the fit_data function.
    data_catalog : astropy.table.Table
        The output galaxy catalog from the get_catalog function.
    kwargs : dict
        Additional parameters to include in the table.

    Returns
    -------
    info_table : astropy.table.Table
        A table containing information about the transient object.
    """

    # Create an Astropy Table table based on the parameters from parameters
    data = {key: [value] for key, value in parameters.items()}
    info_table = table.Table(data)

    # Calculate the number of observations, detections, and detections used
    lc_length = len(output_table)
    detections = (output_table['UL'] == 'False') & (output_table['Ignore'] == 'False')
    # Bound by phase_min and phase_max
    phase_min = kwargs.get('phase_min')
    phase_max = kwargs.get('phase_max')
    used_length = np.sum((output_table['Phase_boom'] < phase_max) & (output_table['Phase_boom'] > phase_min) &
                         (output_table['UL'] == 'False') & (output_table['Ignore'] == 'False'))
    time_span = np.nanmax(output_table['MJD'][detections]) - np.nanmin(output_table['MJD'][detections])

    info_table['lc_length'] = lc_length
    info_table['det_length'] = np.sum(detections)
    info_table['used_length'] = used_length
    info_table['time_span'] = time_span

    # Calculate the number of sources in the catalog, and whether there is SDSS and PSST data
    if data_catalog:
        num_sources = len(data_catalog)
        has_sdss = 'gPSFMag_3pi' in data_catalog.columns
        has_psst = 'psfMag_g_sdss' in data_catalog.columns
        info_table['num_sources'] = num_sources
        info_table['has_sdss'] = has_sdss
        info_table['has_psst'] = has_psst

    # Add all kwargs to the table with single-value lists
    for key, value in kwargs.items():
        info_table[key] = [value]

    # Calculate delta time
    bright_mjd = kwargs.get('bright_mjd')
    first_mjd = kwargs.get('first_mjd')
    delta_time = bright_mjd - first_mjd
    info_table['delta_time'] = delta_time

    # Calculate additional features
    host_Pcc = kwargs.get('host_Pcc', None)
    host_separation = kwargs.get('host_separation', None)
    force_detection = kwargs.get('force_detection', None)
    host_radius = kwargs.get('host_radius', None)
    host_magnitude_g = kwargs.get('host_magnitude_g', None)
    host_magnitude_r = kwargs.get('host_magnitude_r', None)
    green_brightest = kwargs.get('green_brightest', None)
    red_brightest = kwargs.get('red_brightest', None)

    # Continue only if there was a catalog
    if host_Pcc is not None:
        pcc_pcc_threshold = kwargs.get('pcc_pcc_threshold')
        pcc_distance_threshold = kwargs.get('pcc_distance_threshold')
        if (host_Pcc <= pcc_pcc_threshold) | ((host_Pcc <= 0.07) & (host_separation <= pcc_distance_threshold)) | force_detection:
            input_separation = host_separation
            input_size = host_radius
            input_host_g_r = host_magnitude_g - host_magnitude_r
            normal_separation = input_separation / input_size
            deltamag_green = host_magnitude_g - green_brightest
            deltamag_red = host_magnitude_r - red_brightest
            hostless = False
        else:
            input_separation = 0.0
            input_size = 0.0
            normal_separation = 0.0
            input_host_g_r = 0.0
            deltamag_green = host_limit['g'] - green_brightest
            deltamag_red = host_limit['r'] - red_brightest
            hostless = True

        # Add additional features to the table
        info_table['input_separation'] = input_separation
        info_table['input_size'] = input_size
        info_table['input_host_g_r'] = input_host_g_r
        info_table['normal_separation'] = normal_separation
        info_table['deltamag_green'] = deltamag_green
        info_table['deltamag_red'] = deltamag_red
        info_table['hostless'] = hostless

        # Add survey limit info
        info_table['host_limit_g'] = host_limit['g']
        info_table['host_limit_r'] = host_limit['r']

    # Add redshift labels
    redshift = kwargs.get('redshift', None)
    redshift_in = kwargs.get('redshift_in', None)
    specz = kwargs.get('specz', None)
    photoz = kwargs.get('photoz', None)

    if redshift_in is not None:
        redshift_use = redshift_in
        redshift_label = 'Inputed'
    elif redshift is not None:
        redshift_use = redshift
        redshift_label = 'TNS'
    elif specz is not None:
        redshift_use = specz
        redshift_label = 'Specz'
    elif photoz is not None:
        redshift_use = photoz
        redshift_label = 'Photoz'
    else:
        redshift_use = None
        redshift_label = 'None'

    # Add redshift labels to the table
    info_table['redshift_use'] = redshift_use
    info_table['redshift_label'] = redshift_label

    # Calculate transient peak magnitude in r-band and g-band
    if redshift_use and red_brightest:
        absmag_r = calc_absmag(red_brightest, redshift_use)
    else:
        absmag_r = None
    if redshift_use and green_brightest:
        absmag_g = calc_absmag(green_brightest, redshift_use)
    else:
        absmag_g = None
    info_table['absmag_r'] = absmag_r
    info_table['absmag_g'] = absmag_g

    return info_table


def predict(object_name_in=None, ra_in=None, dec_in=None, object_class_in=None, redshift_in=None, acceptance_radius=3, save_ztf=True,
            download_ztf=True, download_osc=False, read_local=True, query_tns=False, save_lc=True, lc_dir='lightcurves',
            read_existing=False, clean_ignore=True, dust_map='SFD', phase_min=-200, phase_max=75,
            n_walkers=50, n_steps=70, n_cores=1, model='full', late_phase=40, default_err=0.1, default_decline_g=0.55,
            default_decline_r=0.37, burn_in=0.75, sigma_clip=2, repeats=4, save_trace=False, save_lcplot=False, use_median=True,
            search_radius=1.0, reimport_catalog=False, catalog_dir='catalogs', save_catalog=True, use_old=True, Pcc_filter='i',
            Pcc_filter_alternative='r', neighbors=20, recalculate_nature=False, use_glade=False, best_index=None,
            max_separation_glade=60.0, dimmest_glade=16.0, max_pcc_glade=0.01, max_distance_glade=1.0, star_separation=1.0,
            star_cut=0.1, save_params=True, params_dir='parameters', classifier='all', plot_output=True, plot_dir='plots',
            do_observability=True, include_het=False, pupil_fraction=0.3, minimum_halflight=0.7, classify=True, ztf_dir='ztf',
            match_radius_arcsec=1.5, pcc_pcc_threshold=0.02, pcc_distance_threshold=8, n_sigma_limit=3, emcee_progress=True,
            running_live=False, osc_dir='osc', local_dir='photometry'):
    """
    Predicts the classification of an object based on its name, right ascension, and declination.

    Parameters
    ----------
    object_name_in : str, optional
        The name of the transient object to predict.
    ra_in : float, optional
        The right ascension of the transient object in degrees.
    dec_in : float, optional
        The declination of the transient object in degrees.
    object_class_in : str, optional
        The class of the transient object (e.g., 'SLSN-I', 'SLSN-II', etc.).
    redshift_in : float, optional
        The redshift of the transient object.
    acceptance_radius : float, optional
        The radius in arcseconds to accept the object for classification. Default is 3.
    save_ztf : bool, optional
        Whether to save the ZTF light curve data. Default is True.
    download_ztf : bool, optional
        Whether to download the ZTF light curve data. Default is True.
    download_osc : bool, optional
        Whether to download the OSC light curve data. Default is False.
    read_local : bool, optional
        Whether to read local light curve data. Default is True.
    query_tns : bool, optional
        Whether to query the TNS for transient information. Default is False.
    save_lc : bool, optional
        Whether to save the light curve data. Default is True.
    lc_dir : str, optional
        The directory to save the light curve data. Default is 'lightcurves'.
    read_existing : bool, optional
        Whether to read existing light curve data instead of downloading. Default is False.
    clean_ignore : bool, optional
        Whether to clean the light curve data by ignoring certain observations. Default is True.
    dust_map : str, optional
        The dust map to use for extinction correction. Default is 'SFD'.
    phase_min : float, optional
        The minimum phase in days to consider for the light curve fit. Default is -200.
    phase_max : float, optional
        The maximum phase in days to consider for the light curve fit. Default is 75.
    n_walkers : int, optional
        The number of walkers to use in the MCMC fit. Default is 50.
    n_steps : int, optional
        The number of steps to take in the MCMC fit. Default is 70.
    n_cores : int, optional
        The number of CPU cores to use for parallel processing. Default is 1.
    model : str, optional
        The model to use for the light curve fit. Options are 'full', 'single', or 'double'. Default is 'full'.
    late_phase : int, optional
        The late phase in days to consider for the light curve fit. Default is 40.
    default_err : float, optional
        The default error to use for the light curve data. Default is 0.1.
    default_decline_g : float, optional
        The default decline rate in the g-band for the light curve fit. Default is 0.55.
    default_decline_r : float, optional
        The default decline rate in the r-band for the light curve fit. Default is 0.37.
    burn_in : float, optional
        The fraction of steps to discard as burn-in in the MCMC fit. Default is 0.75.
    sigma_clip : float, optional
        The sigma clipping threshold to use for the light curve data. Default is 2.
    repeats : int, optional
        The number of times to repeat the MCMC fit. Default is 4.
    save_trace : bool, optional
        Whether to save the trace of the MCMC fit. Default is False.
    save_lcplot : bool, optional
        Whether to save the light curve plot. Default is False.
    use_median : bool, optional
        Whether to use the median of the light curve data for the fit. Default is True.
    search_radius : float, optional
        The search radius in arcminutes for finding host galaxies. Default is 1.0.
    reimport_catalog : bool, optional
        Whether to reimport the galaxy catalog. Default is False.
    catalog_dir : str, optional
        The directory to save the galaxy catalog. Default is 'catalogs'.
    save_catalog : bool, optional
        Whether to save the galaxy catalog. Default is True.
    use_old : bool, optional
        Whether to use the old galaxy catalog. Default is True.
    Pcc_filter : str, optional
        The filter to use for the PCC calculation. Default is 'i'.
    Pcc_filter_alternative : str, optional
        The alternative filter to use for the PCC calculation. Default is 'r'.
    neighbors : int, optional
        The number of neighbors to consider for the PCC calculation. Default is 20.
    recalculate_nature : bool, optional
        Whether to recalculate the nature of the transient object. Default is False.
    use_glade : bool, optional
        Whether to use the GLADE catalog for host galaxy matching. Default is False.
    best_index : int, optional
        The index of the best host galaxy to use from the GLADE catalog. Default is None.
    max_separation_glade : float, optional
        The maximum separation in arcseconds for the GLADE catalog. Default is 60.0.
    dimmest_glade : float, optional
        The dimmest magnitude in the GLADE catalog. Default is 16.0.
    max_pcc_glade : float, optional
        The maximum PCC value for the GLADE catalog. Default is 0.01.
    max_distance_glade : float, optional
        The maximum distance in Mpc for the GLADE catalog. Default is 1.0.
    star_separation : float, optional
        The separation in arcseconds to consider an object as a star. Default is 1.0.
    star_cut : float, optional
        The cut value for star classification. Default is 0.1.
    save_params : bool, optional
        Whether to save the parameters of the prediction. Default is True.
    params_dir : str, optional
        The directory to save the parameters. Default is 'parameters'.
    classifier : str, optional
        The classifier to use for the prediction. Options are 'all', 'slsn', 'tde', or 'hostless'. Default is 'all'.
    plot_output : bool, optional
        Whether to plot the output of the prediction. Default is True.
    plot_dir : str, optional
        The directory to save the plots. Default is 'plots'.
    do_observability : bool, optional
        Whether to calculate the observability of the transient object. Default is True.
    include_het : bool, optional
        Whether to include the HET in the observability calculation. Default is False.
    pupil_fraction : float, optional
        The fraction of the pupil to use for the observability calculation. Default is 0.3.
    minimum_halflight : float, optional
        The minimum half-light radius in arcseconds for the observability calculation. Default is 0.7.
    classify : bool, optional
        Whether to classify the transient object. Default is True.
    ztf_dir : str, optional
        The directory to save the ZTF data. Default is 'ztf'.
    match_radius_arcsec : float, optional
        The radius in arcseconds to match the transient object with the host galaxy. Default is 1.5.
    pcc_pcc_threshold : float, optional
        The PCC threshold to use for host galaxy classification. Default is 0.02.
    pcc_distance_threshold : float, optional
        The distance threshold in arcseconds to use for host galaxy classification. Default is 8.
    n_sigma_limit : int, optional
        The number of sigma to use for the limit in the light curve fit. Default is 3.
    emcee_progress : bool, optional
        Whether to show the progress of the MCMC fit. Default is True.
    running_live : bool, optional
        Whether the function is running live, default is False.
    osc_dir : str, optional
        The directory to save the OSC data. Default is 'osc'.
    local_dir : str, optional
        The directory to save the local photometry data. Default is 'photometry'.

    Returns
    -------
    info_table : astropy.table.Table
        A table containing all information about the transient object.
    """

    print('\n################# FLEET #################')

    # Empty variable
    info_table = table.Table()

    #########################
    # Basic transient info #
    #########################
    ra_deg, dec_deg, transient_source, object_name, ztf_data, osc_data, local_data, ztf_name, tns_name, object_class, redshift = \
        get_transient_info(object_name_in=object_name_in, ra_in=ra_in, dec_in=dec_in, object_class_in=object_class_in, redshift_in=redshift_in,
                           acceptance_radius=acceptance_radius, save_ztf=save_ztf, download_ztf=download_ztf,
                           download_osc=download_osc, read_local=read_local, query_tns=query_tns, ztf_dir=ztf_dir, lc_dir=lc_dir,
                           osc_dir=osc_dir, local_dir=local_dir)
    print('\nPredicting:', object_name)

    if save_params:
        os.makedirs(params_dir, exist_ok=True)
        failed_path = f'{params_dir}/{object_name}_failed.txt'
        failed_table = table.Table()

    # Stop if it failed
    if ra_deg is None:
        print('Coordinates failed')
        if save_params:
            failed_table.write(failed_path, format='ascii', overwrite=True)
            print(f'\nFailed table saved to {failed_path}')
        return info_table
    elif dec_deg <= -32:
        print('Coordinates are too far south')
        if save_params:
            failed_table.write(failed_path, format='ascii', overwrite=True)
            print(f'\nFailed table saved to {failed_path}')
        return info_table

    ####################
    # Light curve info #
    ####################
    input_table = process_lightcurve(object_name, ra_deg=ra_deg, dec_deg=dec_deg, ztf_data=ztf_data, osc_data=osc_data,
                                     local_data=local_data, save_lc=save_lc, lc_dir=lc_dir, read_existing=read_existing,
                                     clean_ignore=clean_ignore, dust_map=dust_map)

    # Stop if it failed
    if input_table is None:
        print('Light curve failed')
        if save_params:
            failed_table.write(failed_path, format='ascii', overwrite=True)
            print(f'\nFailed table saved to {failed_path}')
        return info_table
    elif len(input_table) == 0:
        print('Light curve is empty')
        if save_params:
            failed_table.write(failed_path, format='ascii', overwrite=True)
            print(f'\nFailed table saved to {failed_path}')
        return info_table
    elif np.sum((input_table['UL'] == 'False') & (input_table['Ignore'] == 'False')) == 0:
        print('No useable data in lightcurve')
        if save_params:
            failed_table.write(failed_path, format='ascii', overwrite=True)
            print(f'\nFailed table saved to {failed_path}')
        return info_table

    ###################
    # Fit Light curve #
    ###################
    (parameters, color_peak, late_color, late_color10,
     late_color20, late_color40, late_color60, first_to_peak_r,
     first_to_peak_g, peak_to_last_r, peak_to_last_g,
     bright_mjd, first_mjd, brightest_mag, green_brightest,
     red_brightest, chi2, chains, output_table) = fit_data(input_table, phase_min=phase_min, phase_max=phase_max, n_walkers=n_walkers, n_steps=n_steps,
                                                           n_cores=n_cores, model=model, late_phase=late_phase, default_err=default_err,
                                                           default_decline_g=default_decline_g, default_decline_r=default_decline_r, burn_in=burn_in,
                                                           sigma_clip=sigma_clip, repeats=repeats, save_trace=save_trace, save_lcplot=save_lcplot,
                                                           use_median=use_median, object_name=object_name, plot_dir=plot_dir, n_sigma_limit=n_sigma_limit,
                                                           emcee_progress=emcee_progress)

    # If the fit failed
    if (color_peak is None) or (green_brightest is None) or (red_brightest is None):
        print('\nLight curve fit failed')

        # Create quick info table
        info_table = create_info_table(parameters, output_table, data_catalog=None, object_name_in=object_name_in, ra_in=ra_in, dec_in=dec_in,
                                       object_class_in=object_class_in, redshift_in=redshift_in, acceptance_radius=acceptance_radius, save_ztf=save_ztf,
                                       download_ztf=download_ztf, download_osc=download_osc, read_local=read_local, query_tns=query_tns, save_lc=save_lc,
                                       read_existing=read_existing, clean_ignore=clean_ignore, dust_map=dust_map,
                                       phase_min=phase_min, phase_max=phase_max, n_walkers=n_walkers, n_steps=n_steps, n_cores=n_cores,
                                       model=model, late_phase=late_phase, default_err=default_err, default_decline_g=default_decline_g,
                                       default_decline_r=default_decline_r, burn_in=burn_in, sigma_clip=sigma_clip, repeats=repeats,
                                       use_median=use_median, search_radius=search_radius,
                                       reimport_catalog=reimport_catalog, save_catalog=save_catalog, use_old=use_old,
                                       Pcc_filter=Pcc_filter, Pcc_filter_alternative=Pcc_filter_alternative, neighbors=neighbors,
                                       recalculate_nature=recalculate_nature, use_glade=use_glade, best_index=best_index,
                                       max_separation_glade=max_separation_glade, dimmest_glade=dimmest_glade, max_pcc_glade=max_pcc_glade,
                                       max_distance_glade=max_distance_glade, star_separation=star_separation, star_cut=star_cut, ra_deg=ra_deg,
                                       dec_deg=dec_deg, transient_source=transient_source, object_name=object_name, ztf_name=ztf_name,
                                       tns_name=tns_name, object_class=object_class, redshift=redshift, color_peak=color_peak, late_color=late_color,
                                       late_color10=late_color10, late_color20=late_color20, late_color40=late_color40, late_color60=late_color60,
                                       first_to_peak_r=first_to_peak_r, first_to_peak_g=first_to_peak_g, peak_to_last_r=peak_to_last_r,
                                       peak_to_last_g=peak_to_last_g, bright_mjd=bright_mjd, first_mjd=first_mjd, brightest_mag=brightest_mag,
                                       green_brightest=green_brightest, red_brightest=red_brightest, chi2=chi2, save_params=save_params, classifier=classifier,
                                       plot_output=plot_output, do_observability=do_observability, classify=classify,
                                       include_het=include_het, pupil_fraction=pupil_fraction, minimum_halflight=minimum_halflight,
                                       pcc_pcc_threshold=pcc_pcc_threshold, pcc_distance_threshold=pcc_distance_threshold, n_sigma_limit=n_sigma_limit)

        # Create quick plot
        if plot_output:
            print('\nCreating quick plot...')
            quick_plot(input_table, info_table, plot_dir=plot_dir)

        # Save the info table
        if save_params:
            os.makedirs(params_dir, exist_ok=True)
            params_path = f'{params_dir}/{object_name}_quick.txt'
            info_table.write(params_path, format='ascii', overwrite=True)
            print(f'\nInfo table saved to {params_path}')

        return info_table

    ######################
    # Catalog Operations #
    ######################
    merged_catalog = get_catalog(object_name, ra_deg, dec_deg, search_radius=search_radius, reimport_catalog=reimport_catalog,
                                 catalog_dir=catalog_dir, save_catalog=save_catalog, use_old=use_old,
                                 match_radius_arcsec=match_radius_arcsec)

    data_catalog = catalog_operations(object_name, merged_catalog, ra_deg, dec_deg, Pcc_filter=Pcc_filter,
                                      Pcc_filter_alternative=Pcc_filter_alternative, neighbors=neighbors,
                                      recalculate_nature=recalculate_nature, dust_map=dust_map,
                                      minimum_halflight=minimum_halflight)

    # Overwrite with GLADE if specified
    if use_glade:
        best_index = overwrite_with_glade(ra_deg, dec_deg, object_name, data_catalog,
                                          max_separation_glade=max_separation_glade, dimmest_glade=dimmest_glade,
                                          max_pcc_glade=max_pcc_glade, max_distance_glade=max_distance_glade)

    (host_radius, host_separation, host_ra, host_dec, host_Pcc, host_magnitude,
     host_magnitude_g, host_magnitude_r, host_nature, photoz, photoz_err, specz,
     specz_err, best_host, force_detection) = get_best_host(data_catalog, star_separation=star_separation,
                                                            star_cut=star_cut, best_index=best_index,
                                                            pcc_pcc_threshold=pcc_pcc_threshold,
                                                            pcc_distance_threshold=pcc_distance_threshold)

    # Get the nearest host galaxy
    closest = np.nanargmin(data_catalog['separation'])
    closest_radius = data_catalog['halflight_radius'][closest]
    closest_separation = data_catalog['separation'][closest]
    closest_ra = data_catalog['ra_matched'][closest]
    closest_dec = data_catalog['dec_matched'][closest]
    closest_Pcc = data_catalog['chance_coincidence'][closest]
    closest_magnitude = data_catalog['effective_magnitude'][closest]
    closest_magnitude_g = data_catalog['host_magnitude_g'][closest]
    closest_magnitude_r = data_catalog['host_magnitude_r'][closest]
    closest_nature = data_catalog['object_nature'][closest]

    # Create the info table
    info_table = create_info_table(parameters, output_table, data_catalog, object_name_in=object_name_in, ra_in=ra_in, dec_in=dec_in,
                                   object_class_in=object_class_in, redshift_in=redshift_in, acceptance_radius=acceptance_radius, save_ztf=save_ztf,
                                   download_ztf=download_ztf, download_osc=download_osc, read_local=read_local, query_tns=query_tns, save_lc=save_lc,
                                   read_existing=read_existing, clean_ignore=clean_ignore, dust_map=dust_map,
                                   phase_min=phase_min, phase_max=phase_max, n_walkers=n_walkers, n_steps=n_steps, n_cores=n_cores,
                                   model=model, late_phase=late_phase, default_err=default_err, default_decline_g=default_decline_g,
                                   default_decline_r=default_decline_r, burn_in=burn_in, sigma_clip=sigma_clip, repeats=repeats,
                                   use_median=use_median, search_radius=search_radius,
                                   reimport_catalog=reimport_catalog, save_catalog=save_catalog, use_old=use_old,
                                   Pcc_filter=Pcc_filter, Pcc_filter_alternative=Pcc_filter_alternative, neighbors=neighbors,
                                   recalculate_nature=recalculate_nature, use_glade=use_glade, best_index=best_index,
                                   max_separation_glade=max_separation_glade, dimmest_glade=dimmest_glade, max_pcc_glade=max_pcc_glade,
                                   max_distance_glade=max_distance_glade, star_separation=star_separation, star_cut=star_cut, ra_deg=ra_deg,
                                   dec_deg=dec_deg, transient_source=transient_source, object_name=object_name, ztf_name=ztf_name,
                                   tns_name=tns_name, object_class=object_class, redshift=redshift, color_peak=color_peak, late_color=late_color,
                                   late_color10=late_color10, late_color20=late_color20, late_color40=late_color40, late_color60=late_color60,
                                   first_to_peak_r=first_to_peak_r, first_to_peak_g=first_to_peak_g, peak_to_last_r=peak_to_last_r,
                                   peak_to_last_g=peak_to_last_g, bright_mjd=bright_mjd, first_mjd=first_mjd, brightest_mag=brightest_mag,
                                   green_brightest=green_brightest, red_brightest=red_brightest, chi2=chi2, host_radius=host_radius,
                                   host_separation=host_separation, host_ra=host_ra, host_dec=host_dec, host_Pcc=host_Pcc, host_magnitude=host_magnitude,
                                   host_magnitude_g=host_magnitude_g, host_magnitude_r=host_magnitude_r, host_nature=host_nature, photoz=photoz,
                                   photoz_err=photoz_err, specz=specz, specz_err=specz_err, best_host=best_host, force_detection=force_detection,
                                   save_params=save_params, classifier=classifier, plot_output=plot_output,
                                   do_observability=do_observability, closest=closest, closest_radius=closest_radius, closest_separation=closest_separation,
                                   closest_ra=closest_ra, closest_dec=closest_dec, closest_Pcc=closest_Pcc, closest_magnitude=closest_magnitude,
                                   closest_magnitude_g=closest_magnitude_g, closest_magnitude_r=closest_magnitude_r, closest_nature=closest_nature,
                                   classify=classify, include_het=include_het, pupil_fraction=pupil_fraction, minimum_halflight=minimum_halflight,
                                   match_radius_arcsec=match_radius_arcsec, pcc_pcc_threshold=pcc_pcc_threshold,
                                   pcc_distance_threshold=pcc_distance_threshold, n_sigma_limit=n_sigma_limit)

    ##################
    # Classification #
    ##################
    if classify:
        print('\nClassifying...')
        if classifier == 'all':
            late_probability_avearge, late_probability_std = predict_probability(info_table, prefix="main_late", features=17, model='full')
            slsn_probability_avearge, slsn_probability_std = predict_probability(info_table, prefix="slsn_rapid", features=17, model='full')
            tdes_probability_avearge, tdes_probability_std = predict_probability(info_table, prefix="tde_rapid", features=17, model='full')

        # Define class labels
        class_labels = ['AGN', 'SLSNI', 'SLSNII', 'SNII', 'SNIIb', 'SNIIn', 'SNIa', 'SNIbc', 'Star', 'TDE']

        # Append probabilities and standard deviations to the info table
        for i, label in enumerate(class_labels):
            # Late-time classifier
            info_table[f'P_late_{label}'] = late_probability_avearge[i]
            info_table[f'P_late_{label}_std'] = late_probability_std[i]
            # SLSN rapid classifier
            info_table[f'P_rapid_slsn_{label}'] = slsn_probability_avearge[i]
            info_table[f'P_rapid_slsn_{label}_std'] = slsn_probability_std[i]
            # TDE rapid classifier
            info_table[f'P_rapid_tde_{label}'] = tdes_probability_avearge[i]
            info_table[f'P_rapid_tde_{label}_std'] = tdes_probability_std[i]

    # Calculate Observability
    if do_observability:
        (telescope_arrays, dates_arrays, airmasses_arrays,
         sun_elevations_arrays, observable_arrays) = calculate_observability(ra_deg, dec_deg, do_observability)
        # Add observability to the info table
        info_table['MMT_observable'] = observable_arrays[telescope_arrays.index('MMT')]
        info_table['Magellan_observable'] = observable_arrays[telescope_arrays.index('Magellan')]
        info_table['McDonald_observable'] = observable_arrays[telescope_arrays.index('McDonald')]
    else:
        telescope_arrays = []
        dates_arrays = []
        airmasses_arrays = []
        sun_elevations_arrays = []
        observable_arrays = []
        info_table['MMT_observable'] = None
        info_table['Magellan_observable'] = None
        info_table['McDonald_observable'] = None

    # Plot the results
    if plot_output:
        print('\nPlotting...')
        os.makedirs(plot_dir, exist_ok=True)
        make_plot(input_table, output_table, data_catalog, info_table, telescope_arrays, dates_arrays, airmasses_arrays, sun_elevations_arrays,
                  plot_dir=plot_dir, do_observability=do_observability, include_het=include_het, pupil_fraction=pupil_fraction)

    # Save the info table
    if save_params:
        os.makedirs(params_dir, exist_ok=True)
        params_path = f'{params_dir}/{object_name}.txt'
        info_table.write(params_path, format='ascii', overwrite=True)
        print(f'\nInfo table saved to {params_path}')

    return info_table


# Define the process_object function outside of create_training_set
def _process_single_object(idx, all_objects, output_dir, phase_max, model, overwrite, skip_quick,
                           download_ztf, **kwargs):
    """Process a single object from the training set."""
    # Read in the light curve
    object_name_in = all_objects['Name'][idx]
    ra_in = all_objects['RA'][idx]
    dec_in = all_objects['DEC'][idx]
    redshift_in = all_objects['Redshift'][idx]
    object_class_in = all_objects['Classification'][idx]

    # Check if quick exits
    skip = False
    if skip_quick:
        if os.path.exists(f'{output_dir}/{object_name_in}_quick.txt'):
            skip = True
        else:
            skip = False

    # Run through FLEET
    output_file = f'{output_dir}/{object_name_in}.txt'
    if (os.path.exists(output_file) or skip) and not overwrite:
        print(f'Training set already exists: {object_name_in}')
        return None
    else:
        try:
            return predict(object_name_in=object_name_in, ra_in=ra_in, dec_in=dec_in,
                           object_class_in=object_class_in, redshift_in=redshift_in,
                           phase_max=phase_max, model=model, params_dir=output_dir,
                           query_tns=False, do_observability=False, classify=False,
                           download_ztf=download_ztf, read_existing=True, **kwargs)
        except Exception as e:
            print(f"Error processing {object_name_in}: {e}")
            return None


def combine_training_set(params_dir, phase_max=None, model=None, overwrite_out=False,
                         output_filename='table_combined'):
    """
    Function that combines all tables created by create_training_set into one big table,
    excluding '_quick.txt' files.

    Parameters
    ----------
    params_dir : str
        The directory to save the training set.
    phase_max : int
        The maximum phase to use for training.
    model : str
        The model to use for training.
    overwrite_out : bool
        Whether to overwrite the output combined table.
    output_filename : str
        The name of the output combined table file.
    """

    # Find all files
    if (phase_max is None) and (model is None):
        output_dir = f'{params_dir}'
    else:
        output_dir = f'{params_dir}_{model}_{phase_max}'
    file_directory = os.path.join(output_dir, '*.txt')
    filenames = glob.glob(file_directory)
    filenames = [f for f in filenames if ('_quick.txt' not in f) and ('_failed.txt' not in f)]

    # Get column names from the first file to set up the structure
    with open(filenames[0], 'r') as f:
        header = f.readline().strip()
        column_names = header.split()

    # Prepare lists to store each row's data
    all_rows = []

    # Process each file
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            # Skip header line
            f.readline()
            # Read data line (assuming one row per file)
            data_line = f.readline().strip()
            if data_line:  # Check if the line isn't empty
                row_data = data_line.split()
                if len(row_data) == len(column_names):  # Ensure the number of columns matches
                    all_rows.append(row_data)
                    print(f'Processed {filename} - {i+1}/{len(filenames)}')
                else:
                    print(f"Skipping {filename} due to column mismatch: expected {len(column_names)}, got {len(row_data)}")

    # Create the final table directly from all rows
    final_table = table.Table(rows=all_rows, names=column_names)

    # Write output table
    if (phase_max is None) and (model is None):
        if output_filename.endswith('.txt'):
            output_filename = output_filename[:-4]
        combined_dir = f'{output_filename}.txt'
    else:
        combined_dir = f'{params_dir}/table_{model}_{phase_max}.txt'
    if not os.path.exists(combined_dir) or overwrite_out:
        final_table.write(combined_dir, format='ascii', overwrite=True)
        print(f"\nCombined {len(all_rows)} files into {combined_dir}")
    else:
        print(f"\nCombined file already exists: {combined_dir}")


def create_training_set(phase_max=70, model='double', params_dir='training_set',
                        overwrite=False, num_processes=1, object_list=None,
                        skip_quick=False, overwrite_out=False, save_combined=True,
                        download_ztf=False, all_objects=None, **kwargs):
    """
    Create a training set for the classifier based on a list of
    classified transients. The training set is created by running
    FLEET on each transient in the list and saving the results
    to a table.

    Parameters
    ----------
    phase_max : int
        The maximum phase to use for training. Default is 70.
    model : str
        The model to use for training. Default is 'double'.
        Can be 'single', 'double', or 'full'.
    params_dir : str
        The directory to save the training set. Default is 'training_set'.
    overwrite : bool
        Whether to overwrite existing training set files. Default is False.
    num_processes : int
        Number of processes to use for parallel processing. Default is 1.
    object_list : list
        List of objects to process. If None, all objects in the training set will be processed.
    skip_quick : bool
        Also remove the _quick files from the done files.
    overwrite_out : bool
        Whether to overwrite the output combined table
    kwargs : dict
        Additional parameters to pass to the predict function.
    save_combined : bool
        Whether to save the combined table after processing. Default is True.
    download_ztf : bool
        Whether or not to query Alerce
    all_objects : astropy.Table
        Table with input objects
    """
    # Read in the list of SNe that will be used for training
    if all_objects is None:
        reference_transients = pkg_resources.resource_filename(__name__, 'sne_best.txt')
        all_objects = table.Table.read(reference_transients, format='ascii')
        if object_list is not None:
            all_objects = all_objects[np.isin(all_objects['Name'], object_list)]

    # Set up output directory
    output_dir = f'{params_dir}_{model}_{phase_max}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a partial function with fixed arguments
    process_func = partial(_process_single_object,
                           all_objects=all_objects,
                           output_dir=output_dir,
                           phase_max=phase_max,
                           model=model,
                           overwrite=overwrite,
                           skip_quick=skip_quick,
                           download_ztf=download_ztf,
                           **kwargs)

    # If single process, use simple loop for better debugging
    if num_processes == 1:
        for i in range(len(all_objects)):
            process_func(i)

        # Combine all tables
        if save_combined:
            combine_training_set(params_dir, phase_max, model, overwrite_out)

        return

    # Otherwise use Pool for parallel processing
    indices = list(range(len(all_objects)))
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the function to all indices
        _ = pool.map(process_func, indices)

    # Combine all tables
    if save_combined:
        combine_training_set(params_dir, phase_max, model, overwrite_out)
