from astropy import table
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import datetime
from sklearn.metrics import confusion_matrix
import itertools
from multiprocessing import Pool
import glob
import sys
import os

def do_everything(training_days = 15, testing_days = 15, grouping = 6, SMOTE_state = 40, n_estimators = 10, max_depth = 5, clf_state = 40, sorting_state = 40, features = 3, model = 0, clean = 0, n_cores = 1, center_or_peak = 'center', plot_importance = False):
    '''
    Use a random forest classifier to test a given data set with a (if you want) different data set.

    Parameters
    ----------------------------
    training_days  : read file with these days for training data
    testing_days   : But test on file with these many days
    center_or_peak : use the 'center' or the 'peak' of the emcee
    grouping       : integer, which grouping from the options to use
    SMOTE_state    : random SMOTE seed number
    n_estimators   : number of estimators for random forest
    max_depth      : Max depth of random forest
    clf_state      : random seed for random forest
    sorting_state  : random seed for shuffling the training file
    n_cores        : How many cores to use for parallel running
    features       : 0 = no delta_observed (in time) + separation/radius + deltamag
                     1 = with delta_observed
                     2 = with delta_observed and deltamag
                     3 = with delta_observed and separation/radius
                     4 = No contextual information
                     5 = Just Amplitude
                     6 = Just Contextual Information
    model          : 0 = simple model with single exponential
                     1 = model with exponential + decline
    clean          : 0 = Remove all the objects not in 3PI
                     1 = Remove all the objects not in 3PI AND all the orphans

    Output
    ----------------------------
    Saves a confusion matrix and astropy Table with all the data
    '''

    name = 'all_data/output_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.txt'%(training_days, testing_days, center_or_peak, grouping, SMOTE_state, n_estimators, max_depth, clf_state, sorting_state, features, model, clean)
    print(training_days, testing_days, center_or_peak, grouping, SMOTE_state, n_estimators, max_depth, clf_state, sorting_state, features, model, clean)

    if (len(glob.glob(name)) > 0) & (plot_importance == False):
        print('exists')
        return

    def classify_object(excluded_object, i, sample = False):
        print(i, excluded_object)
        # Create Table without object
        data_train  = training_data [training_names != excluded_object]
        class_train = training_class[training_names != excluded_object]

        # Create Table with only object
        data_test  = testing_data [testing_names == excluded_object]

        # SMOTE the data
        sampler = SMOTE(random_state=SMOTE_state)
        data_train_smote, class_train_smote = sampler.fit_resample(data_train, class_train)

        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=clf_state)
        clf.fit(data_train_smote, class_train_smote)

        # Predict Excluded Object
        predicted_probability = clf.predict_proba(data_test)[0]

        if sample:
            return clf

        if len(predicted_probability) == n_classes:
            return predicted_probability.tolist()
        else:
            return np.array(np.nan * np.ones(n_classes)).tolist()

    # Import Data
    if model == 0:
        training_table_in = table.Table.read('training_set/%s_table_%s_single.txt'%(center_or_peak, training_days), format = 'ascii')
        testing_table_in  = table.Table.read('training_set/%s_table_%s_single.txt'%(center_or_peak, testing_days ), format = 'ascii')
    elif model == 1:
        training_table_in = table.Table.read('training_set/%s_table_%s_double.txt'%(center_or_peak, training_days), format = 'ascii')
        testing_table_in  = table.Table.read('training_set/%s_table_%s_double.txt'%(center_or_peak, testing_days ), format = 'ascii')

    # Remove bad objects from training sample
    bad = ['2020cui','2019lwy','2019cvi','2018jsc','2005bf' ,'2005gi' ,'2007ib' ,'2008aq' ,
           '2008ax' ,'2009N'  ,'2010id' ,'2012aw' ,'2013ai' ,'2013am' ,'2013bu' ,'2013ej' ,
           '2013fs' ,'2016X'  ,'2018cyg','2018epm','2018fjw','2018fii','2018fuw','2018gvt',
           '2018imj','2018lcd','2019B'  ,'2019bvq','2019cda','2019ci' ,'2019dok','2019gaf',
           '2019gqk','2019hau','2019iex','2019keo','2019lkw','2019oa' ,'2019otb','2019pjs',
           '2019sjx','2019tqb','2019wbg','2020ekk']
    good = [i not in bad for i in training_table_in['object_name']]
    training_table_in = training_table_in[good]

    # Shuffle Order of Table
    order = np.arange(len(training_table_in))
    np.random.seed(sorting_state)
    np.random.shuffle(order)

    # Randomize
    training_table = training_table_in[order]
    testing_table  = testing_table_in

    # Select Only Clean Data
    if clean == 0:
        clean_training = training_table[np.isfinite(training_table['red_amplitude']) & np.isfinite(training_table['Pcc'])]
        clean_testing  = testing_table [np.isfinite(testing_table ['red_amplitude']) & np.isfinite(testing_table ['Pcc'])]
    if clean == 1:
        clean_training = training_table[np.isfinite(training_table['red_amplitude']) & np.isfinite(training_table['Pcc']) & (training_table['Pcc'] <= 0.1)]
        clean_testing  = testing_table [np.isfinite(testing_table ['red_amplitude']) & np.isfinite(testing_table ['Pcc']) & (testing_table ['Pcc'] <= 0.1)]

    # Read Only Relevant Data
    if model == 0:
        if features == 0 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                                      'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 1 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 2 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   ,                       'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 3 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                               ].to_pandas())
        if features == 4 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   ,                                                                                   ].to_pandas())
        if features == 5 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                                                                                                                        ].to_pandas())
        if features == 6 : training_data = np.array(clean_training       [                                                                                           'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 7 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                                      'normal_separation'                                                               ].to_pandas())
        if features == 8 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'color'                  ].to_pandas())
        if features == 9 : training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'color'                  ].to_pandas())
        if features == 10: training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'Pcc'     ].to_pandas())
        if features == 11: training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'redshift'].to_pandas())
        if features == 12: training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'absmag'  ].to_pandas())
        if features == 13: training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'model_color'            ].to_pandas())
        if features == 14: training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ].to_pandas())
        if features == 15: training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                                      'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ].to_pandas())
        if features == 16: training_data = np.array(clean_training       ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'model_color', 'redshift'].to_pandas())
        training_names    = np.array(clean_training['object_name'])
        training_class_in = np.array(clean_training['class'])

        if features == 0 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                                      'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 1 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 2 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   ,                       'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 3 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                               ].to_pandas())
        if features == 4 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   ,                                                                                   ].to_pandas())
        if features == 5 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                                                                                                                        ].to_pandas())
        if features == 6 : testing_data = np.array(clean_testing         [                                                                                           'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 7 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                                      'normal_separation'                                                               ].to_pandas())
        if features == 8 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'color'                  ].to_pandas())
        if features == 9 : testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'color'                  ].to_pandas())
        if features == 10: testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'Pcc'     ].to_pandas())
        if features == 11: testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'redshift'].to_pandas())
        if features == 12: testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                                   , 'absmag'  ].to_pandas())
        if features == 13: testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'model_color'            ].to_pandas())
        if features == 14: testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ].to_pandas())
        if features == 15: testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                                      'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ].to_pandas())
        if features == 16: testing_data = np.array(clean_testing         ['red_amplitude',                   'green_amplitude',                     'delta_time'   , 'normal_separation'                                    , 'model_color', 'redshift'].to_pandas())
        testing_names     = np.array(clean_testing ['object_name'])
        testing_class_in  = np.array(clean_testing ['class'])

    elif model == 1:
        if features == 0 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2',                  'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 1 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 2 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   ,                       'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 3 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                                               ].to_pandas())
        if features == 4 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   ,                                                                                   ].to_pandas())
        if features == 5 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2',                                                                                                    ].to_pandas())
        if features == 6 : training_data = np.array(clean_training       [                                                                                           'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 7 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2',                  'normal_separation'                                                               ].to_pandas())
        if features == 8 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                    , 'color'                  ].to_pandas())
        if features == 9 : training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'color'                  ].to_pandas())
        if features == 10: training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                                   , 'Pcc'     ].to_pandas())
        if features == 11: training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                                   , 'redshift'].to_pandas())
        if features == 12: training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                                   , 'absmag'  ].to_pandas())
        if features == 13: training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                    , 'model_color'            ].to_pandas())
        if features == 14: training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ].to_pandas())
        if features == 15: training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2',                  'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ].to_pandas())
        if features == 16: training_data = np.array(clean_training       ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                    , 'model_color', 'redshift'].to_pandas())
        training_names    = np.array(clean_training['object_name'])
        training_class_in = np.array(clean_training['class'])

        if features == 0 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2',                  'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 1 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 2 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   ,                       'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 3 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                                               ].to_pandas())
        if features == 4 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   ,                                                                                   ].to_pandas())
        if features == 5 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2',                                                                                                    ].to_pandas())
        if features == 6 : testing_data = np.array(clean_testing         [                                                                                           'normal_separation' , 'deltamag_red', 'deltamag_green'                            ].to_pandas())
        if features == 7 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2',                  'normal_separation'                                                               ].to_pandas())
        if features == 8 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                    , 'color'                  ].to_pandas())
        if features == 9 : testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'color'                  ].to_pandas())
        if features == 10: testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                                   , 'Pcc'     ].to_pandas())
        if features == 11: testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                                   , 'redshift'].to_pandas())
        if features == 12: testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                                   , 'absmag'  ].to_pandas())
        if features == 13: testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                    , 'model_color'            ].to_pandas())
        if features == 14: testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ].to_pandas())
        if features == 15: testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2',                  'normal_separation' , 'deltamag_red', 'deltamag_green' , 'model_color'            ].to_pandas())
        if features == 16: testing_data = np.array(clean_testing         ['red_amplitude', 'red_amplitude2', 'green_amplitude', 'green_amplitude2', 'delta_time'   , 'normal_separation'                                    , 'model_color', 'redshift'].to_pandas())
        testing_names     = np.array(clean_testing ['object_name'])
        testing_class_in  = np.array(clean_testing ['class'])

    # Grouping 0 [LBVs + Varstars] and [SNIb + SNIbn]
    if grouping == 0:
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn')] = 'SNIb'
        training_class_in[np.where(training_class_in == 'SNIb' )] = 'SNIb'
        testing_class_in [np.where(testing_class_in  == 'SNIbn')] = 'SNIb'
        testing_class_in [np.where(testing_class_in  == 'SNIb' )] = 'SNIb'

        classes_names = {'AGN'   :  0, 'CV'    :  1, 'SLSN-I'  :  2, 'SLSN-II' :  3, 'SNII' :  4,
                         'SNIIP' :  5, 'SNIIb' :  6, 'SNIIn'   :  7, 'SNIa'    :  8, 'SNIb' :  9,
                         'SNIbc' : 10, 'SNIc'  : 11, 'SNIc-BL' : 12, 'TDE'     : 13, 'Star' : 14}

    elif grouping == 1:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn')] = 'SNIb'
        training_class_in[np.where(training_class_in == 'SNIb' )] = 'SNIb'
        testing_class_in [np.where(testing_class_in  == 'SNIbn')] = 'SNIb'
        testing_class_in [np.where(testing_class_in  == 'SNIb' )] = 'SNIb'

        classes_names = {'AGN'   :  0, 'SLSN-I'  :  1, 'SLSN-II' :  2, 'SNII' :  3, 'SNIIP' :  4,
                         'SNIIb' :  5, 'SNIIn'   :  6, 'SNIa'    :  7, 'SNIb' :  8, 'SNIbc' :  9,
                         'SNIc'  : 10, 'SNIc-BL' : 11, 'TDE'     : 12, 'Star' : 13}

    elif grouping == 2:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'SNIbc'

        classes_names = {'AGN'   :  0, 'SLSN-I' :  1, 'SLSN-II' : 2, 'SNII'  : 3, 'SNIIP' : 4,
                         'SNIIb' :  5, 'SNIIn'  :  6, 'SNIa'    : 7, 'SNIbc' : 8, 'TDE'   : 9,
                         'Star'  : 10}

    elif grouping == 3:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'SNIbc'

        training_class_in[np.where(training_class_in == 'SNII'   )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNII'   )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIP'  )] = 'SNII'

        classes_names = {'AGN'   : 0, 'SLSN-I' : 1, 'SLSN-II' : 2, 'SNII' : 3, 'SNIIb' : 4,
                         'SNIIn' : 5, 'SNIa'   : 6, 'SNIbc'   : 7, 'TDE'  : 8, 'Star'  : 9}

    elif grouping == 4:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'SNIbc'

        training_class_in[np.where(training_class_in == 'SNII'   )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNII'   )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIP'  )] = 'SNII'

        training_class_in[np.where(training_class_in == 'TDE'    )] = 'Nuclear'
        training_class_in[np.where(training_class_in == 'AGN'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'TDE'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'AGN'    )] = 'Nuclear'

        classes_names = {'Nuclear' : 0, 'SLSN-I' : 1, 'SLSN-II' : 2, 'SNII'    : 3, 'SNIIb'   : 4,
                         'SNIIn'   : 5, 'SNIa'   : 6, 'SNIbc'   : 7, 'Star' : 8}

    elif grouping == 5:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'SNIbc'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'SNIbc'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'SNIbc'

        training_class_in[np.where(training_class_in == 'SNII'   )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIb'  )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIn'  )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNII'   )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIb'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIn'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIP'  )] = 'SNII'

        training_class_in[np.where(training_class_in == 'TDE'    )] = 'Nuclear'
        training_class_in[np.where(training_class_in == 'AGN'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'TDE'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'AGN'    )] = 'Nuclear'

        classes_names={'Nuclear':0, 'SLSN-I':1, 'SLSN-II':2, 'SNII':3, 'SNIa':4, 'SNIbc':5, 'Star':6}

    elif grouping == 6:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIa'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIa'   )] = 'SNI'

        training_class_in[np.where(training_class_in == 'SNII'   )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIb'  )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIn'  )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNII'   )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIb'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIn'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIP'  )] = 'SNII'

        training_class_in[np.where(training_class_in == 'TDE'    )] = 'Nuclear'
        training_class_in[np.where(training_class_in == 'AGN'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'TDE'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'AGN'    )] = 'Nuclear'

        classes_names={'Nuclear':0, 'SLSN-I':1, 'SLSN-II':2, 'SNII':3, 'SNI':4, 'Star':5}

    elif grouping == 7:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIa'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIa'   )] = 'SNI'

        training_class_in[np.where(training_class_in == 'SNII'   )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIb'  )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIn'  )] = 'SNII'
        training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNII'   )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIb'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIn'  )] = 'SNII'
        testing_class_in [np.where(testing_class_in  == 'SNIIP'  )] = 'SNII'

        training_class_in[np.where(training_class_in == 'TDE'    )] = 'Nuclear'
        training_class_in[np.where(training_class_in == 'AGN'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'TDE'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'AGN'    )] = 'Nuclear'

        training_class_in[np.where(training_class_in == 'SLSN-I' )] = 'SLSN'
        training_class_in[np.where(training_class_in == 'SLSN-II')] = 'SLSN'
        testing_class_in [np.where(testing_class_in  == 'SLSN-I' )] = 'SLSN'
        testing_class_in [np.where(testing_class_in  == 'SLSN-II')] = 'SLSN'

        classes_names={'Nuclear':0, 'SLSN':1, 'SNII':2, 'SNI':3, 'Star':4}

    elif grouping == 8:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Star'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Star'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Star'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Star'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIa'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIa'   )] = 'SNI'

        training_class_in[np.where(training_class_in == 'SNII'   )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIIb'  )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIIn'  )] = 'SNI'
        training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNII'   )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIIb'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIIn'  )] = 'SNI'
        testing_class_in [np.where(testing_class_in  == 'SNIIP'  )] = 'SNI'

        training_class_in[np.where(training_class_in == 'TDE'    )] = 'Nuclear'
        training_class_in[np.where(training_class_in == 'AGN'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'TDE'    )] = 'Nuclear'
        testing_class_in [np.where(testing_class_in  == 'AGN'    )] = 'Nuclear'

        classes_names={'Nuclear':0, 'SLSN-I':1, 'SLSN-II':2, 'SNI':3, 'Star':4}

    elif grouping == 9:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Other'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Other'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Other'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIa'   )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIa'   )] = 'Other'

        training_class_in[np.where(training_class_in == 'SNII'   )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIIb'  )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIIn'  )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNII'   )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIIb'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIIn'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIIP'  )] = 'Other'

        training_class_in[np.where(training_class_in == 'TDE'    )] = 'Other'
        training_class_in[np.where(training_class_in == 'AGN'    )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'TDE'    )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'AGN'    )] = 'Other'

        classes_names={'SLSN-I':0, 'SLSN-II':1, 'Other':2}

    elif grouping == 10:
        training_class_in[np.where(training_class_in == 'LBV'    )] = 'Other'
        training_class_in[np.where(training_class_in == 'Varstar')] = 'Other'
        training_class_in[np.where(training_class_in == 'CV'     )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'LBV'    )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'Varstar')] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'CV'     )] = 'Other'

        training_class_in[np.where(training_class_in == 'SNIbn'  )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIb'   )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIbc'  )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIc'   )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIc-BL')] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIa'   )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIbn'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIb'   )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIbc'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIc'   )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIc-BL')] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIa'   )] = 'Other'

        training_class_in[np.where(training_class_in == 'SNII'   )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIIb'  )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIIn'  )] = 'Other'
        training_class_in[np.where(training_class_in == 'SNIIP'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNII'   )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIIb'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIIn'  )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'SNIIP'  )] = 'Other'

        training_class_in[np.where(training_class_in == 'TDE'    )] = 'Other'
        training_class_in[np.where(training_class_in == 'AGN'    )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'TDE'    )] = 'Other'
        testing_class_in [np.where(testing_class_in  == 'AGN'    )] = 'Other'

        training_class_in[np.where(training_class_in == 'SLSN-I' )] = 'SLSN'
        training_class_in[np.where(training_class_in == 'SLSN-II')] = 'SLSN'
        testing_class_in [np.where(testing_class_in  == 'SLSN-I' )] = 'SLSN'
        testing_class_in [np.where(testing_class_in  == 'SLSN-II')] = 'SLSN'

        classes_names={'SLSN':0, 'Other':1}

    training_class = np.array([classes_names[i] for i in training_class_in]).astype(int)
    testing_class  = np.array([classes_names[i] for i in testing_class_in ]).astype(int)

    # Numbers of Objects per class
    testing_count  = np.array([len(np.where(testing_class_in  == i)[0]) for i in classes_names])
    training_count = np.array([len(np.where(training_class_in == i)[0]) for i in classes_names])

    # Number of Classes used
    n_classes = len(np.unique(training_class))

    if plot_importance:
        import matplotlib.pyplot as plt
        from sklearn.inspection import permutation_importance
        from scipy.stats import spearmanr
        from scipy.cluster import hierarchy
        from matplotlib.pyplot import cm
        from matplotlib import rcParams
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 14

        # Calculate Basic Importances
        clf = classify_object(testing_names[0], 0, sample = True)
        importances = clf.feature_importances_

        # Get Training data and labels
        excluded_object = testing_names[0]
        data_train  = training_data [training_names != excluded_object]
        class_train = training_class[training_names != excluded_object]

        # Calculate Correlated Importances
        result = permutation_importance(clf, data_train, class_train, n_repeats=10, random_state=clf_state)
        # Sort
        perm_sorted_idx = result.importances_mean.argsort()
        tree_importance_sorted_idx = np.argsort(importances)
        tree_indices = np.arange(0, len(importances)) + 0.5

        # Plot
        if features == 3:
            names = np.array([r'$W_r$',r'$W_g$',r'$\Delta t$',r'$R_n$'])
        elif features == 9:
            names = np.array([r'$W_r$',r'$W_{r2}$',r'$W_g$',r'$W_{g2}$',r'$\Delta t$',r'$R_n$',r'$\Delta m_r$',r'$\Delta m_g$','g-r'])
        elif features == 13:
            names = np.array([r'$W_r$',r'$W_g$',r'$\Delta t$',r'$R_n$','g-r'])
        plt.boxplot(result.importances[perm_sorted_idx].T, vert=False,labels=names[perm_sorted_idx])
        plt.xlabel('Correlated Importance')
        plt.savefig('Importance_%s_%s.pdf'%(training_days, features), bbox_inches = 'tight')
        plt.clf()

        # Plot Correlation Matrix
        corr = spearmanr(data_train).correlation
        corr_linkage = hierarchy.ward(corr)
        dendro = hierarchy.dendrogram(corr_linkage, labels=names, leaf_rotation=90, no_plot = True)
        dendro_idx = np.arange(0, len(dendro['ivl']))

        #plt.title('Features #%s     Training = %sd'%(features, training_days))
        plt.imshow(np.abs(corr[dendro['leaves'], :][:, dendro['leaves']]), cmap = cm.Greens)
        plt.xticks(dendro_idx, dendro['ivl'])
        plt.yticks(dendro_idx, dendro['ivl'])
        plt.colorbar(label = 'Correlation')
        plt.tight_layout()
        plt.savefig('Correlation_%s_%s.pdf'%(training_days, features), bbox_inches = 'tight')
        plt.clf()

        return 

    # Classify in Multiprocess
    Start = datetime.datetime.now()
    samples_all = np.array([testing_names, np.arange(len(testing_names))])
    if n_cores == 1:
        big_table_classes = np.array([classify_object(*i) for i in samples_all.T])
    else:
        pool  = Pool(n_cores)
        output_pool = pool.starmap(classify_object, samples_all.T)
        big_table_classes = np.hstack(output_pool).reshape(len(clean_testing), n_classes)
    End = datetime.datetime.now()
    print("It took = " + str(End - Start))

    # Test One
    clf = classify_object(testing_names[0], 0, sample = True)

    # Classes used
    classes_used    = clf.classes_
    predicted_label = [classes_used[np.argmax(i)] for i in big_table_classes]
    names_used      = np.array([i for i in classes_names])#[classes_used]

    # Generate Output Table
    output_table  = table.Table(big_table_classes      , names = names_used[classes_used])
    name_col      = table.Table.Column(testing_names   , name = 'name'     )
    class_col     = table.Table.Column(testing_class_in, name = 'class'    )
    true_col      = table.Table.Column(testing_class   , name = 'true'     )
    predicted_col = table.Table.Column(predicted_label , name = 'predicted')
    output_table.add_columns([name_col, class_col, true_col, predicted_col])

    if len(glob.glob('all_data')) == 0:
        os.system('mkdir all_data')

    output_table.write('all_data/output_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.txt'%(training_days, testing_days, center_or_peak, grouping, SMOTE_state, n_estimators, max_depth, clf_state, sorting_state, features, model, clean), format = 'ascii')

def create_array(training_days_array, testing_days_array, grouping_array, SMOTE_state_array, n_estimators_array, max_depth_array, clf_state_array, sorting_state_array, features_array, models_array, clean_array, match_seed = False):
    # Create Input Array
    input_array_in = np.zeros(11)

    # Match the seed number of SMOTE and Shuffler
    if match_seed:
        for a in training_days_array:
            for b in testing_days_array:
                for c in grouping_array:
                    for d in SMOTE_state_array:
                        for e in n_estimators_array:
                            for f in max_depth_array:
                                for g in clf_state_array:
                                    for i in features_array:
                                        for j in models_array:
                                            for k in clean_array:
                                                input_array_in = np.vstack([input_array_in, np.array([a, b, c, d, e, f, g, d, i, j, k])])

    else:
        for a in training_days_array:
            for b in testing_days_array:
                for c in grouping_array:
                    for d in SMOTE_state_array:
                        for e in n_estimators_array:
                            for f in max_depth_array:
                                for g in clf_state_array:
                                    for h in sorting_state_array:
                                        for i in features_array:
                                            for j in models_array:
                                                for k in clean_array:
                                                    input_array_in = np.vstack([input_array_in, np.array([a, b, c, d, e, f, g, h, i, j, k])])

    input_array = input_array_in[1:].astype(int)

    return input_array

# Shared Parameters
grouping_array       = np.array([4])
SMOTE_state_array    = np.array([38,39,40,41,42])
n_estimators_array   = np.array([100])
clf_state_array      = np.array([38,39,40,41,42])
sorting_state_array  = np.array([38,39,40,41,42])

# Individual Parameters
training_days_array  = np.array([20])
testing_days_array   = np.array([20])
max_depth_array      = np.array([7])
features_array       = np.array([13,16])
models_array         = np.array([0])
clean_array          = np.array([0])
input_array_rapid    = create_array(training_days_array, testing_days_array, grouping_array, SMOTE_state_array, n_estimators_array, max_depth_array, clf_state_array, sorting_state_array, features_array, models_array, clean_array, match_seed = True)

training_days_array  = np.array([70])
testing_days_array   = np.array([70])
max_depth_array      = np.array([9])
features_array       = np.array([7])
models_array         = np.array([1])
clean_array          = np.array([0,1])
input_array_full     = create_array(training_days_array, testing_days_array, grouping_array, SMOTE_state_array, n_estimators_array, max_depth_array, clf_state_array, sorting_state_array, features_array, models_array, clean_array, match_seed = True)

# Run On Odyssey
pool  = Pool(25)
index = float(sys.argv[1])

if index == 1: pool.starmap(do_everything, input_array_rapid)
if index == 2: pool.starmap(do_everything, input_array_full )