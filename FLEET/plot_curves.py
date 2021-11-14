from astropy import table
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib.pyplot import cm
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
fontsize = 12

# Create Colors for Plotting
colors    = np.ndarray((6,4))
colors[0] = plt.cm.viridis_r(np.linspace(0.05,1.0,5))[0]
colors[1] = plt.cm.viridis_r(np.linspace(0.05,1.0,5))[1]
colors[2] = plt.cm.viridis_r(np.linspace(0.05,1.0,5))[2]
colors[3] = plt.cm.viridis_r(np.linspace(0.05,1.0,5))[3]
colors[4] = plt.cm.viridis_r(np.linspace(0.05,1.0,5))[4]
colors[5] = [0.8627451 , 0.07843137, 0.23529412, 1.]
alpha     = 0.7

def pure_complete(training_days_array, testing_days_array, grouping_array, SMOTE_state_array, n_estimators_array, max_depth_array, clf_state_array, sorting_state_array, features_array, models_array, clean_array, true_ratios = True, match_seed = False):
    '''
    Calculate the curve of pureness and completeness with errorbars for the given 
    set of parameters. Averaged over all of them.
    '''

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

    # Classes
    # ['Nuclear', 'SLSN-I', 'SLSN-II', 'SNII', 'SNI', 'Star']
    # [0            1           2         3      4      5   ]

    all_purity = np.ones(100) * np.nan
    all_loss   = np.ones(100) * np.nan

    for j in range(len(input_array)):
        print(j, '/', len(input_array))
        
        # File name
        training_days  = input_array[j][0]
        testing_days   = input_array[j][1]
        center_or_peak = 'center'
        grouping       = input_array[j][2]
        SMOTE_state    = input_array[j][3]
        n_estimators   = input_array[j][4]
        max_depth      = input_array[j][5]
        clf_state      = input_array[j][6]
        sorting_state  = input_array[j][7]
        features       = input_array[j][8]
        models         = input_array[j][9]
        clean          = input_array[j][10]
        file = 'all_data/output_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.txt'%(training_days, testing_days, center_or_peak, grouping, SMOTE_state, n_estimators, max_depth, clf_state, sorting_state, features, models, clean)

        # Read Data File
        data_table = table.Table.read(file, format = 'ascii')

        cuts = np.linspace(0, 1, 100)
        total_SLSNI = len(np.where(np.array(data_table['class']) == 'SLSN-I')[0])

        purity = np.array([])
        loss   = np.array([])

        if true_ratios:
            total_total   = len(data_table)
            total_SNI     = np.sum([i in ['SNI'    , 'SNIa', 'SNIbc' ] for i in data_table['class']])
            total_SNII    = np.sum([i in ['SNII'   , 'SNIIb', 'SNIIn'] for i in data_table['class']])
            total_SLSNI   = np.sum([i in ['SLSN-I'                   ] for i in data_table['class']])
            total_SLSNII  = np.sum([i in ['SLSN-II'                  ] for i in data_table['class']])
            total_Nuclear = np.sum([i in ['Nuclear'                  ] for i in data_table['class']])
            total_Star    = np.sum([i in ['Star'                     ] for i in data_table['class']])

            fraction_SNI     = total_SLSNI * 73.9 / 1.5 / total_SNI
            fraction_SNII    = total_SLSNI * 19.6 / 1.5 / total_SNII
            fraction_SLSNI   = total_SLSNI *  1.5 / 1.5 / total_SLSNI
            fraction_SLSNII  = total_SLSNI *  0.9 / 1.5 / total_SLSNII
            fraction_Nuclear = total_SLSNI *  0.6 / 1.5 / total_Nuclear
            fraction_Star    = total_SLSNI *  3.5 / 1.5 / total_Star
        else:
            fraction_SNI     = 1.0
            fraction_SNII    = 1.0
            fraction_SLSNI   = 1.0
            fraction_SLSNII  = 1.0
            fraction_Nuclear = 1.0
            fraction_Star    = 1.0

        for i in range(len(cuts)):
            match = np.array(data_table['SLSN-I']) >= cuts[i]

            if np.sum(match) > 0:
                # Normalize to True numbers
                found_SNI         = np.sum([i in ['SNI'    , 'SNIa' , 'SNIbc'] for i in data_table['class'][match]]) * fraction_SNI
                found_SNII        = np.sum([i in ['SNII'   , 'SNIIb', 'SNIIn'] for i in data_table['class'][match]]) * fraction_SNII
                found_SLSNI       = np.sum([i in ['SLSN-I'                   ] for i in data_table['class'][match]]) * fraction_SLSNI
                found_SLSNII      = np.sum([i in ['SLSN-II'                  ] for i in data_table['class'][match]]) * fraction_SLSNII
                found_Nuclear     = np.sum([i in ['Nuclear'                  ] for i in data_table['class'][match]]) * fraction_Nuclear
                found_Star        = np.sum([i in ['Star'                     ] for i in data_table['class'][match]]) * fraction_Star
                final_found_SLSNI = found_SLSNI / (found_SNI+found_SNII+found_SLSNI+found_SLSNII+found_Nuclear+found_Star) * 100
                lost_SLSN1        = found_SLSNI / total_SLSNI  * 100
            else:
                final_found_SLSNI = 0.0
                lost_SLSN1        = 0.0

            purity = np.append(purity, final_found_SLSNI)
            loss   = np.append(loss  , lost_SLSN1)

        all_purity = np.vstack([all_purity, purity])
        all_loss   = np.vstack([all_loss  , loss  ])

    purity_out = np.nanmean(all_purity, axis = 0) / 100
    purity_std = np.nanstd (all_purity, axis = 0) / 100

    loss_out = np.nanmean(all_loss, axis = 0) / 100
    loss_std = np.nanstd (all_loss, axis = 0) / 100

    return cuts, purity_out, purity_std, loss_out, loss_std

# Shared Parameters
grouping_array       = np.array([4])
SMOTE_state_array    = np.array([38,39,40,41,42])
n_estimators_array   = np.array([100])
clf_state_array      = np.array([38,39,40,41,42])
sorting_state_array  = np.array([38,39,40,41,42])
clean_array          = np.array([0])

# Individual Parameters
training_days_array  = np.array([20])
testing_days_array   = np.array([20])
max_depth_array      = np.array([7])
features_array       = np.array([13])
models_array         = np.array([0])
cuts_rapid, purity_out_rapid, purity_std_rapid, loss_out_rapid, loss_std_rapid = pure_complete(training_days_array, testing_days_array, grouping_array, SMOTE_state_array, n_estimators_array, max_depth_array, clf_state_array, sorting_state_array, features_array, models_array, clean_array, match_seed = True)

training_days_array  = np.array([70])
testing_days_array   = np.array([70])
max_depth_array      = np.array([9])
features_array       = np.array([7])
models_array         = np.array([1])
cuts_full, purity_out_full, purity_std_full, loss_out_full, loss_std_full = pure_complete(training_days_array, testing_days_array, grouping_array, SMOTE_state_array, n_estimators_array, max_depth_array, clf_state_array, sorting_state_array, features_array, models_array, clean_array, match_seed = True)

training_days_array  = np.array([20])
testing_days_array   = np.array([20])
max_depth_array      = np.array([7])
features_array       = np.array([16])
models_array         = np.array([0])
cuts_redshift, purity_out_redshift, purity_std_redshift, loss_out_redshift, loss_std_redshift = pure_complete(training_days_array, testing_days_array, grouping_array, SMOTE_state_array, n_estimators_array, max_depth_array, clf_state_array, sorting_state_array, features_array, models_array, clean_array, match_seed = True)

plt.plot        (cuts_rapid, purity_out_rapid                                                                   , color = colors[0], alpha = alpha    , zorder = 4)
plt.fill_between(cuts_rapid, purity_out_rapid-purity_std_rapid, purity_out_rapid+purity_std_rapid               , color = colors[0], alpha = alpha-0.2, zorder = 2, label = 'Rapid')

plt.plot        (cuts_full, purity_out_full                                                                     , color = colors[2], alpha = alpha    , zorder = 4)
plt.fill_between(cuts_full, purity_out_full-purity_std_full, purity_out_full+purity_std_full                    , color = colors[2], alpha = alpha-0.2, zorder = 2, label = 'Full Light Curve')

plt.plot        (cuts_redshift, purity_out_redshift                                                             , color = colors[4], alpha = alpha    , zorder = 4)
plt.fill_between(cuts_redshift, purity_out_redshift-purity_std_redshift, purity_out_redshift+purity_std_redshift, color = colors[4], alpha = alpha-0.2, zorder = 2, label = 'Redshift')

plt.axhline(y = 0.25, color = 'k', linestyle = '--', linewidth = 1, zorder = 0, alpha = 0.5)
plt.axhline(y = 0.50, color = 'k', linestyle = '--', linewidth = 1, zorder = 0, alpha = 0.5)
plt.axhline(y = 0.75, color = 'k', linestyle = '--', linewidth = 1, zorder = 0, alpha = 0.5)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.linspace(0, 1.0, 11), fontsize = fontsize)
plt.yticks(np.linspace(0, 1.0, 11), fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.xlabel('P(SLSN-I)', fontsize = fontsize)
plt.ylabel('Purity', fontsize = fontsize)
plt.legend(loc = 'upper left', title = 'Classifier:', fontsize = fontsize)
plt.savefig('all_purity.pdf', bbox_inches = 'tight')
plt.clf()



plt.plot        (cuts_rapid, loss_out_rapid                                                             , color = colors[0], alpha = alpha    , zorder = 4)
plt.fill_between(cuts_rapid, loss_out_rapid-loss_std_rapid, loss_out_rapid+loss_std_rapid               , color = colors[0], alpha = alpha-0.2, zorder = 2, label = 'Rapid')

plt.plot        (cuts_full, loss_out_full                                                               , color = colors[2], alpha = alpha    , zorder = 4)
plt.fill_between(cuts_full, loss_out_full-loss_std_full, loss_out_full+loss_std_full                    , color = colors[2], alpha = alpha-0.2, zorder = 2, label = 'Full Light Curve')

plt.plot        (cuts_redshift, loss_out_redshift                                                       , color = colors[4], alpha = alpha    , zorder = 4)
plt.fill_between(cuts_redshift, loss_out_redshift-loss_std_redshift, loss_out_redshift+loss_std_redshift, color = colors[4], alpha = alpha-0.2, zorder = 2, label = 'Redshift')

plt.axhline(y = 0.25, color = 'k', linestyle = '--', linewidth = 1, zorder = 0, alpha = 0.5)
plt.axhline(y = 0.50, color = 'k', linestyle = '--', linewidth = 1, zorder = 0, alpha = 0.5)
plt.axhline(y = 0.75, color = 'k', linestyle = '--', linewidth = 1, zorder = 0, alpha = 0.5)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.linspace(0, 1.0, 11), fontsize = fontsize)
plt.yticks(np.linspace(0, 1.0, 11), fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.xlabel('P(SLSN-I)', fontsize = fontsize)
plt.ylabel('Completeness', fontsize = fontsize)
plt.legend(loc = 'upper right', title = 'Classifier:', fontsize = fontsize)
plt.savefig('all_completeness.pdf', bbox_inches = 'tight')
plt.clf()