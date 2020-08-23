from astropy import table
import numpy as np
import emcee

def linex(x, amplitude, amplitude2, offset, magnitude):
    '''
    Loss Function for fitting an asymmetrical parabola

    Parameters
    ---------------
    amplitude  : How wide the light curve is 
    amplitude2 : Assymetric parameter
    offset     : Offset in time
    magnitude  : Brightest magnitude

    '''
    slope = np.exp(amplitude * (x - offset)) - amplitude * amplitude2 * (x - offset)
    y = slope - 1 + magnitude
    return y

def lnlike_double(theta, x, y, z):
    # Likelihood function allowing W2 to very
    amplitude, amplitude2, offset, magnitude = theta

    # Fit red lightcurve
    weight = 1.0 / (z ** 2)
    error  = y - linex(x, amplitude, amplitude2, offset, magnitude)
    Likely = -0.5*(np.sum(weight * error ** 2 + np.log(2.0 * np.pi / weight)))

    return Likely

def lnlike_single(theta, x, y, z, W2 = 0.6):
    # Likelihood function fixing W1 to 0.6
    # the average from fitting the full light curves
    amplitude, offset, magnitude = theta

    # Fit red lightcurve
    weight = 1.0 / (z ** 2)
    error  = y - linex(x, amplitude, W2, offset, magnitude)
    Likely = -0.5*(np.sum(weight * error ** 2 + np.log(2.0 * np.pi / weight)))

    return Likely

def lnprior_double(theta):
    # Log prior funtion, define the priors here
    amplitude, amplitude2, offset, magnitude = theta

    if 0.01  < amplitude2 < 1   and \
       -10   <  amplitude < 0   and \
       -50   <     offset < 50  and \
       -30   <  magnitude < 30  :
        return 0
    return -np.inf

def lnprior_single(theta):
    # Log prior funtion, define the priors here
    amplitude, offset, magnitude = theta

    if -10   <  amplitude < 0   and \
       -50   <     offset < 50  and \
       -30   <  magnitude < 30  :
        return 0
    return -np.inf

def lnprob_double(theta, x, y, z):
    # Return likelihood function + Prior
    lp = lnprior_double(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_double(theta, x, y, z)

def lnprob_single(theta, x, y, z):
    # Return likelihood function + Prior
    lp = lnprior_single(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_single(theta, x, y, z)

def fit_linex(output_table, date_range = np.inf, n_walkers = 50, n_steps = 500, n_cores = 1, model = 'double', g_correct = 0, r_correct = 0):
    '''
    Use emcee to fit an exponential function to the green and red light curves
    of a transient.

    Parameters
    ---------------
    output_table  : Data with photometry
    date_range    : Maximum number of days from the first detection
    n_walkers     : Number of walkers for emcee
    n_steps       : Number of steps for emcee
    n_cores       : Number of cores for the emcee
    model         : 'double' will fit an exponential rise and linear decline
                    'single' will just fit a single exponential
    g_correct     : extinction value in g band
    r_correct     : extinction value in r band

    Output
    ---------------
    red_amplitude  , red_amplitude2  , red_offset  , red_magnitude   : Best fit parameters for r-band light curve
    green_amplitude, green_amplitude2, green_offset, green_magnitude : Best fit parameters for g-band light curve
    model_color     : g-r color from the model light curves at the time of brightest measured magnitude
    bright_mjd      : Brightest MJD in either g or r
    first_mjd       : First MJD in either g or r
    green_brightest : Brightest measured g-band magnitude
    red_brightest   : Brightest measured r-band magnitude
    '''

    flot = lambda x : np.array(x).astype(float)
    # Read in Data
    all_magnitudes  = flot(output_table['Mag'])
    all_filters     = np.array(output_table['Filter']).astype('str')
    all_upperlimits = np.array(output_table['UL']).astype('str')
    all_MJDs        = flot(output_table['MJD'])
    all_ignores     = np.array(output_table['Ignore']).astype('str')

    # Separate into green and red filters
    green = np.where(((all_filters == 'g') | (all_filters == "g'")) & (all_upperlimits == 'False') & (all_ignores == 'False'))
    red   = np.where(((all_filters == 'r') | (all_filters == 'R') | (all_filters == "r'")) & (all_upperlimits == 'False') & (all_ignores == 'False'))

    # Get Data into a useful format
    g_time   = np.array(output_table['MJD'   ][green])
    r_time   = np.array(output_table['MJD'   ][red  ])
    g_mag    = np.array(output_table['Mag'   ][green]) - g_correct
    r_mag    = np.array(output_table['Mag'   ][red  ]) - r_correct
    g_magerr = np.array(output_table['MagErr'][green])
    r_magerr = np.array(output_table['MagErr'][red  ])

    # Append all magnitudes
    all_mags  = np.append(g_mag, r_mag)
    all_times = np.append(g_time, r_time)

    # Make sure there's at least 2 data points in each band
    if (len(g_mag) > 1) & (len(r_mag) > 1):
        # Find brightest date and magnitude
        bright_mjd = all_times[np.nanargmin(all_mags)]
        first_mjd  = np.nanmin(all_MJDs[np.where((all_upperlimits == 'False') & (all_ignores == 'False'))])
        bright_mag = np.nanmin(all_mags)
        
        # Split into g and r
        g_phase  = np.array(output_table['MJD'][green] - bright_mjd)
        r_phase  = np.array(output_table['MJD'][red  ] - bright_mjd)

        # From first point
        g_first = np.array(output_table['MJD'][green] - first_mjd)
        r_first = np.array(output_table['MJD'][red  ] - first_mjd)

        # Overwrite Errorbars that don't exist
        g_magerr[np.isnan(g_magerr)] = 0.1
        r_magerr[np.isnan(r_magerr)] = 0.1
        g_magerr[g_magerr==0.0     ] = 0.1
        r_magerr[r_magerr==0.0     ] = 0.1

        # Select only a subset of data that's within 50 days of peak
        good_g = np.where(np.isfinite(g_phase) & (g_first < date_range) & np.isfinite(g_mag) & np.isfinite(g_magerr))[0]
        good_r = np.where(np.isfinite(r_phase) & (r_first < date_range) & np.isfinite(r_mag) & np.isfinite(r_magerr))[0]
        
        # Make sure there's at least 2 data points in each band
        if (len(g_mag[good_g]) > 1) & (len(r_mag[good_r]) > 1):

            # Brightest Bands
            green_brightest = np.nanmin(g_mag[good_g])
            red_brightest   = np.nanmin(r_mag[good_r])

            # Select the data to go into the MCMC
            x_r, y_r, z_r = r_phase[good_r], r_mag[good_r], r_magerr[good_r]
            x_g, y_g, z_g = g_phase[good_g], g_mag[good_g], g_magerr[good_g]

            def create_prior():
                Amp  = np.random.uniform(         -0.30,          -0.05, n_walkers)
                Amp2 = np.random.uniform(           0.9,            1.0, n_walkers)
                Off  = np.random.uniform(            -8,              8, n_walkers)
                Mag  = np.random.uniform(bright_mag-0.3, bright_mag+0.3, n_walkers)
                
                pos = np.array([Amp, Amp2, Off, Mag]).T
                return pos

            # Make prior with proper priors
            pos_in = create_prior()
            pos_out = pos_in[0:1]
            while len(pos_out) < n_walkers:
                pos = pos_in[[np.isfinite(lnprior_double(i)) for i in pos_in]]
                pos_out = np.append(pos_out, pos, axis = 0)

            # Crop to correct length
            if len(pos_out) != n_walkers:
                pos = pos_out[1:n_walkers+1]
            else:
                pos = pos_out
            # If not fitting for a decline, remove that parameter
            if model == 'single':
                pos = pos[:,[0,2,3]]

            # Number of parameters being fit
            n_dim = pos.shape[1]

            # Run the MCMC
            print( "Running MCMC ...")
            # Fit rest days since first data point
            if model == 'double':
                sampler_red   = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_double, args=(x_r, y_r, z_r), threads=n_cores)
                sampler_green = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_double, args=(x_g, y_g, z_g), threads=n_cores)
            elif model == 'single':
                sampler_red   = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_single, args=(x_r, y_r, z_r), threads=n_cores)
                sampler_green = emcee.EnsembleSampler(n_walkers, n_dim, lnprob_single, args=(x_g, y_g, z_g), threads=n_cores)

            # Run the MCMC
            sampler_red.run_mcmc(pos, n_steps)
            sampler_green.run_mcmc(pos, n_steps)

            # Only consider the last quarter of the chain for parameter estimation
            samples_r_crop = sampler_red.chain  [:, int(3*n_steps/4):, :].reshape((-1, n_dim))
            samples_g_crop = sampler_green.chain[:, int(3*n_steps/4):, :].reshape((-1, n_dim))

            # Obtain the parametrs of the best fit
            if model == 'single':
                amplitude_mcmc_r, offset_mcmc_r, magnitude_mcmc_r = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_r_crop, [15.87, 50, 84.13], axis=0)))
                amplitude_mcmc_g, offset_mcmc_g, magnitude_mcmc_g = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_g_crop, [15.87, 50, 84.13], axis=0)))
                amplitude2_mcmc_r = (0.6, 0.6, 0.6)
                amplitude2_mcmc_g = (0.6, 0.6, 0.6)
            elif model == 'double':
                amplitude_mcmc_r, amplitude2_mcmc_r, offset_mcmc_r, magnitude_mcmc_r = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_r_crop, [15.87, 50, 84.13], axis=0)))
                amplitude_mcmc_g, amplitude2_mcmc_g, offset_mcmc_g, magnitude_mcmc_g = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_g_crop, [15.87, 50, 84.13], axis=0)))

            # Get best fit parameters
            red_amplitude , green_amplitude  =  amplitude_mcmc_r[0],  amplitude_mcmc_g[0]
            red_amplitude2, green_amplitude2 = amplitude2_mcmc_r[0], amplitude2_mcmc_g[0]
            red_offset    , green_offset     =     offset_mcmc_r[0],     offset_mcmc_g[0]
            red_magnitude , green_magnitude  =  magnitude_mcmc_r[0],  magnitude_mcmc_g[0]

            # Get color at brightest point
            brightest_phase = np.append(r_phase[good_r], g_phase[good_g])[np.argmin(np.append(r_mag[good_r], g_mag[good_g]))]
            model_color     = linex(brightest_phase, green_amplitude, green_amplitude2, green_offset, green_magnitude)-linex(brightest_phase, red_amplitude, red_amplitude2, red_offset, red_magnitude)

            return red_amplitude, red_amplitude2, red_offset, red_magnitude, green_amplitude, green_amplitude2, green_offset, green_magnitude, model_color, bright_mjd, first_mjd, green_brightest, red_brightest

        else:
            print('Not enough g and r band points inside given range \n')
            return np.nan * np.ones(13)
    else:
        print('Not enough g and r band points \n')
        return np.nan * np.ones(13)
