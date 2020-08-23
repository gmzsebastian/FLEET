
# FLEET
Finding Luminous and Exotic Extragalactic Transients.

A machine-learning pipeline designed to predict the probability of transients to be a superluminous supernova, associated with this paper : arxiv_link

# Setup
In order to use FLEET you will need to have two keys in your system. One to query 3PI, and one to query the TNS. FLEET will search for these files in your home directory:

```
/Users/username/3PI_key.txt
/Users/username/tns_key.txt
```
You can request one from https://mastweb.stsci.edu/ps1casjobs/home.aspx and https://wis-tns.weizmann.ac.il/, respectively.

FLEET needs dust maps to calculate the extinction to each target, for that you will need to install `dustmaps`, in the terminal install through pip:

```
pip install dustmaps
```

And then inside Python import the necessary dust maps and install them in whichever directory your prefer.

```
from dustmaps.config import config
config['data_dir'] = '/path/to/store/maps/in'

import dustmaps.sfd
dustmaps.sfd.fetch()
```

You will also need `mastcasjobs`, it should be installed automatically, but if it doesn't you need to install these two modules:
```	
pip install git+git://github.com/dfm/casjobs@master	
pip install git+git://github.com/rlwastro/mastcasjobs@master	
```	

# Example
Assuming you have your 3PI and TNS keys set up, simply run the `main_assess` function on the transient of your choice.

```
from FLEET.classify import predict_SLSN
P_SLSN = predict_SLSN('ZTF19acaiylt', '08:06:54.3654', '+39:00:23.8252')
```

If you don't have a name AND coordinates, you can specify either of them and the function will search in ZTF and TNS for the name of a corresponding set of coordiantes, or vice-versa. Just keep in mind that it will be a little slower.

The `predict_SLSN` function will generate several folders:
* `catalogs/` has the catalog of nearby sources for each transient
* `lightcurves/` has the appended lists of light curve information for each transient
* `osc/` has the photometry from the Open Supernova Catalog
* `ztf/` has the photometry from ZTF
* `ignore/` is where the individual ignore files are stored
* `photometry/` is where you can store your own photometry
* `plots/` has output light curve and field plots with the models shown

You recommend using `querry_multiple_osc` when running FLEET on several objects. This function will automatically download OSC data in bulk instead of individually querying each object. For example:
```
from FLEET.classify import predict_SLSN
from FLEET.transient import querry_multiple_osc

object_names = ['2020a','2020b','2020c','2020d','2020e']
querry_multiple_osc(','.object_names)

for object_name in object_names:
	predict_SLSN(object_name)
```

If you set `plot_lightcurve = True` this will generate a plot an image of the field, light curve, and corresponding best fit model.
<p align="center"><img src="2020kn.pdf" align="center" alt="2020kn" width="900"/></p>

# Modification
We encourage you to modify the `predict_SLSN` function depending on your goals. For example, if you only need part of it, or if you want to run multiple classifiers on a single object.

You can modify the `create_training_testing` function to use a different set of features, depth, seed number, training days, etc. For example something like this would allow you to get errorbars on the predicted probability:
```
p_out = np.array([])
for seed in range(10, 20):
	predicted_probability = create_training_testing(object_name, features_table, clf_state = seed)
	p_out = np..append(p_out, predicted_probability)
average_p = np.mean(p_out)
std_p = np.std(p_out)
```
  
