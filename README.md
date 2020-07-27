# FLEET
Finding Luminous and Exotic Extragalactic Transients

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
P_SLSN = predict_SLSN('ZTF19acaiylt', '08:06:54.3654', '+39:00:23.8252')
```

If you don't have a name AND coordinates, you can specify either of them and the function will search in ZTF and TNS for the name of a corresponding set of coordiantes, or vice-versa. Just keep in mind that it will be a little slower.

The `predict_SLSN` function will generate several folders:
* catalogs/ has the catalog of nearby sources for each transient
* lightcurves/ has the appended lists of light curve information for each transient
* osc/ has the photometry from the Open Supernova Catalog
* ztf/ has the photometry from ZTF
* ignore/ is where the individual ignore files are stored
* photometry/ is where you store your own photometry

You can use the `querry_multiple_osc` function to automatically download OSC data in bulk instead of individually quering each object.





