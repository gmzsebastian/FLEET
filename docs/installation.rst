.. _installation:

Installation
============

There are two ways to install ``fleet``, either using pip or from source.

Using pip
---------

To install the latest release version you can run::

    pip install fleet_pipe

From source
-----------

To install the latest development version directly from GitHub you can run::

    git clone https://github.com/gmzsebastian/fleet.git
    cd fleet
    pip install -e .

Requirements
------------

In order to use FLEET you will need to set up a few things.

Bash Profile
^^^^^^^^^^^^

FLEET requires a bash profile to be set up with the following environment variables to store reference data. You should setup a ``fleet_data`` key in your bash profile (e.g. ``~/.bash_profile``):

.. code-block:: bash

    export fleet_data="/path/to/store/fleet/data"

PanSTARRS API key
^^^^^^^^^^^^^^^^^

You will need to obtain a PanSTARRS API key to query host galaxy information. You can do this by registering at the `PanSTARRS website <https://mastweb.stsci.edu/ps1casjobs/home.aspx>`_. Once you have your key, place a file
named ``3PI_key.txt`` either in the ``fleet_data`` directly you just setup or directly in your home directory (e.g. ``/Users/username/3PI_key.txt``) with your API key and password in the following content:

.. code-block:: text

    123123123123 mypassword

Alternatively, if you set the key ``use_old = False`` when running FLEET, the key will not be necessary. This is not recommended, since it will use a different catalog than the one used to train the original data.

TNS API key
^^^^^^^^^^^

You will also need a Transient Name Server (TNS) API key to query the TNS database to obtain details for existing transients. You can obtain this key by registering at the `TNS website <https://wis-tns.weizmann.ac.il>`_.
Once you have your key, place a file named ``TNS_key.txt`` either in the ``fleet_data`` directory you just set up or directly in your home directory (e.g. ``/Users/username/TNS_key.txt``) with your API key in the following format:

.. code-block:: text

    123abc123abc123abc123abc123abc
    12345
    bot_name

The first line is your API key, the second line is your TNS password, and the third line is the name of the bot you want to use for queries.

Alternatively, you can set up the following environment variables in your bash profile:

.. code-block:: bash

    export TNS_API_KEY="123abc123abc123abc123abc123abc"
    export TNS_ID="12345"
    export TNS_USERNAME="bot_name"

Dust Maps
^^^^^^^^^

You will need to download the dust maps used by FLEET from MAST to do extinction corrections. The ``dustmaps`` package should be automatically installed, but in case it is not, you can install it using pip:

.. code-block:: bash

    pip install dustmaps
    pip install git+git://github.com/dfm/casjobs@master	
    pip install git+git://github.com/rlwastro/mastcasjobs@master	

After installing the package, you can download the dust maps by running the following command in Python:

.. code-block:: python

    from dustmaps.config import config
    config['data_dir'] = '/path/to/store/maps/in'

    import dustmaps.sfd
    dustmaps.sfd.fetch()

Pickle Files
^^^^^^^^^^^^

Lastly, in order to not have to re-train the model every time you run FLEET, you will need to generate the pickle files that store the trained model, which will be saved to the ``fleet_data`` directory. You can do this by running the following command in Python:

.. code-block:: python

    from fleet.classify import save_pickles
    save_pickles()


