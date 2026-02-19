from setuptools import setup
from setuptools.command.install import install


class InstallAndFetchDustMaps(install):
    def run(self):
        super().run()
        from dustmaps import config, sfd
        config.config.reset()
        sfd.fetch()


setup(name='fleet_pipe',
      version='3.0.1',
      description='Finding Luminous and Exotic Extragalactic Transients',
      url='https://github.com/gmzsebastian/fleet',
      author=['Sebastian Gomez'],
      author_email="sebastian.gomez@austin.utexas.edu",
      license='MIT',
      packages=['fleet'],
      license_files=["LICENSE"],
      include_package_data=True,
      package_data={'fleet': ['training_set/*.txt', 'classification_catalog.dat', 'GLADE_short.txt']},
      # cmdclass={'install': InstallAndFetchDustMaps},
      install_requires=[
        'numpy',
        'astroquery',
        'dustmaps',
        'bs4',
        'ephem',
        'datetime',
        'astral==1.10.1',
        'PyAstronomy',
        'scikit-learn',
        'imbalanced-learn',
        'Pillow',
        'matplotlib',
        'lmfit',
        'alerce',
        'ephem',
        'extinction',
        'pandas',
        'python-dateutil',
        'astropy',
        'pathlib',
        'requests',
        'scipy',
        'emcee',
        'casjobs',
        'mastcasjobs',
      ],
      test_suite='nose.collector',
      zip_safe=False)
