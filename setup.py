from setuptools import setup
from setuptools.command.install import install


class InstallAndFetchDustMaps(install):
    def run(self):
        super().run()
        from dustmaps import config, sfd
        config.config.reset()
        sfd.fetch()


setup(name='fleet-pipe',
      version='1.0.0',
      description='Finding Luminous and Exotic Extragalactic Transients',
      url='https://github.com/gmzsebastian/FLEET',
      author=['Sebastian Gomez'],
      author_email="sgomez@cfa.harvard.edu",
      license='GNU GPL 3.0',
      packages=['FLEET'],
      package_data={'FLEET': ['training_set/*.txt', 'classification_catalog.dat']},
      include_package_data=True,
      cmdclass={'install': InstallAndFetchDustMaps},
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
        'ephem',
        'extinction',
        'pandas',
        'python-dateutil',
        'astropy',
        'pathlib',
        'requests',
        'scipy',
        'emcee',
        'casjobs @ git+git://github.com/dfm/casjobs@master',
        'mastcasjobs @ git+git://github.com/rlwastro/mastcasjobs@master',
      ],
      test_suite='nose.collector',
      zip_safe = False)
