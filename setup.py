from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.rst"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.3'
DESCRIPTION = 'Retrive region-specific demand data'
LONG_DESCRIPTION = ('This is a work in progress. For now, the package offers' +
                    'time series of estimated electricity demand on a country level. ' +
                    'Right now it is basically an API to retrieve the 2020 data provided as the' +
                    'supplementary of this paper:' +
                    'Long term load projection in high resolution for all countries globally by '+
                    'Toktarova et al.')

# Setting up
setup(
    name="pydemand",
    version=VERSION,
    author="Lukas Franken",
    author_email="<lukas.franken@ed.ac.uk>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['wget',
                      'numpy',
                      'pandas',
                      'matplotlib',
                      'pycountry'],
    keywords=['python', 'demand', 'energy', 'systems', 'electricity', 
              'series', 'prediction', 'estimation', 'heat'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)