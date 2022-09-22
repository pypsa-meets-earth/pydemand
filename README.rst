=====
Pydem
=====

Pydem is a energy demand estimation Python library.

Right now, it is in an infant state, only retrieving country-scale electricity
demand data estimated for 2020 in `Long term load projection in high resolution for all countries globally`_.

.. _Long term load projection in high resolution for all countries globally: https://www.sciencedirect.com/science/article/abs/pii/S0142061518336196/

!! While these methods are scalable, they do not guarantee accurate results !!
We refer to the linked paper for a discussion of what to consider before using the data.

=====
Installation
=====

The library can be install via ``pip`` from `pypi`

.. code:: shell

    pip install pydem


=====
What is next?
=====

This package is a work in progress. We plan to

1. Retrieve an electricity demand estimation for any passed geospatial polygon
   based on statistical models, trained on the existing data.
2. Add estimations of heat demand profiles to the package, which are also
   retrievable by polygon.