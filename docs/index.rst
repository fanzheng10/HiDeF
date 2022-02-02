.. hidef documentation master file, created by
   sphinx-quickstart on Tue Oct 27 21:35:14 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HiDeF's documentation!
=================================

HiDeF [#f1]_ aims to reimagine hierarchical data clustering. The name HiDeF stands for “Hierarchical community Decoding Framework”. HiDeF integrates graph-based community detection and the idea of “persistent homology” in order to determine robust clustering patterns in complex data at multiple scales. Given the inputs in data points in graph or matrix formats, HiDeF returns a list of multiscale clusters with measurement of their robustness, as well as a directed acyclic graph (DAG) to represent the organization of these clusters.

Installation
------------

.. toctree::
   self

Local installation of python package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
From source: ::

  python setup.py install

Installation via ``pip`` ::

  pip install hidef

Cytoscape
^^^^^^^^^^
HiDeF is separately distributed via the `CDAPS <https://apps.cytoscape.org/apps/cycommunitydetection>`_ framework [#f2]_ in Cytoscape.

.. Note::
   We try to maintain timely synchronization of the HiDeF versions across the Python package and Cytoscape. However, it may be possible to have small difference in results across the platforms due to the Cytoscape version is behind the latest version of the Python package.

What's new
___________

Version 1.1.4:

- Updated ``setup.py`` and added ``requirements.txt`` to specify minimum versions of
  packages HiDef depends on.

- Added ``Makefile`` which includes shortcuts to build and deploy software

Version 1.1.3:

- Community detection with multiple resolutions now run in parallel with python multiprocessing module
- The default algorithm changed to Leiden as it is faster than Louvain
- Now support multiplex community detection

Tutorials
_________
.. toctree::
   tutorial/index

API
___
.. toctree::
   :maxdepth: 2

   api

References
----------
.. rubric:: Footnotes
.. [#f1] Zheng, F, Zhang, S, et al. HiDeF: identifying persistent structures in multiscale ‘omics data. *Genome Biology*, 22, 21 (2021).
.. [#f2] Singhal, A. Cao, S. Churas, C. et al. Multiscale community detection in Cytoscape. *PLoS Comput. Biol*. 16, e1008239 (2020).


