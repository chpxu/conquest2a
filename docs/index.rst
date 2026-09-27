.. CONQUEST2a documentation master file, created by
   sphinx-quickstart on Fri Apr 10 23:04:44 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.



CONQUEST2a: post-processing utilities for CONQUEST
==================================================

CONQUEST2a (C2a) is a collection of modules providing various post-processing utilities for `CONQUEST <https://conquest.readthedocs.io/>`__. It aims to be simple and flexible, requiring minimal dependencies on CONQUEST-specific input/output files. C2a is currently able to process and plot (partial) density of states, bandstructures, charge densities, band densities; convert from CONQUEST coordinates to other file formats; calculate nearest neighbours, planar and dihedral angles; create, transform and write unit cells; extract stresses, forces, bandgaps from output.

This library operates independently of CONQUEST's ASE interface, which is documented here: https://conquest.readthedocs.io/en/latest/ase-conquest.html. This library does not manage CONQUEST and only serves to analyse output from runs or prepare coordinates for runs.

Note: this library explicitly depends on ASE for: units and reading cube files. ASE is listed as an explicit dependency but otherwise you should make sure it is available.

..    :maxdepth: 2
..    :caption: Contents:
..    :glob:


Reference
----------
* :doc:`src/types`
* :doc:`src/conquest`
* :doc:`src/pdos`
* :doc:`src/band`
* :doc:`src/supercell`
* :doc:`src/density`
* :doc:`src/io`
* :doc:`src/roadmap`

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Reference

   src/types
   src/conquest
   src/pdos
   src/band
   src/supercell
   src/density
   src/io
   src/roadmap


Contributing
------------

- Please take a look at the `GitHub page`_ for the source code.
- Suggest features, or report bugs on the `GitHub issues`_. Your feedback and help is greatly appreciated!

.. _GitHub issues: https://github.com/chpxu/conquest2a/issues
.. _GitHub page: https://github.com/chpxu/conquest2a

Licence
-------

CONQUEST2a is available freely under the MIT Licence.