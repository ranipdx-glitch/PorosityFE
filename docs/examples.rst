Examples
========

The PorosityFE source tree ships a small gallery of runnable scripts that
exercise the public API end-to-end. Each script is standalone, uses the
``Agg`` matplotlib backend so it works headlessly, prints a short summary
to stdout, and writes a single PNG to ``examples/output/``.

Run any example from the repository root:

.. code-block:: bash

   python examples/uniform_spherical.py

Or sweep them all:

.. code-block:: bash

   for f in examples/*.py; do python "$f" || echo "FAILED: $f"; done

Gallery
-------

* ``examples/uniform_spherical.py`` --
  Uniform porosity with spherical voids; empirical compression knockdown
  via the Judd-Wright model.
* ``examples/clustered_midplane.py`` --
  Gaussian-clustered porosity at the laminate midplane; ILSS knockdown.
* ``examples/interface_penny.py`` --
  Interface-concentrated porosity with penny-shaped voids -- the
  worst-case ILSS morphology.
* ``examples/discrete_voids.py`` --
  Explicit :class:`~porosity_fe_analysis.VoidGeometry` ellipsoids layered
  on top of a low-uniform background; visualises the stress-concentration
  field around one void.
* ``examples/compute_degraded_clt_moduli.py`` --
  Classical lamination theory moduli with porosity degradation applied
  per ply.

The complete gallery (with descriptions and PNG previews) lives in
`examples/README.md <https://github.com/ranipdx-glitch/PorosityFE/blob/master/examples/README.md>`_
on GitHub.
