PorosityFE documentation
========================

**PorosityFE** is a finite-element / micromechanics toolkit for analysing
composite laminates with distributed and discrete porosity. It models the
through-thickness porosity profile, degrades the lamina stiffness via
Mori-Tanaka homogenization, and evaluates Tsai-Wu failure on a structured
8-node hex mesh.

This site is the **API reference** for the public Python module
``porosity_fe_analysis``. The same code also ships a Streamlit web app
(``app.py``) and a CLI (``validate_porosity_cli``), which are documented
in the project README on GitHub.

Installation
------------

PorosityFE targets Python 3.9 or newer. The recommended install (from a
fresh clone of the repository) is an editable install with the ``docs``
extra so this site can be rebuilt locally:

.. code-block:: bash

   pip install -e ".[docs]"

For the analysis-only runtime (no docs / web extras):

.. code-block:: bash

   pip install -e .

Quickstart
----------

A 2 %-porosity T800/epoxy laminate, compressed in displacement control:

.. code-block:: python

   from porosity_fe_analysis import (
       MATERIALS, PorosityField, CompositeMesh, FESolver,
   )

   mat = MATERIALS["T800_epoxy"]
   field = PorosityField(mat, void_volume_fraction=0.02,
                         distribution="uniform",
                         void_shape="spherical")
   mesh = CompositeMesh(field, mat, nx=20, ny=8, nz=12)
   solver = FESolver(mesh, mat, field)
   result = solver.solve(loading="compression", applied_strain=-0.01)
   print(f"max Tsai-Wu index = {result.max_failure_index:.3f}")
   print(f"stiffness knockdown = {result.knockdown:.3f}")

For the empirical (closed-form) knockdown models, use
:class:`~porosity_fe_analysis.EmpiricalSolver`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   examples
