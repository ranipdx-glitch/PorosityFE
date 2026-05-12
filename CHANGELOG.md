# Changelog

All notable changes to PorosityFE will be documented in this file.

## [1.2.0] - 2026-05-11

### Fixed
- **FE: penny / oblate voids were silently routed to the prolate Eshelby
  formula** because `ar = max/min ≥ 1` made the oblate branch unreachable.
  `_mt_effective_stiffness` now detects the axis of symmetry, uses the
  correct g-function branch (prolate vs oblate), and permutes the Voigt
  tensor to align with the actual void axis. Penny-shaped voids
  (`VOID_SHAPES['penny']`) now produce physically-correct anisotropy
  with through-disk degradation exceeding in-plane degradation. (#32)
- **FE: stiffness assembly now rejects inverted elements** (non-positive
  `det(J)`) at quadrature time rather than letting a wrong-sign block
  silently corrupt the assembled global K. (#33)
- **FE: engineering strain in post-processing is now rotated with
  `strain_transformation_3d`** (was `stress_transformation_3d`), so
  `FieldResults.strain_local` shear components are no longer off by 2x.
  Stress / Tsai-Wu paths were already correct and are unchanged. (#38)
- **GUI: `porosity-fe` console script** now prints a friendly stderr
  message and exits non-zero when PyQt6 is missing, instead of leaking
  a Python traceback. Error text mentions the discoverable
  `pip install porosity-fe[gui]` form. (#46)

### Added
- **CSV export** alongside the existing JSON export in the GUI's
  *File → Export Results* menu. Format is picked from the typed
  extension (`.csv` / `.json`) or the selected filter. CSV layout:
  `#`-prefixed config metadata, then a flat
  `mode,model,failure_stress_MPa,knockdown` table. (#30)
- **`EmpiricalSolver` constructor overrides** for `judd_wright_alpha`,
  `power_law_n`, and `linear_beta` — partial-merge with QI defaults,
  layup-scaled the same way, replacing the documented subclass
  workaround. (#16)
- **Input validation** on `MaterialProperties`, `CompositeMesh`,
  `VoidGeometry`, and `Hex8Element` constructors (positive moduli /
  strengths, Poisson ratios in `(-1, 0.5)`, positive geometry, fraction
  range on `node_porosities`). (#13)
- **Validation reporting** now publishes both property-weighted and
  point-weighted MAE — the published 7.7% is the property-weighted form;
  point-weighted is ~7.0%. (#36) New `summarize_mae()` helper exposes
  both for downstream tools.
- **GUI threading hardening**: `closeEvent` waits for the worker to
  exit; `_stop_requested` uses `threading.Event` for cross-core memory
  fences; the Run button is disabled before `_build_config` runs;
  `_on_stop` warns when the worker doesn't terminate within 2s. (#17)
- **Numerical-stability guards** in Mori-Tanaka inversion (pinv
  fallback + finite check) and Tsai-Wu evaluation (strength-floor clamp,
  non-finite check, fp clip on `elem_Vp`). (#14)
- **Error-message polish** on bad material / void-shape / mode /
  cluster_location names; UTF-8 encoding on all file I/O; floating-point
  noise clip on `Vp ≈ 1.0`; string `Vp` rejected with `TypeError`. (#21, #22, #23)
- **PyInstaller spec parity**: `PorosityFE.spec` now mirrors
  `ValidatePorosity.spec`'s dataset bundling and adds commonly-missed
  hidden imports (`PyQt6.sip`, `PyQt6.QtSvg`,
  `scipy.sparse.csgraph._validation`, `scipy.special.cython_special`). (#24)
- **Documentation**: README "Inputs and Conventions" section clarifies
  Vp as a fraction in [0, 1] with do/don't table; "Defining a custom
  material" worked example; per-ply vs. specimen-average porosity behavior
  for empirical vs. FE paths. (#1, #2, #4)

### Changed
- **Version source of truth**: `validate_porosity --version` now reads
  from `importlib.metadata` with a hard-coded fallback. Version aligned
  across `pyproject.toml`, `CHANGELOG.md`, `CITATION.cff`, README BibTeX,
  and `PorosityFE.spec`. (#45)

## [1.1.1] - 2026-04-19

### Changed
- **Removed `flexural_strength` property** from the validation schema and all
  datasets. The property consistently showed high MAE (8-40%) because 3-point
  bend failure in cross-ply and UD laminates involves mixed compression +
  interlaminar shear mechanisms that the current Judd-Wright mode mapping
  (`compression` proxy) cannot capture. Removing it focuses the validation
  database on properties the model can predict well.
- Overall validation MAE: **9.76% → 7.69%** (35 property-dataset pairs,
  down from 41)
- Affected datasets (6): Almeida 1994, Ghiorse 1993, Liu 2006, Olivier 1995,
  Stamopoulos 2016, Tang 1987 — all retain their other properties

### Added
- CI badge for "Build Executables" workflow in README
- Explicit documentation of model scope and property coverage

## [1.1.0] - 2026-04-19

### Added
- Expanded validation database from 3 to 13 peer-reviewed experimental papers
- Unified JSON Schema (Draft-07) for validation datasets
- 3 new material presets: IM7/8551, T300/934, CF/PEEK
- 3 CLT helper functions: `compute_degraded_clt_moduli`,
  `compute_degraded_clt_flexural_modulus`, `_build_clt_abd`
- Master validation runner `validation/validate_all.py` with strength
  (Judd-Wright) and modulus (CLT) prediction, aggregated MAE report
- Cross-platform CLI executable `validate_porosity` (Linux/macOS/Windows)
- GitHub Actions workflow `build-executables.yml` that builds and releases
  the CLI on Ubuntu, macOS, and Windows runners
- 28 new tests bringing total to 186

### Classical validation datasets added
- Ghiorse (1993) SAMPE Quarterly — AS4/3501-6
- Almeida & Nogueira Neto (1994) Compos. Struct. — 0-10% void range
- Tang, Lee & Springer (1987) J. Comp. Mater. — T300/976
- Bowles & Frimpong (1992) J. Comp. Mater. — IM7/8551-7
- Jeong (1997) J. Comp. Mater. — AS4 fabric
- Olivier, Cottu & Ferret (1995) Composites — T300/914

### Recent validation datasets added
- Liu et al. (2018) J. Comp. Mater. — T300/924, 6 porosity levels
- Zhang et al. (2025) Polymers — CF/PEEK thermoplastic matrix
- Wen et al. (2023) J. Reinf. Plast. Compos. — T700/epoxy + temperature
- Wang et al. (2022) J. Comp. Mater. — CF/epoxy + micro-CT damage evolution

## [1.0.0] - 2026-04-03

### Added
- PyQt6 desktop GUI with interactive porosity analysis
- Empirical strength models: Judd-Wright (exponential) and Power Law correlations
- 3D finite element solver with Eshelby-based stiffness degradation
- 3 porosity distribution types: uniform, clustered (midplane/surface/quarter), interface-concentrated
- 3 void morphologies: spherical, cylindrical (prolate), penny-shaped (oblate)
- 4 loading modes: compression, tension, shear, ILSS
- 3 built-in material presets: T800/epoxy, E-glass/epoxy, T700/epoxy
- Discrete void modeling with stress concentration factors
- Tsai-Wu failure criterion for multiaxial states
- Visualization: porosity fields, 3D meshes, damage contours, knockdown curves
- JSON export of analysis results
- PyInstaller macOS app bundle
- Comprehensive test suite
