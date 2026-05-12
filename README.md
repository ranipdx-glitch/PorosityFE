# PorosityFE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/elhajjar1/PorosityFE/actions/workflows/tests.yml/badge.svg)](https://github.com/elhajjar1/PorosityFE/actions/workflows/tests.yml)
[![Build Executables](https://github.com/elhajjar1/PorosityFE/actions/workflows/build-executables.yml/badge.svg)](https://github.com/elhajjar1/PorosityFE/actions/workflows/build-executables.yml)
[![Release](https://img.shields.io/github/v/release/elhajjar1/PorosityFE?include_prereleases)](https://github.com/elhajjar1/PorosityFE/releases)

A desktop application and Python library for predicting how porosity defects degrade the strength and stiffness of fiber-reinforced composite laminates.

## Why This Tool?

Manufacturing defects like porosity are inevitable in composite structures. Engineers need to quantify how much strength is lost for a given porosity level, void morphology, and loading mode. PorosityFE provides:

- **Fast empirical screening** using calibrated Judd-Wright and power-law models
- **3D finite element analysis** with Eshelby-based stiffness degradation at each element
- **Layup-aware predictions** that account for how ply orientation affects porosity sensitivity
- **Interactive GUI** for rapid parametric studies without writing code

![Empirical Knockdown Curves](screenshots/knockdown_curves.png)

## Features

- **Two porosity types**: Distributed microporosity (continuous field) + discrete macrovoids (explicit geometry)
- **Three distribution models**: Uniform, clustered (midplane/surface/quarter), interface-concentrated
- **Three void morphologies**: Spherical, cylindrical (prolate), penny-shaped (oblate)
- **Four loading modes**: Compression, tension, shear, ILSS
- **Three empirical knockdown models**: Judd-Wright (exponential), power-law, linear
- **Two solver tiers**: Empirical correlations (fast) + 3D finite element (detailed)
- **Six material presets**: T800/epoxy, T700/epoxy, E-glass/epoxy, IM7/8551, T300/934, CF/PEEK (or define your own)
- **Discrete void modeling**: Explicit ellipsoidal voids with stress concentration factors
- **Tsai-Wu failure criterion**: Full 3D multiaxial strength evaluation

## Visualizations

### Porosity Distribution Models
![Porosity Distributions](screenshots/porosity_distributions.png)

### Void Stress Concentration Field
![Void SCF](screenshots/void_scf.png)

## Installation

### From source (recommended)
```bash
git clone https://github.com/elhajjar1/PorosityFE.git
cd PorosityFE
pip install -e ".[all]"
```

### Dependencies only
```bash
pip install numpy scipy matplotlib PyQt6
```

### Run tests
```bash
pytest tests/ -v
```

## Usage

### Desktop GUI
```bash
python porosity_gui.py
```

### Command-line analysis
```bash
python porosity_fe_analysis.py
```
Runs the full analysis across 5 porosity levels (1%-8%) and 5 configurations, generating PNG plots and JSON results.

### Python library
```python
from porosity_fe_analysis import *

# Quick empirical screening
material = MATERIALS['T800_epoxy']
pf = PorosityField(material, 0.05, distribution='interface', void_shape='penny')
mesh = CompositeMesh(pf, material, nx=50, ny=20, nz=24)

solver = EmpiricalSolver(mesh, material)
result = solver.get_failure_load(mode='compression', model='judd_wright')
print(f"Knockdown: {result['knockdown']:.3f}")

# Compare all configurations at 3% porosity
results = compare_configurations(0.03, material_name='T800_epoxy')
```

### Defining a custom material

Six material presets are available in the `MATERIALS` dict (see the
[Features](#features) list). To analyze a system that isn't in the built-in
set, construct a `MaterialProperties` instance directly and pass it through
the analysis pipeline — there is no separate registration step required.

```python
from porosity_fe_analysis import MaterialProperties, PorosityField, CompositeMesh, EmpiricalSolver

# Define a custom T700 / toughened-epoxy system
my_material = MaterialProperties(
    # Stiffness (MPa)
    E11=140000.0, E22=10500.0, E33=10500.0,
    G12=4900.0,   G13=4900.0,  G23=3700.0,
    nu12=0.30,    nu13=0.30,   nu23=0.42,
    # Strength (MPa)
    sigma_1c=1300.0, sigma_1t=2500.0,
    sigma_2t=70.0,   sigma_2c=210.0,
    tau_12=90.0,     tau_ilss=85.0,
    # Geometry
    t_ply=0.180,  n_plies=24,
    # Constituents (used by the FE solver's micromechanics path)
    matrix_modulus=3400.0,  matrix_poisson=0.36,
    fiber_modulus=240000.0, fiber_volume_fraction=0.60,
)

pf = PorosityField(my_material, void_volume_fraction=0.03, distribution='uniform')
mesh = CompositeMesh(pf, my_material, nx=30, ny=10, nz=12)
solver = EmpiricalSolver(mesh, my_material)
print(solver.get_failure_load(mode='compression', model='judd_wright'))
```

Notes:
- All stiffnesses and strengths are **MPa**; thicknesses are **mm**.
- All `MaterialProperties` fields are required — there are no defaults for
  the engineering constants. Cross-check anisotropy bounds (`ν12 < 0.5`,
  `E11 > E22`, etc.) before running.
- The FE micromechanics path uses `matrix_modulus`, `matrix_poisson`,
  `fiber_modulus`, and `fiber_volume_fraction` to compute the Eshelby
  stiffness degradation. Supply realistic constituent values even if you
  only plan to run the empirical solver — both paths share the same
  `MaterialProperties` object.
- To use this material with the included CLI / batch scripts, you can also
  add it to the `MATERIALS` dict in `porosity_fe_analysis.py` so it can be
  selected by name (`compare_configurations(..., material_name='my_system')`).
- The empirical knockdown coefficients (`alpha`, `n`, `beta`) live on
  `EmpiricalSolver`, not on `MaterialProperties`. To recalibrate them for
  a non-standard material system, see
  [Calibrating `alpha` / `n` for a custom material](#calibrating-alpha--n-for-a-custom-material).

### Build macOS GUI app
```bash
pip install pyinstaller
python -m PyInstaller PorosityFE.spec --noconfirm --clean
# App at dist/PorosityFE.app
```

### Build validate_porosity CLI executable (Linux / macOS / Windows)
```bash
pip install pyinstaller
python -m PyInstaller ValidatePorosity.spec --noconfirm --clean
# Linux/macOS: dist/validate_porosity/validate_porosity
# Windows:     dist\validate_porosity\validate_porosity.exe
```

Pre-built executables for all three platforms are produced automatically
by GitHub Actions on every push; download them from the Actions tab
(artifact names: `validate_porosity-linux`, `-macos`, `-windows`) or
from the Releases page for tagged versions.

CLI usage:
```bash
validate_porosity --help             # show all options
validate_porosity                    # run against bundled datasets, write to cwd
validate_porosity --output-dir /tmp  # write reports elsewhere
validate_porosity --quiet            # suppress progress output
```

## Output Files

| File Pattern | Description |
|---|---|
| `porosity_profile_*.png` | Through-thickness porosity profiles |
| `porosity_mesh_3d_*.png` | 3D mesh with hexahedral elements |
| `porosity_mesh_detail_*.png` | Cross-section element detail |
| `porosity_damage_*.png` | Stiffness reduction contour maps |
| `porosity_comparison_*.png` | Model comparison bar charts |
| `porosity_knockdown_curves.png` | Knockdown vs porosity curves |
| `porosity_analysis_results_*.json` | Numerical results (JSON) |

The GUI's **File → Export Results** menu writes the active run's empirical knockdown table to either JSON or CSV — the format is picked from the file extension you type in the save dialog (`.json` or `.csv`). CSV files include the analysis configuration as `#`-prefixed comment lines at the top (which pandas, Excel, and MATLAB all ignore by default), followed by a flat `mode,model,failure_stress_MPa,knockdown` table.

## Inputs and Conventions

### Void volume fraction `Vp`

`Vp` is **always** a dimensionless **fraction in `[0, 1]`** — never a percent.

| You want to model | Pass | Do **not** pass |
|---|---|---|
| 1% porosity  | `Vp = 0.01` | `Vp = 1.0`  |
| 3% porosity  | `Vp = 0.03` | `Vp = 3.0`  |
| 10% porosity | `Vp = 0.10` | `Vp = 10.0` |

The plotting axes display `Vp * 100 (%)` for readability; that is a display
convention only. The constructor (`PorosityField(..., void_volume_fraction=Vp)`)
rejects values outside `[0, 1]` with a `ValueError` and offers a percent-vs-fraction
hint when the value is plausibly a percent (`Vp ≥ 1.001`).

### Per-ply vs. specimen-average porosity

The empirical knockdown path (`EmpiricalSolver`) treats `Vp` as the
**specimen-average** void volume fraction. Strength is degraded once at the
laminate level via `σ(Vp) = KD(Vp) · σ₀`; modulus reduction is **not** applied
per-ply by the empirical solver.

The FE path (`FESolver`) builds a 3D hexahedral mesh and applies stiffness
degradation **per element**, with the local porosity at each element coming
from `PorosityField.local_porosity(x, y, z)`. The through-thickness profile
depends on the `distribution` argument:

| `distribution`  | Meaning                                                                |
|-----------------|------------------------------------------------------------------------|
| `'uniform'`     | Same `Vp` in every element (and every ply).                            |
| `'clustered'`   | Gaussian profile centered at `cluster_location` (midplane / surface / quarter), normalized so the through-thickness mean equals `Vp`. |
| `'interface'`   | Gaussian peaks at every ply-to-ply interface, normalized so the through-thickness mean equals `Vp`. |

In all three cases the input `Vp` is the **target specimen average**; the
profile is normalized so that averaging it over the full thickness recovers
`Vp` (subject to discrete-void contributions, which are added afterwards).

Layup orientation enters the empirical path through the matrix-dominated
fraction `f_md` (see [Layup scaling](#layup-scaling) below) — the model
penalizes matrix-dominated layups more than fiber-dominated ones for the
same `Vp`.

## Physics Models

### Empirical Strength Knockdown

Both empirical models below take the **specimen-average** void volume fraction `Vp` as a dimensionless **fraction in `[0, 1]`** (e.g. 3% porosity → `Vp = 0.03`, never `Vp = 3.0`). Each returns a knockdown factor `KD ∈ (0, 1]` that multiplies the pristine strength `σ₀`.

**Judd-Wright** (exponential decay):
```
KD = exp(-alpha * Vp)
```
- `alpha` is an empirical sensitivity coefficient. It is **dimensionless** when `Vp` is a fraction.
- For small `Vp`, `KD ≈ 1 − alpha · Vp`, so `alpha` is approximately the fractional strength loss per unit `Vp`. Judd & Wright (1978) reported ILSS dropping ~7% per 1% voids in CFRP, which corresponds to `alpha ≈ 7`.
- The exponential form is an engineering convention re-fit of Judd & Wright's linear data; it is well-behaved up to `Vp ≈ 0.04–0.05` and over-penalizes higher porosities.

**Power Law**:
```
KD = (1 - Vp)^n
```
- `n` is a phenomenological exponent rooted in Mackenzie (1950) spherical-void elasticity and generalized empirically (Rice, 2005). `n = 1` corresponds to a simple area-reduction rule of mixtures; `n > 1` captures stress concentration around voids.

**Linear**:
```
KD = max(1 - beta * Vp, 0)
```
- `beta` is the same kind of sensitivity coefficient as `alpha` in Judd-Wright (dimensionless when `Vp` is a fraction; `beta ≈ -ΔKD / ΔVp`). The linear form matches the original Judd & Wright (1978) "ILSS drops ~7% per 1% voids" reading directly, and is included for screening and easy hand-checks. It saturates to zero at `Vp = 1/beta` and is best used at low `Vp` (typically `< 0.05`).

#### QI-calibrated coefficients (Elhajjar 2025)

`f_md_ref = 0.5` is the layup-scaling reference (`scale = 1.0` at `f_md = 0.5`); the QI coefficients below were tuned with this scaling already applied, so they are NOT raw fits to a single layup but represent the model's effective-baseline values:

| Loading mode | `alpha` (Judd-Wright) | `n` (Power-Law) | `beta` (Linear) |
|---|---|---|---|
| Compression       | 6.9  | 2.8 | 5.5 |
| Tension           | 3.9  | 1.8 | 3.5 |
| Shear (in-plane)  | 8.0  | 3.5 | 7.0 |
| ILSS              | 10.0 | 4.5 | 9.0 |

These values sit inside published CFRP ranges: `alpha ≈ 1–3` for fiber-dominated tension and `5–10` for matrix-dominated ILSS / flexure; `n ≈ 1–2` for stiffness-like properties and `3–5` for compression / ILSS.

#### Layup scaling

PorosityFE adapts the QI coefficients to the user's layup via a matrix-dominated fraction `f_md` computed from the ply angles (0° → 0.0, ±45° → 0.5, 90° → 1.0):

```
alpha_eff(mode) = alpha_QI(mode) * (f_md / 0.5)
n_eff(mode)     = max(n_QI(mode) * (f_md / 0.5), 0.1)
```

A floor of `0.15` is applied to the scale (`0.80` for ILSS, which is always matrix-dominated):

| Layup | `f_md` | scale | `alpha_eff` (compression) |
|---|---|---|---|
| `[0]_16` (UD)              | 0.00 | 0.15 (floor) | 1.035 |
| `[±45]_4s` (off-axis)      | 0.50 | 1.00          | 6.90  |
| `[90]_8` (transverse)      | 1.00 | 2.00          | 13.80 |

Matrix-dominated layups are penalized more by porosity than fiber-dominated layups, as expected physically.

#### Validity bounds

The calibration data covers `Vp ≲ 0.05`. Beyond that, both forms should be treated as extrapolations. `Vp` alone does not capture void *morphology* (size, aspect ratio, clustering), which is a known source of scatter — use the FE solver path when spatial stress-concentration detail is needed.

#### Calibrating `alpha` / `n` for a custom material

If your material system differs significantly from the calibration set:

1. Manufacture a ladder of coupons spanning `Vp ≈ 0–5%` (vary autoclave debulk pressure or cure vacuum).
2. Measure void content per ASTM D2734 (matrix burnoff) or D3171 (acid digestion); cross-check by μCT or polished cross-section.
3. Run the relevant strength test: ASTM D2344 (ILSS / short-beam shear), D7264 (flexure), D3039 (tension), or D6641 (compression).
4. Normalize each datum by the void-free baseline: `KD = σ(Vp) / σ(0)`.
5. Regress `ln(KD)` vs `Vp` (slope `= −alpha`) for Judd-Wright, or `ln(KD)` vs `ln(1 − Vp)` (slope `= n`) for the power law.

Custom `alpha` / `n` / `beta` values are exposed as keyword-only constructor arguments on `EmpiricalSolver`. Overrides are partial (modes you don't pass keep the calibrated defaults) and are layup-scaled exactly like the built-in coefficients:

```python
# Fitted ILSS alpha for a custom material; other modes keep QI defaults.
solver = EmpiricalSolver(
    mesh, material,
    judd_wright_alpha={'ilss': 12.0},
    power_law_n={'ilss': 5.2},
    linear_beta={'ilss': 11.0},
)
```

At the QI reference layup (`f_md = 0.5`, `scale = 1.0`) the override is used directly; for a different layup it scales the same way as the QI baseline (e.g. with a UD layup, `judd_wright_alpha={'ilss': 12.0}` becomes an effective `12.0 × 0.80 = 9.6` because ILSS uses a 0.80 floor — see [Layup scaling](#layup-scaling)). Override values must be positive finite numbers; mode keys must be a subset of `{'compression', 'tension', 'shear', 'ilss'}`.

### Finite Element Solver

The FE solver builds a 3D hexahedral mesh and degrades element stiffness based on local porosity:

1. **Eshelby inclusion theory** computes degraded matrix properties (voids as zero-stiffness ellipsoids)
2. **Micromechanics rules** map matrix degradation to composite degradation ratios for E11, E22, G12
3. **Tsai-Wu criterion** evaluates multiaxial failure at each integration point

### Failure Criterion

Full 3D Tsai-Wu with degraded strengths:
```
F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + F66*s6^2 + 2*F12*s1*s2 = 1
```

## Validation

PorosityFE is validated against **13 peer-reviewed experimental datasets**
covering carbon/epoxy, IM7/toughened epoxy, T300/epoxy systems, and
CF/PEEK thermoplastic. Validation is automated via `validate_porosity`
CLI (pre-built for Linux/macOS/Windows on the [Releases page](https://github.com/elhajjar1/PorosityFE/releases))
or in-process via `validation/validate_all.py`.

**Model scope (validated properties):**

| Property | # papers | Overall MAE |
|---|---|---|
| ILSS (short-beam shear) | 9 | 4.3% |
| Tensile strength | 7 | 6.9% |
| Tensile modulus | 3 | 1.3% |
| Transverse tensile modulus | 3 | 3.4% |
| Flexural modulus (D-matrix CLT) | 5 | 8.9% |
| Compression strength | 2 | 11.4% |
| Shear strength | 2 | 13.5% |
| Transverse tensile strength | 3 | 14.5% |
| Shear modulus (A-matrix CLT) | 1 | 15.4% |

Overall MAE:
- Property-weighted: **7.69%** across 35 (paper, property) pairs (each entry weighted equally — what `validate_porosity` reports as the headline).
- Point-weighted: **7.03%** across 239 individual (Vp, normalized) data points (each measurement weighted equally — the standard convention in regression-error reporting).

The two aggregations differ because datasets carry very different numbers of points; `validate_porosity` prints both in the run summary.

## Limitations

- Empirical models are calibrated for porosity levels up to ~10%
- FE solver uses analytical stiffness degradation (not full nonlinear FE)
- Thermal residual stresses are not included
- Fatigue and environmental effects are not modeled
- Delamination initiation/propagation is not explicitly simulated
- **Flexural strength** was removed from the validation database in v1.1.1
  because 3-point bend failure involves mixed compression + interlaminar
  shear mechanisms that the Judd-Wright mode proxy cannot capture reliably
  (observed MAE 8-40% across papers)

## Citation

If you use PorosityFE in your research, please cite:

```bibtex
@software{elhajjar2026porosityfe,
  author = {Elhajjar, Rani},
  title = {{PorosityFE}: Porosity-Degraded Composite Laminate Analysis},
  year = {2026},
  url = {https://github.com/elhajjar1/PorosityFE},
  version = {1.2.0}
}
```

Related publication:
> Elhajjar, R. (2025). Fat-tailed failure strength distributions and manufacturing defects in advanced composites. *Scientific Reports*, 15, 25977. [DOI: 10.1038/s41598-025-06693-4](https://doi.org/10.1038/s41598-025-06693-4)

## References

- Judd & Wright (1978) - Voids and their effects on the mechanical properties of composites — an appraisal. *SAMPE Journal* 14(1), 10–14
- Mackenzie (1950) - The elastic constants of a solid containing spherical holes. *Proc. Phys. Soc. B* 63(1), 2–11
- Rice (2005) - Use of normalized porosity in models for the porosity dependence of mechanical properties. *J. Mater. Sci.* 40, 983–989
- Eshelby (1957) - The determination of the elastic field of an ellipsoidal inclusion
- Mura (1987) - Micromechanics of Defects in Solids
- Tsai & Wu (1971) - A general theory of strength for anisotropic materials
- Elhajjar (2025) - Fat-tailed failure strength distributions and manufacturing defects

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, suggesting features, and submitting pull requests.

## License

MIT License. See [LICENSE](LICENSE) for details.
