#!/usr/bin/env python3
"""CLI entry point for the porosity validation suite.

Runs model predictions against all bundled experimental datasets and
generates an aggregated MAE report (PNG plot + Markdown table).

Usage:
    validate_porosity                    # uses bundled datasets, writes to cwd
    validate_porosity --output-dir PATH  # write reports to specified dir
    validate_porosity --datasets PATH    # use datasets from specified dir
    validate_porosity --quiet            # suppress per-dataset output
"""

import argparse
import os
import sys


def _resolve_version() -> str:
    """Return the package version from importlib.metadata when installed.

    Falls back to a hard-coded string when running from a source checkout
    that hasn't been pip-installed (e.g. during tests). The hard-coded
    fallback must be kept in sync with pyproject.toml on each release.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("porosity-fe")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    return "1.2.0"


def _resolve_bundled_datasets_dir() -> str:
    """Return path to the bundled datasets directory.

    When frozen by PyInstaller, the datasets are under sys._MEIPASS.
    When run from source, they're at validation/datasets/ next to this file.
    """
    if getattr(sys, 'frozen', False):
        # Running inside a PyInstaller bundle
        base = sys._MEIPASS
        return os.path.join(base, 'validation', 'datasets')
    # Running from source
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'validation', 'datasets')


def _resolve_bundled_schema_dir() -> str:
    """Return path to the bundled schemas directory."""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
        return os.path.join(base, 'validation', 'schemas')
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'validation', 'schemas')


def _ensure_validation_imports():
    """Make validation/ importable regardless of whether we're frozen."""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
        sys.path.insert(0, base)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='validate_porosity',
        description='Run porosity validation suite against experimental datasets.',
    )
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Directory containing dataset JSON files '
             '(default: bundled datasets)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=os.getcwd(),
        help='Directory to write report files (default: current directory)',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress per-dataset progress output',
    )
    parser.add_argument(
        '--version', action='version',
        version=f'validate_porosity {_resolve_version()}',
    )
    args = parser.parse_args(argv)

    _ensure_validation_imports()

    # Import after path setup
    try:
        from validation.validate_all import (
            run_all_datasets,
            generate_master_report,
            summarize_mae,
        )
    except ImportError as e:
        print(f"ERROR: Cannot import validation module: {e}", file=sys.stderr)
        return 2

    # Resolve dataset directory
    datasets_dir = args.datasets or _resolve_bundled_datasets_dir()
    if not os.path.isdir(datasets_dir):
        print(f"ERROR: Datasets directory not found: {datasets_dir}",
              file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Running porosity validation suite")
        print(f"  Datasets: {datasets_dir}")
        print(f"  Output:   {args.output_dir}")
        print()

    # Run predictions
    results = run_all_datasets(datasets_dir=datasets_dir)

    if not args.quiet:
        print(f"Loaded {len(results)} datasets. Generating report...")

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate report
    plot_path, md_path = generate_master_report(results,
                                                  output_dir=args.output_dir)

    # Print summary
    n_datasets_ok = 0
    n_datasets_err = 0
    for ds_name, ds_results in results.items():
        if 'error' in ds_results:
            n_datasets_err += 1
            if not args.quiet:
                print(f"  [ERROR] {ds_name}: {ds_results['error']}")
            continue
        n_datasets_ok += 1

    summary = summarize_mae(results)

    print()
    print(f"Report: {plot_path}")
    print(f"Detail: {md_path}")
    print()
    print(f"Datasets processed:  {n_datasets_ok} succeeded, "
          f"{n_datasets_err} failed")
    if summary['n_entries']:
        print(
            f"Overall MAE (property-weighted): "
            f"{summary['property_weighted_mae']:.2f}%  "
            f"(n={summary['n_entries']} paper-property entries)"
        )
        print(
            f"Overall MAE (point-weighted):    "
            f"{summary['point_weighted_mae']:.2f}%  "
            f"(n={summary['n_points']} individual data points)"
        )
        print(f"Best MAE:            {summary['best_mae']:.2f}%")
        print(f"Worst MAE:           {summary['worst_mae']:.2f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
