# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Porosity FE Analysis Mac App (PyQt6)
import glob
import os

block_cipher = None

_spec_dir = os.path.dirname(os.path.abspath(SPEC))

# Mirror ValidatePorosity.spec's data bundling so any future GUI feature
# that loads validation datasets / schemas at runtime works in the frozen
# bundle. (Today the GUI doesn't load these, but the asymmetry between the
# two specs is a footgun — see issue #24.)
_dataset_files = [
    (path, 'validation/datasets')
    for path in glob.glob(os.path.join(_spec_dir, 'validation', 'datasets',
                                        '*.json'))
]
_schema_files = [
    (os.path.join(_spec_dir, 'validation', 'schemas',
                  'validation_dataset_schema.json'),
     'validation/schemas'),
]
_init_files = []
_validation_init = os.path.join(_spec_dir, 'validation', '__init__.py')
if os.path.exists(_validation_init):
    _init_files.append((_validation_init, 'validation'))

_datas = _dataset_files + _schema_files + _init_files

a = Analysis(
    ['porosity_gui.py'],
    pathex=[_spec_dir],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        'numpy',
        'scipy',
        'scipy.interpolate',
        'scipy.linalg',
        'scipy.sparse',
        'scipy.sparse.linalg',
        'scipy.sparse.csgraph._validation',
        'scipy.special.cython_special',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_qtagg',
        'mpl_toolkits.mplot3d',
        'mpl_toolkits.mplot3d.art3d',
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtPrintSupport',
        'PyQt6.QtSvg',
        'PyQt6.sip',
        'json',
        'dataclasses',
        'porosity_fe_analysis',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
        'tkinter',
        '_tkinter',
        'pkg_resources',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PorosityFE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PorosityFE',
)

app = BUNDLE(
    coll,
    name='PorosityFE.app',
    icon=None,
    bundle_identifier='com.composites.porosity-fe',
    info_plist={
        'CFBundleName': 'Porosity FE Analysis',
        'CFBundleDisplayName': 'Porosity FE Analysis',
        'CFBundleShortVersionString': '1.2.0',
        'CFBundleVersion': '1.2.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
    },
)
