"""
PyInstaller spec file for creating an executable with scipy support.
Replace 'your_script.py' with the actual name of your main Python script.
"""
import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

SCRIPT_NAME = 'main.py'

hiddenimports = collect_submodules('scipy')
scipy_datas = collect_data_files('scipy')

a = Analysis(
    [SCRIPT_NAME],
    pathex=[],
    binaries=[],
    datas=scipy_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create the EXE
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='graph-similarity-1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for GUI applications
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)