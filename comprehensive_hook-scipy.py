"""
PyInstaller hook for scipy.
This hook collects all scipy submodules and data files, avoiding problematic version checks.
"""
import glob
import os
import sysconfig
from PyInstaller.compat import is_win, is_linux
from PyInstaller.utils.hooks import (
    get_module_file_attribute,
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

# Collect all scipy submodules
hiddenimports = collect_submodules('scipy')

# Collect all data files
datas = collect_data_files('scipy')

# Collect binaries
binaries = []

# Package the DLL bundle that official scipy wheels for Windows ship
# The DLL bundle will either be in extra-dll on Windows proper
# and in .libs if installed on a virtualenv created from MinGW (Git-Bash for example)
if is_win:
    # Add DLLs from extra-dll and .libs folders
    extra_dll_locations = ['extra-dll', '.libs']
    for location in extra_dll_locations:
        dll_glob = os.path.join(os.path.dirname(get_module_file_attribute('scipy')), location, "*.dll")
        dll_files = glob.glob(dll_glob)
        if dll_files:
            for file in dll_files:
                binaries.append((file, "."))

    # Try to collect any additional dynamic libraries
    scipy_libs = collect_dynamic_libs('scipy')
    binaries.extend(scipy_libs)

# If scipy is provided by Debian's python3-scipy, its scipy.__config__ submodule is renamed to a dynamically imported
# scipy.__config__${SOABI}__
if is_linux and "dist-packages" in get_module_file_attribute("scipy"):
    hiddenimports.append('scipy.__config__' + sysconfig.get_config_var('SOABI') + '__')

# Make sure to include numpy.f2py and its submodules
# (to avoid problems with numpy 2.0+ excluding f2py)
try:
    import numpy

    if hasattr(numpy, '__version__'):
        numpy_version = numpy.__version__
        # Simply collect all f2py submodules regardless of version
        # to avoid version checking issues
        f2py_modules = collect_submodules('numpy.f2py', filter=lambda name: name != 'numpy.f2py.tests')
        hiddenimports.extend(f2py_modules)
except Exception:
    # If there's any issue, just skip this part
    pass