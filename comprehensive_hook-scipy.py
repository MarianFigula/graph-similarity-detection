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

hiddenimports = collect_submodules('scipy')

datas = collect_data_files('scipy')

binaries = []

if is_win:
    extra_dll_locations = ['extra-dll', '.libs']
    for location in extra_dll_locations:
        dll_glob = os.path.join(os.path.dirname(get_module_file_attribute('scipy')), location, "*.dll")
        dll_files = glob.glob(dll_glob)
        if dll_files:
            for file in dll_files:
                binaries.append((file, "."))

    scipy_libs = collect_dynamic_libs('scipy')
    binaries.extend(scipy_libs)

if is_linux and "dist-packages" in get_module_file_attribute("scipy"):
    hiddenimports.append('scipy.__config__' + sysconfig.get_config_var('SOABI') + '__')

try:
    import numpy

    if hasattr(numpy, '__version__'):
        numpy_version = numpy.__version__
        f2py_modules = collect_submodules('numpy.f2py', filter=lambda name: name != 'numpy.f2py.tests')
        hiddenimports.extend(f2py_modules)
except Exception:
    pass