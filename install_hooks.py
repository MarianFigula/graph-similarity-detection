"""
This script installs improved PyInstaller hooks for scipy and related packages.
Run this script before running PyInstaller to ensure all hooks are properly installed.
"""
import os
import shutil
import sys
import importlib.util
import platform
import glob


def get_pyinstaller_hooks_dir():
    """Find the PyInstaller hooks directory in the current environment."""
    try:
        import PyInstaller
        hooks_dir = os.path.join(os.path.dirname(PyInstaller.__file__), 'hooks')
        return hooks_dir
    except ImportError:
        print("ERROR: PyInstaller not found in the current environment.")
        return None


def backup_file(file_path):
    """Create a backup of the original file if it exists."""
    if os.path.exists(file_path):
        backup_path = file_path + '.bak'
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
            print(f"Created backup: {backup_path}")
        return backup_path
    return None


def install_hook(hook_name, hook_content):
    """Install a hook file with the given name and content."""
    hooks_dir = get_pyinstaller_hooks_dir()
    if not hooks_dir:
        return False

    hook_path = os.path.join(hooks_dir, hook_name)

    # Backup the original file if it exists
    backup_file(hook_path)

    # Write the new hook file
    with open(hook_path, 'w') as f:
        f.write(hook_content)

    print(f"Successfully installed hook: {hook_path}")
    return True


def create_comprehensive_scipy_hook():
    """Create comprehensive hook for scipy."""
    hook_content = """
"""
    with open('comprehensive_hook-scipy.py', 'r') as f:
        hook_content = f.read()

    return install_hook('hook-scipy.py', hook_content)


def disable_problematic_scipy_hooks():
    """Disable other problematic scipy-related hooks."""
    hooks_dir = get_pyinstaller_hooks_dir()
    if not hooks_dir:
        return False

    # Find all scipy-related hooks
    scipy_hooks = glob.glob(os.path.join(hooks_dir, 'hook-scipy.*.py'))

    for hook_path in scipy_hooks:
        # Skip the main scipy hook
        if os.path.basename(hook_path) == 'hook-scipy.py':
            continue

        # Create a backup
        backup_file(hook_path)

        # Replace with a simpler version that just collects submodules
        hook_name = os.path.basename(hook_path)
        module_name = hook_name[5:-3]  # Remove 'hook-' and '.py'

        simple_hook_content = f"""
# Simplified hook for {module_name}
# This hook simply collects all submodules to avoid version checking issues
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('{module_name}')
"""
        with open(hook_path, 'w') as f:
            f.write(simple_hook_content)

        print(f"Simplified hook: {hook_path}")

    return True


def main():
    """Main function to install all hooks."""
    print("Installing improved PyInstaller hooks...")

    # Install the comprehensive scipy hook
    if not create_comprehensive_scipy_hook():
        print("Failed to install scipy hook.")
        return

    # Disable other problematic scipy hooks
    disable_problematic_scipy_hooks()

    print("\nAll hooks installed successfully!")
    print("You can now run PyInstaller to create your executable.")


if __name__ == "__main__":
    main()