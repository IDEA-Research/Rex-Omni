"""
fastevaluate - Fast evaluation module for COCO/LVIS metrics
"""

import importlib.util

# Import the C++ extension module
# The C++ extension is compiled as a module named 'fastevaluate'
# When packages=['fastevaluate'] is declared in setup.py, the .so file
# is typically installed in the same directory as this __init__.py
import os

# Get the directory where this __init__.py is located
_package_dir = os.path.dirname(os.path.abspath(__file__))

# Try to find and load the compiled .so file
_fastevaluate = None

# Method 1: Look for .so file in the same directory as __init__.py
# This is where it's typically installed when packages=['fastevaluate'] is used
try:
    so_files = [
        f
        for f in os.listdir(_package_dir)
        if f.startswith("fastevaluate") and f.endswith(".so")
    ]
    if so_files:
        so_path = os.path.join(_package_dir, so_files[0])
        spec = importlib.util.spec_from_file_location("fastevaluate", so_path)
        _fastevaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_fastevaluate)
except (OSError, ImportError, AttributeError, FileNotFoundError):
    pass

# Method 2: Try loading from parent directory (for some installation setups)
if _fastevaluate is None:
    try:
        parent_dir = os.path.dirname(_package_dir)
        so_files = [
            f
            for f in os.listdir(parent_dir)
            if f.startswith("fastevaluate") and f.endswith(".so")
        ]
        if so_files:
            so_path = os.path.join(parent_dir, so_files[0])
            spec = importlib.util.spec_from_file_location("fastevaluate", so_path)
            _fastevaluate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_fastevaluate)
    except (OSError, ImportError, AttributeError, FileNotFoundError):
        pass

if _fastevaluate is None:
    raise ImportError(
        f"Could not import fastevaluate C++ extension. "
        f"Searched for .so files in: {_package_dir}\n"
        f"Please ensure the package is properly installed with 'pip install -e .' or 'pip install .'"
    )

# Expose the evaluate function
if hasattr(_fastevaluate, "evaluate"):
    evaluate = _fastevaluate.evaluate
else:
    raise AttributeError(
        "fastevaluate module does not have 'evaluate' attribute. "
        "The C++ extension may not have been compiled correctly. "
        f"Available attributes: {dir(_fastevaluate)}"
    )

__all__ = ["evaluate"]
