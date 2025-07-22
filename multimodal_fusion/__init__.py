"""Multi-Modal Electron Microscopy Data Fusion Package."""

__version__ = "0.5.0"

# Import the compiled C++ module (now inside the package)
try:
    from multimodal_fusion.ctvlib import ctvlib  # Import from within package
    _cpp_available = True
except ImportError as e:
    print(f"Warning: Could not import ctvlib module: {e}")
    print("You may need to compile the C++ extensions.")
    print("Run: pip install -e . --verbose")
    ctvlib = None
    _cpp_available = False

# Expose main classes at package level 
from multimodal_fusion.fusion import DataFusion
