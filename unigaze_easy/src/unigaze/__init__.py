

# Expose package version from installed metadata
try:
    from importlib.metadata import version, PackageNotFoundError  # Py3.8+: importlib_metadata backport if needed
except ImportError:  # very old Python
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("unigaze")   # <-- must match your PyPI/package name
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for source tree without installation

from .loader import load