"""Kestrel Feature: visual identity"""
from importlib.metadata import PackageNotFoundError, version as _version

from .feature import VisualIdentityFeature

try:
    __version__ = _version("kestrel-feature-visual")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["VisualIdentityFeature", "__version__"]
