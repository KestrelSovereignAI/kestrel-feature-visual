"""Kestrel Feature: visual identity"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .feature import VisualIdentityFeature
from .selfie_spec import (
    SELFIE_SPEC_SCHEMA_VERSION,
    ResolvedLoraSelfieSpec,
    ResolvedSelfiePrompt,
    bind_lora_selfie_spec,
    resolve_lora_selfie_spec,
    resolve_selfie_prompt,
)

try:
    __version__ = _version("kestrel-feature-visual")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = [
    "SELFIE_SPEC_SCHEMA_VERSION",
    "ResolvedLoraSelfieSpec",
    "ResolvedSelfiePrompt",
    "VisualIdentityFeature",
    "__version__",
    "bind_lora_selfie_spec",
    "resolve_lora_selfie_spec",
    "resolve_selfie_prompt",
]
