"""Wire-Cell labeling and NuGraph conversion utilities.

Imports are lazy so the lightweight labeling API remains usable in the SL7
labeling environment, which intentionally does not provide Torch/NuGraph.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ConversionConfig": ("pywcml.config", "ConversionConfig"),
    "EventIdentity": ("pywcml.identity", "EventIdentity"),
    "LabelingConfig": ("pywcml.labeling", "LabelingConfig"),
    "NeutrinoVertex": ("pywcml.labeling", "NeutrinoVertex"),
    "RecoArrays": ("pywcml.labeling", "RecoArrays"),
    "SemanticTruth": ("pywcml.labeling", "SemanticTruth"),
    "SimIDETruth": ("pywcml.labeling", "SimIDETruth"),
    "StreamingH5Writer": ("pywcml.h5writer", "StreamingH5Writer"),
    "WCMLArrays": ("pywcml.io", "WCMLArrays"),
    "WCMLConverter": ("pywcml.converter", "WCMLConverter"),
    "convert_npz_directory": ("pywcml.converter", "convert_npz_directory"),
    "convert_npz_file": ("pywcml.converter", "convert_npz_file"),
    "label_event": ("pywcml.labeling", "label_event"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
