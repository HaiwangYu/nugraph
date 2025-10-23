import os

_fast = os.environ.get("NUGRAPH_FAST_IMPORT", "0").lower() in ("1","true","yes","y")
_skip_data   = os.environ.get("NUGRAPH_SKIP_DATA_IMPORT", "0").lower() in ("1","true","yes","y")
_skip_models = os.environ.get("NUGRAPH_SKIP_MODELS_IMPORT", "0").lower() in ("1","true","yes","y")

__all__: list[str] = []

if not _fast:
    if not _skip_data:
        from . import data
        __all__.append("data")
    if not _skip_models:
        from . import models
        __all__.append("models")

from . import util  # safe/lightweight; keep last if it’s truly light
