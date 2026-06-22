"""Startup patch for this venv (auto-loaded because pylib\\ is on PYTHONPATH).

Fixes a Windows-only bug in speechbrain 1.x that breaks pyannote speaker
diarization. speechbrain's LazyModule has a guard meant to stop Python's
inspect.py stack-walk from triggering its OPTIONAL integration imports
(k2, flair, ...). The guard checks the caller path ends with "/inspect.py"
(Unix slash). On Windows the path is "...\\inspect.py", so the guard never
fires, every optional integration tries to import, and the missing libraries
(k2 has no Windows wheels) raise and crash diarization model loading.

We replace LazyModule.ensure_module with the identical logic, but matching both
path separators. Done via a one-shot import hook so we patch the module right
after it loads, without importing heavy speechbrain at interpreter startup and
without editing anything under site-packages.
"""
import sys
import inspect
import warnings
import importlib
from importlib.machinery import PathFinder

_TARGET = "speechbrain.utils.importutils"


def _corrected_ensure_module(self, stacklevel):
    importer_frame = None
    try:
        importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
    except AttributeError:
        warnings.warn(
            "Failed to inspect frame to check if we should ignore importing a "
            "module lazily."
        )
    # The fix: match Windows "\\inspect.py" as well as Unix "/inspect.py".
    if importer_frame is not None and importer_frame.filename.endswith(
        ("/inspect.py", "\\inspect.py")
    ):
        raise AttributeError()
    if self.lazy_module is None:
        try:
            if self.package is None:
                self.lazy_module = importlib.import_module(self.target)
            else:
                self.lazy_module = importlib.import_module(
                    f".{self.target}", self.package
                )
        except Exception as e:
            raise ImportError(f"Lazy import of {repr(self)} failed") from e
    return self.lazy_module


def _patch(module):
    lazy_module_cls = getattr(module, "LazyModule", None)
    if lazy_module_cls is not None:
        lazy_module_cls.ensure_module = _corrected_ensure_module


class _OneShotPatcher:
    """Intercepts the import of speechbrain.utils.importutils, lets it load
    normally, then applies the patch. Removes itself after firing."""

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        original_exec = spec.loader.exec_module

        def exec_module(mod, _orig=original_exec):
            _orig(mod)
            _patch(mod)

        spec.loader.exec_module = exec_module
        return spec


# If speechbrain was somehow already imported, patch immediately; else hook it.
if _TARGET in sys.modules:
    _patch(sys.modules[_TARGET])
else:
    sys.meta_path.insert(0, _OneShotPatcher())
