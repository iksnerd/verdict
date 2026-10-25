"""Old import path, kept so existing code runs: the module is `verdict.backends.systemone`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("verdict.backends.systemone")
