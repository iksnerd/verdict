"""Documented import path, kept: the module is `verdict.backends.systemone`."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("verdict.backends.systemone")
