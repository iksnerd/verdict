"""verdict: typed decisions. The CLI is `verdict`; the Python API is loaded on first use, so
`import verdict` stays cheap for the CLI's startup."""
_SDK = ("Verdict", "VerdictError", "Result", "Noul", "Choice", "Score")


def __getattr__(name: str):
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("verdict")
        except PackageNotFoundError:
            return "0+unknown"
    if name in _SDK:
        from . import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module 'verdict' has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_SDK])
