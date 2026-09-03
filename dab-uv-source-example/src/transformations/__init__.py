"""Transformations package — version is read from pyproject.toml."""

try:
    from importlib.metadata import version as _version
    __version__ = _version("dab-uv-source-example")
except Exception:
    __version__ = "0.0.0-dev"
