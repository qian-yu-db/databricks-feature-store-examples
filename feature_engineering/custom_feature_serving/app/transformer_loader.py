"""Module-level singleton for the claims_fe transformer.

The wheel's regex compilation runs at first import of claims_fe.lexicons.
Instantiating ClaimFeatureTransformer is otherwise cheap — but we still
want exactly one instance per app process to keep things clean.
"""
from claims_fe.transformer import ClaimFeatureTransformer

_instance: ClaimFeatureTransformer | None = None


def get() -> ClaimFeatureTransformer:
    global _instance
    if _instance is None:
        _instance = ClaimFeatureTransformer()
    return _instance
