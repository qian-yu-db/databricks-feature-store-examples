"""ClaimFeatureTransformer — MLflow pyfunc wrapper around the claims_fe extractors."""
import json

import mlflow.pyfunc
import pandas as pd

from claims_fe.features.structural import structural_features
from claims_fe.features.temporal import temporal_features
from claims_fe.features.text_flags import text_flag_features


FEATURE_VERSION = "0.1.0"


def _extract_one(payload_json: str) -> dict:
    errors = []
    try:
        payload = json.loads(payload_json)
    except Exception as exc:
        return {
            "claim_id": None, "claim_num": None,
            "feature_version": FEATURE_VERSION,
            "extraction_errors": [f"json_parse_error: {exc}"],
        }

    claims = payload.get("Claims") or []
    if not claims:
        return {
            "claim_id": None, "claim_num": None,
            "feature_version": FEATURE_VERSION,
            "extraction_errors": ["no_claims_in_payload"],
        }

    claim = claims[0]
    out = {}
    for fn, name in [
        (structural_features, "structural"),
        (temporal_features, "temporal"),
        (text_flag_features, "text_flags"),
    ]:
        try:
            out.update(fn(claim))
        except Exception as exc:
            errors.append(f"{name}_error: {exc}")

    out["feature_version"] = FEATURE_VERSION
    out["extraction_errors"] = errors
    return out


class ClaimFeatureTransformer(mlflow.pyfunc.PythonModel):
    """
    Pattern A feature-engineering pyfunc.

    Input : DataFrame with a single string column 'payload_json' (raw claim JSON).
    Output: DataFrame with a single string column 'features_json' (engineered features).
    """

    def load_context(self, context):
        # Nothing to load — lexicons and regex are compiled at module import.
        pass

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            series = model_input.iloc[:, 0]
        else:
            series = pd.Series(model_input)

        outputs = [json.dumps(_extract_one(s)) for s in series.astype(str).tolist()]
        return pd.DataFrame({"features_json": outputs})
