"""Pydantic v2 request / response models for the FE transform endpoint."""
from pydantic import BaseModel, Field


class TransformRequest(BaseModel):
    payload_json: str = Field(
        ...,
        description="Raw claim JSON as a string. Shape: {\"Claims\": [<claim>]}.",
    )


class TransformResponse(BaseModel):
    features_json: str = Field(
        ...,
        description="Flat JSON string with engineered features — same contract as the pyfunc.",
    )


class HealthResponse(BaseModel):
    ok: bool
    feature_version: str
