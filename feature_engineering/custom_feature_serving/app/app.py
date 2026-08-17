"""FastAPI entry point for the claims_fe feature-engineering Databricks App.

- POST /transform : same contract as the Model Serving pyfunc
- GET  /health   : liveness + feature version
- GET  /ui       : Gradio manual-inspection UI (mounted)

Same underlying logic as the pyfunc (the claims_fe wheel); the difference is
the serving shell — FastAPI + orjson instead of MLflow's scoring server.
"""
from contextlib import asynccontextmanager

import gradio as gr
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

import transformer_loader
from models import HealthResponse, TransformRequest, TransformResponse
from ui import build_ui


FEATURE_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the transformer singleton before the first request — avoids first-request spike
    transformer_loader.get()
    yield


app = FastAPI(
    title="Claims FE Transformer",
    version=FEATURE_VERSION,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, feature_version=FEATURE_VERSION)


@app.post("/transform", response_model=TransformResponse)
def transform(req: TransformRequest) -> TransformResponse:
    transformer = transformer_loader.get()
    result = transformer.predict(None, pd.DataFrame({"payload_json": [req.payload_json]}))
    return TransformResponse(features_json=result.iloc[0]["features_json"])


# Mount the Gradio UI at the ROOT. The root URL of the deployed app is the UI —
# users land on something useful, and the Databricks Apps platform probe to GET /
# gets a 200 HTML response. Explicit FastAPI routes above (/health, /transform,
# /docs, /openapi.json) take precedence over the Gradio mount.
app = gr.mount_gradio_app(app, build_ui(), path="/")


if __name__ == "__main__":
    # Databricks Apps injects the target port via one of several env vars depending on
    # runtime version. Templates that use `python app.py` + `gradio.launch()` rely on
    # Gradio reading these automatically. We do the same here so uvicorn binds to the
    # port the platform proxy is routing traffic to — hardcoding 8080 is what caused
    # "App Not Available" with zero access logs (proxy forwarding to a dark port).
    import os

    import uvicorn

    port = int(
        os.environ.get("DATABRICKS_APP_PORT")
        or os.environ.get("GRADIO_SERVER_PORT")
        or os.environ.get("PORT")
        or "8080"
    )
    host = os.environ.get("GRADIO_SERVER_NAME") or "0.0.0.0"
    print(f"Starting uvicorn on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
