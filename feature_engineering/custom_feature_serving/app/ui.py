"""Gradio UI — mounted at / by app.py.

Four components per the plan:
1. "Load sample" dropdown (samples come from notebook 05, baked into samples.json at deploy time)
2. Payload JSON textarea
3. Transform button
4. Features table + latency readout + raw JSON

Transforms call the pyfunc transformer directly in-process — no HTTP round-trip.
(Earlier design used an HTTP loopback to `localhost:8080`, which broke when the
Databricks Apps runtime injected a non-8080 port via DATABRICKS_APP_PORT.)
"""
import json
import pathlib
import time

import gradio as gr
import pandas as pd

import transformer_loader


FEATURE_VERSION = "0.1.0"

_SAMPLES_PATH = pathlib.Path(__file__).parent / "samples.json"
if _SAMPLES_PATH.exists():
    SAMPLES: dict[str, str] = json.loads(_SAMPLES_PATH.read_text())
else:
    SAMPLES = {}


def _transform_in_process(payload_json: str):
    if not payload_json.strip():
        return [["(empty input)", ""]], "-", ""
    t0 = time.perf_counter()
    try:
        transformer = transformer_loader.get()
        result = transformer.predict(None, pd.DataFrame({"payload_json": [payload_json]}))
        latency_ms = (time.perf_counter() - t0) * 1000
        features_json_str = result.iloc[0]["features_json"]
        features = json.loads(features_json_str)
        table_rows = [[k, str(v)] for k, v in features.items()]
        return table_rows, f"{latency_ms:.1f} ms", json.dumps(features, indent=2)
    except Exception as e:
        return [["error", str(e)]], "-", str(e)


def _load_sample(sample_id: str) -> str:
    return SAMPLES.get(sample_id, "") if sample_id else ""


def build_ui() -> gr.Blocks:
    sample_ids = sorted(SAMPLES.keys())
    header = (
        f"# Claims FE Transformer\n\n"
        f"**Wheel**: `claims_fe {FEATURE_VERSION}` — 32 engineered features. "
        f"Paste a claim JSON or load one of the {len(sample_ids)} samples."
    )

    with gr.Blocks(title="Claims FE Transformer") as ui:
        gr.Markdown(header)

        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=sample_ids,
                label="Load sample from fe_test_payloads",
                value=None,
                interactive=True,
            )

        payload_input = gr.TextArea(
            label="Payload JSON",
            lines=12,
            placeholder='{"Claims": [{...}]}',
        )
        transform_btn = gr.Button("Transform", variant="primary")

        with gr.Row():
            features_table = gr.Dataframe(
                headers=["feature", "value"],
                label="Extracted features",
                interactive=False,
                wrap=True,
            )
            latency_display = gr.Textbox(label="Latency", interactive=False)

        raw_json = gr.Code(label="Raw features_json", language="json")

        sample_dropdown.change(_load_sample, inputs=sample_dropdown, outputs=payload_input)
        transform_btn.click(
            _transform_in_process,
            inputs=payload_input,
            outputs=[features_table, latency_display, raw_json],
        )

    return ui
