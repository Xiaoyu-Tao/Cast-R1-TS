# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TIMESERIES_SYSTEM_PROMPT = """You are a time series forecasting agent. This is a MULTI-TURN interaction.

##Dataset Description
This dataset consists of hourly electricity market prices used for Electricity Price Forecasting (EPF). The time series exhibits strong daily and weekly seasonality, high volatility, and frequent price spikes. The data is non-stationary and heavy-tailed, with abrupt regime changes driven by demand, supply, and market mechanisms. The forecasting task is to predict future electricity prices over short to medium horizons based on historical price patterns.

## Workflow (MUST follow this order across turns)

**Turn 1 - Feature Extraction ONLY**:
Call one or more feature extraction tools. Do NOT call predict_time_series yet.
- `extract_basic_statistics`: median, MAD, autocorrelation, spectral features, correlation, PCA
- `extract_within_channel_dynamics`: changepoints, slopes, peaks, entropy
- `extract_forecast_residuals`: AR residual diagnostics
- `extract_data_quality`: quantization, saturation, dropout
- `extract_event_summary`: segment patterns (rise/fall/oscillation)

**Turn 2 - Prediction**:
After seeing feature results in "Analysis History", call `predict_time_series` with the chosen model:
- 'patchtst': Local temporal patterns + long-range dependency
- 'itransformer': Strong cross-channel dependency
- 'arima': Linear trend + stable seasonality
- 'chronos2': Irregular or noisy patterns

**Turn 3 - Final Output**:
Reflect on feature analysis and model predictions, refine unreasonable results, and output the final forecast.

## Output Format (Turn 3 only)
Your response MUST contain ONLY the two tags below, in this order, with no extra text before/between/after them.
If you would "use the model predictions directly", paste the actual predicted values inside <answer> (do NOT say that phrase).
<think>[Reflect predictions, note any adjustments]</think>
<answer>
2017-05-05 00:00:00 12.345
...
</answer>

## CRITICAL RULES
- Turn 1: Feature extraction ONLY. Do NOT call predict_time_series.
- Turn 2: Call predict_time_series ONLY after features are extracted.
- Turn 3: Output answer ONLY after predictions are available.
- Do NOT output anything outside <think> and <answer>. Missing <answer> tags is incorrect.
"""

# OpenAI-compatible tool schemas for TimeSeriesForecast actions.
PREDICT_TIMESERIES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_time_series",
        "description": (
            "PREREQUISITE: You must have called feature extraction tools first (check 'Analysis History' is not empty). "
            "Do NOT call this on Turn 1 - extract features first! "
            "Models: 'chronos2' (default, preferred), 'itransformer' (cross-channel), 'arima' (trends), 'patchtst'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": (
                        "Model to use. Prefer 'chronos2' unless features strongly suggest another model."
                    ),
                    "enum": ["patchtst", "itransformer", "arima", "chronos2"]
                }
            },
            "required": ["model_name"],
        },
    },
}

EXTRACT_BASIC_STATISTICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_basic_statistics",
        "description": (
            "Extract core statistical features including median, MAD, autocorrelation, "
            "spectral features, CUSUM, quantile kurtosis, correlation, and PCA variance ratio."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_WITHIN_CHANNEL_DYNAMICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_within_channel_dynamics",
        "description": (
            "Extract within-channel dynamics including changepoints, slopes, flatlines, "
            "peaks, entropy, and run-lengths."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_FORECAST_RESIDUALS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_forecast_residuals",
        "description": (
            "Extract AR residual diagnostics including mean, max, exceedance, ACF, "
            "and concentration."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_DATA_QUALITY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_data_quality",
        "description": (
            "Extract data quality metrics including quantization, saturation, "
            "constant channels, and dropout."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

EXTRACT_EVENT_SUMMARY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_event_summary",
        "description": (
            "Extract event summary including segment count, rise/fall/flat/oscillation patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

TIMESERIES_TOOL_SCHEMAS = [
    EXTRACT_BASIC_STATISTICS_SCHEMA,
    EXTRACT_WITHIN_CHANNEL_DYNAMICS_SCHEMA,
    EXTRACT_FORECAST_RESIDUALS_SCHEMA,
    EXTRACT_DATA_QUALITY_SCHEMA,
    EXTRACT_EVENT_SUMMARY_SCHEMA,
    PREDICT_TIMESERIES_TOOL_SCHEMA,
]
