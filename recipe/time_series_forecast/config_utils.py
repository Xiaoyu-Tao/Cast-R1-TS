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
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple


def _parse_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _load_lengths_from_base_yaml() -> Tuple[Optional[int], Optional[int]]:
    base_path = Path(__file__).parent / "base.yaml"
    if not base_path.exists():
        return None, None
    try:
        text = base_path.read_text(encoding="utf-8")
    except Exception:
        return None, None

    lookback_match = re.search(r"lookback_window:\s*(\d+)", text)
    horizon_match = re.search(r"forecast_horizon:\s*(\d+)", text)
    lookback = int(lookback_match.group(1)) if lookback_match else None
    horizon = int(horizon_match.group(1)) if horizon_match else None
    return lookback, horizon


def get_default_lengths() -> Tuple[int, int]:
    """
    Resolve default lookback/forecast lengths.
    Priority: ENV overrides -> base.yaml -> 96.
    """
    lookback = _parse_env_int("LOOKBACK_WINDOW")
    horizon = _parse_env_int("FORECAST_HORIZON")

    if lookback is None or horizon is None:
        base_lookback, base_horizon = _load_lengths_from_base_yaml()
        if lookback is None:
            lookback = base_lookback
        if horizon is None:
            horizon = base_horizon

    if lookback is None:
        lookback = 96
    if horizon is None:
        horizon = 96

    return lookback, horizon
