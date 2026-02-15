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
import asyncio
import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from arft.agent_flow.agent_flow import AgentFlowBase, AgentFlowOutput, AgentFlowStep, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.time_series_forecast.prompts import *
from recipe.time_series_forecast.utils import (
    parse_time_series_string,
    parse_time_series_to_dataframe,
    format_predictions_to_string,
    get_last_timestamp,
    predict_time_series_async,
    extract_basic_statistics,
    format_basic_statistics,
    extract_within_channel_dynamics,
    format_within_channel_dynamics,
    extract_forecast_residuals,
    format_forecast_residuals,
    extract_data_quality,
    format_data_quality,
    extract_event_summary,
    format_event_summary,
)
from recipe.time_series_forecast.reward import (
    compute_score,
    extract_values_from_time_series_string,
    extract_ground_truth_values,
)
import numpy as np

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("time_series_forecast_agent")
class TimeSeriesForecastAgentFlow(AgentFlowBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_analysis = []
        self.time_series_data = ""  # Raw string data
        self.steps = []
        
        # Parsed data cache
        self.timestamps = None
        self.values = None
        self.prediction_results = None
        self.final_answer = None

        # Feature extraction caches
        self.basic_statistics = None
        self.within_channel_dynamics = None
        self.forecast_residuals = None
        self.data_quality = None
        self.event_summary = None
        self.io_log_path = os.getenv(
            "TS_FORECAST_IO_JSONL_PATH",
            os.path.join(os.path.dirname(__file__), "time_series_forecast_io.jsonl"),
        )

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level TimeSeriesForecastAgentFlow initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_steps = kwargs.get("max_steps", 5)
        cls.max_parallel_calls = kwargs.get("max_parallel_calls", 5)
        cls.lookback_window = kwargs.get("lookback_window", 96)
        cls.forecast_horizon = kwargs.get("forecast_horizon", 96)
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.tool_schemas = TIMESERIES_TOOL_SCHEMAS

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        self.time_series_data = raw_prompt[0]["content"]
        
        # Get ground_truth from reward_model field
        reward_model = kwargs.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        
        # Parse the input data once at the beginning
        try:
            self.timestamps, self.values = parse_time_series_string(self.time_series_data)
        except Exception as e:
            logger.error(f"Error parsing time series data: {e}")
            self.timestamps, self.values = [], []

        metrics = {}
        request_id = uuid4().hex[:8]
        
        MSE_data = np.nan
        MAE_data = np.nan
        io_records: list[dict[str, Any]] = []

        num_steps = 0
        while num_steps < self.max_steps:
            num_steps += 1

            messages = [
                {"role": "system", "content": TIMESERIES_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt()}
            ]

            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )

            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            response_ids = output.token_ids[: self.response_length]
            
            # Decode response to check for final answer
            response_text = await self.loop.run_in_executor(None, self.tokenizer.decode, response_ids)

            # io_records.append(
            #     {
            #         "step": num_steps,
            #         "input": messages[1]["content"],
            #         "output": response_text,
            #     }
            # )

            # Check if response contains final answer with <answer> tags
            final_answer = self._extract_final_answer(response_text)
            
            if final_answer:
                # Final answer detected - but first validate the workflow was followed
                workflow_valid, workflow_penalty, workflow_msg = self._validate_workflow_completion(final_answer)
                
                if not workflow_valid:
                    # Workflow not completed - apply penalty (reward hacking prevention)
                    reward_score = workflow_penalty
                    logger.warning(f"Workflow violation detected: {workflow_msg}. Penalty: {reward_score}")
                    # Don't accept this as final answer - force model to continue
                    # Set final_answer to None so the loop continues
                    final_answer = None
                else:
                    # Workflow completed - compute reward based on prediction accuracy
                    self.final_answer = final_answer
                    reward_score = self._compute_final_reward(final_answer, ground_truth)
                    logger.info(f"Final answer detected. Reward score: {reward_score}")
                    # if reward_score > 0.5:
                    #     for record in io_records:
                    #         record["request_id"] = request_id
                    #         record["reward_score"] = reward_score
                    #     await self._append_jsonl_records(self.io_log_path, io_records)
            else:
                # No final answer yet - process tool calls
                _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
                tool_calls = tool_calls[:self.max_parallel_calls]
                
                # Process tool calls (no intermediate reward for tool calls)
                for tool_call in tool_calls:
                    if tool_call.name == "predict_time_series":
                        if not self._has_any_feature_analysis():
                            logger.warning("predict_time_series called without feature analysis, rejected")
                            continue
                        
                        model_name = None
                        if hasattr(tool_call, 'arguments') and tool_call.arguments:
                            args = tool_call.arguments
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}
                            if isinstance(args, dict):
                                model_name = args.get("model_name")
                        if model_name not in {"chronos2", "arima", "patchtst", "itransformer"}:
                            model_name = "chronos2"
                        await self.predict(model_name=model_name, **kwargs)
                    elif tool_call.name == "extract_basic_statistics":
                        await self.extract_basic_statistics(**kwargs)
                    elif tool_call.name == "extract_within_channel_dynamics":
                        await self.extract_within_channel_dynamics(**kwargs)
                    elif tool_call.name == "extract_forecast_residuals":
                        await self.extract_forecast_residuals(**kwargs)
                    elif tool_call.name == "extract_data_quality":
                        await self.extract_data_quality(**kwargs)
                    elif tool_call.name == "extract_event_summary":
                        await self.extract_event_summary(**kwargs)
                
                
                # Small reward for making progress (using tools correctly)
                reward_score = 0.05 if tool_calls else 0.0

            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                reward_score=reward_score,
            )
            step = await self._postprocess(step, **kwargs)
            step.extra_fields['reward_extra_info'] = {'MSE': MSE_data, 'MAE': MAE_data}
            self.steps.append(step)
            
            # If final answer is detected, we can stop
            if final_answer:
                break

        
        if self.final_answer and ground_truth:
            try:
                pred_values = extract_values_from_time_series_string(self.final_answer)
                gt_values = extract_ground_truth_values(ground_truth)
                
                if pred_values and gt_values:
                    # Align lengths (use minimum length)
                    min_len = min(len(pred_values), len(gt_values))
                    pred_arr = np.array(pred_values[:min_len])
                    gt_arr = np.array(gt_values[:min_len])
                    
                    # Calculate MSE and MAE
                    MSE_data = float(np.mean((pred_arr - gt_arr) ** 2))
                    MAE_data = float(np.mean(np.abs(pred_arr - gt_arr)))
                    
                    logger.info(f"Metrics - MSE: {MSE_data:.4f}, MAE: {MAE_data:.4f}")
            except Exception as e:
                logger.error(f"Error calculating MSE/MAE: {e}")
        
        self.steps[-1].extra_fields['reward_extra_info'] = {'MSE': MSE_data, 'MAE': MAE_data}

        return AgentFlowOutput(steps=self.steps, metrics=metrics)

    def _validate_workflow_completion(self, final_answer: str) -> tuple[bool, float, str]:
        """
        Validate that the workflow was properly completed before accepting final answer.
        Prevents reward hacking where model skips tools and copies input data.
        
        Returns:
            Tuple of (is_valid, penalty_score, message)
        """
        # Check 1: Must have called at least one feature extraction tool
        if not self._has_any_feature_analysis():
            return False, -0.5, "No feature extraction tools were called. You must analyze the data first."
        
        # Check 2: Must have called predict_time_series
        if self.prediction_results is None:
            return False, -0.5, "predict_time_series was not called. You must get model predictions first."
        
        # Check 3: Check if answer is just copying the input data (reward hacking detection)
        if self._is_copying_input(final_answer):
            return False, -1.0, "Answer appears to copy input data. Predictions must be for FUTURE timestamps."
        
        return True, 0.0, "Workflow completed correctly"
    
    def _is_copying_input(self, final_answer: str) -> bool:
        """
        Detect if the model is copying input data instead of providing predictions.
        Checks if timestamps in answer overlap significantly with input timestamps.
        """
        if self.timestamps is None or len(self.timestamps) == 0:
            return False
        
        # Extract timestamps from the answer
        answer_timestamps = []
        lines = final_answer.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to extract timestamp (format: "2018-05-19 00:00:00 value")
            match = re.match(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                answer_timestamps.append(match.group(1))
        
        if not answer_timestamps:
            return False
        
        # Convert input timestamps to strings for comparison
        input_ts_strings = set()
        for ts in self.timestamps:
            if hasattr(ts, 'strftime'):
                input_ts_strings.add(ts.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                input_ts_strings.add(str(ts))
        
        # Count how many answer timestamps are in the input
        overlap_count = sum(1 for ts in answer_timestamps if ts in input_ts_strings)
        overlap_ratio = overlap_count / len(answer_timestamps) if answer_timestamps else 0
        
        # If more than 50% of answer timestamps are from input, it's likely copying
        if overlap_ratio > 0.5:
            logger.warning(f"Detected input copying: {overlap_ratio:.1%} of answer timestamps match input")
            return True
        
        return False

    def _extract_final_answer(self, response_text: str) -> Optional[str]:
        """
        Extract final answer from response text if <answer> tags are present.
        
        Args:
            response_text: The model's response text
            
        Returns:
            The content inside <answer> tags, or None if not found
        """
        match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _compute_final_reward(self, final_answer: str, ground_truth: str) -> float:
        """
        Compute reward score based on final prediction and ground truth.
        
        Args:
            final_answer: The predicted values as string
            ground_truth: The ground truth values as string
            
        Returns:
            Reward score
        """
        if not ground_truth:
            return 0.0
        
        try:
            # Use the compute_score function from reward.py
            score = compute_score(
                data_source="time_series",
                solution_str=f"<answer>{final_answer}</answer>",
                ground_truth=ground_truth
            )
            return score
        except Exception as e:
            logger.error(f"Error computing final reward: {e}")
            return -0.5

    async def _append_jsonl_records(self, path: str, records: list[dict[str, Any]]) -> None:
        def _write_records() -> None:
            if not records:
                return
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        await self.loop.run_in_executor(None, _write_records)

    def _truncate_time_series_data(self, data: str, head: int = 5, tail: int = 5) -> str:
        """
        Truncate time series data to show only first and last few rows.
        This saves tokens while preserving context about data format and range.
        """
        lines = data.strip().split('\n')
        if len(lines) <= head + tail:
            return data
        
        head_lines = lines[:head]
        tail_lines = lines[-tail:]
        omitted = len(lines) - head - tail
        
        return '\n'.join(head_lines) + f'\n... ({omitted} rows omitted) ...\n' + '\n'.join(tail_lines)

    def _get_current_turn_info(self) -> tuple[int, str]:
        """
        Determine current turn number and action based on state.
        Returns: (turn_number, action_instruction)
        """
        has_features = self._has_any_feature_analysis()
        has_predictions = self.prediction_results is not None
        
        if not has_features:
            return 1, "Call feature extraction tools (e.g., extract_basic_statistics). Do NOT call predict_time_series yet."
        elif not has_predictions:
            return 2, "Call predict_time_series with your chosen model (e.g., 'chronos2')."
        else:
            return 3, (
                "Output your final answer in <think>...</think><answer>...</answer> format using the model predictions. "
                "Do NOT output any text outside these tags."
            )

    def _build_user_prompt(self) -> str:
        """
        Build user prompt dynamically based on current state.
        After predictions: keep partial data (first/last 5 rows) to save tokens.
        Before predictions: include full raw data for model analysis.
        """
        history = self._format_history_analysis()
        prediction = self._format_prediction_results()
        turn_num, action = self._get_current_turn_info()
        
        if self.prediction_results:
            truncated_data = self._truncate_time_series_data(self.time_series_data)
            return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {self.lookback_window} rows
### Forecast Horizon: {self.forecast_horizon} rows
### Historical Data (truncated)
{truncated_data}

### Analysis History
{history}

### Model Predictions
{prediction}

**Instructions**:
1. Reflect on whether the model predictions are reasonable given the historical patterns.
2. Based on this reflection, refine the predictions.
3. Output the final refined prediction.
4. Output ONLY <think>...</think> and <answer>...</answer>. Do not include anything else.

<think>
[Reflection on the consistency between historical patterns and model predictions]
</think>

<answer>
[Final prediction after reflection and refinement]
</answer>"""
        else:
            return f"""**[Turn {turn_num}] Action: {action}**
### Lookback Window: {self.lookback_window} rows
### Forecast Horizon: {self.forecast_horizon} rows
### Historical Data
{self.time_series_data}

### Analysis History
{history}

### Model Predictions
{prediction}

**Check your current state and act accordingly:**
- If "Analysis History" is empty → Call feature extraction tools. Do NOT call predict_time_series yet.
- If "Analysis History" has features but "Model Predictions" is empty → Call predict_time_series with model_name.
"""

    def _format_history_analysis(self) -> str:
        """Format history analysis records"""
        if not self.history_analysis:
            return "No previous analysis performed."
        
        return "\n".join(self.history_analysis)  # Show all analyses

    def _format_prediction_results(self) -> str:
        """Format prediction results for prompt"""
        if not self.prediction_results:
            return "No predictions available yet. Call predict_time_series to generate forecasts."
        
        return f"Model Predictions ({self.forecast_horizon} steps):\n{self.prediction_results}"

    def _has_any_feature_analysis(self) -> bool:
        """Check if any feature analysis has been performed."""
        return any([
            self.basic_statistics is not None,
            self.within_channel_dynamics is not None,
            self.forecast_residuals is not None,
            self.data_quality is not None,
            self.event_summary is not None,
        ])

    async def predict(self, model_name: str = "chronos2", **kwargs) -> float:
        """
        Generate predictions using the specified model.

        Args:
            model_name: Name of the model to use ("chronos2" or "arima")

        Returns:
            Reward score (0.0 for tool execution, actual reward at final step)
        """
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for prediction")
                return 0.0

            # Convert to DataFrame for prediction
            context_df = parse_time_series_to_dataframe(self.time_series_data)

            # Generate predictions using the specified model
            pred_df = await predict_time_series_async(
                context_df,
                prediction_length=self.forecast_horizon,
                model_name=model_name,
            )

            # Get last timestamp for formatting
            last_ts = get_last_timestamp(self.time_series_data)

            # Format predictions as string
            self.prediction_results = format_predictions_to_string(pred_df, last_ts)

            # Record to history with model name
            self.history_analysis.append(
                f"Model Prediction: Generated {self.forecast_horizon}-step forecast using {model_name.upper()} model"
            )

            logger.info(f"Prediction completed using {model_name} model")

            return 0.0  # No intermediate reward

        except Exception as e:
            logger.error(f"Error in predict with {model_name}: {e}")
            self.history_analysis.append(f"Prediction failed ({model_name}): {str(e)}")
            return 0.0

    async def extract_basic_statistics(self, **kwargs) -> float:
        """Extract core statistical features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for basic statistics extraction")
                return 0.0

            features = extract_basic_statistics(data=self.values)
            self.basic_statistics = features

            analysis_record = format_basic_statistics(features)
            self.history_analysis.append(analysis_record)

            logger.info("Basic statistics extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_basic_statistics: {e}")
            return 0.0

    async def extract_within_channel_dynamics(self, **kwargs) -> float:
        """Extract within-channel dynamics features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for within-channel dynamics extraction")
                return 0.0

            features = extract_within_channel_dynamics(data=self.values)
            self.within_channel_dynamics = features

            analysis_record = format_within_channel_dynamics(features)
            self.history_analysis.append(analysis_record)

            logger.info("Within-channel dynamics extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_within_channel_dynamics: {e}")
            return 0.0

    async def extract_forecast_residuals(self, **kwargs) -> float:
        """Extract forecast residual features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for forecast residuals extraction")
                return 0.0

            features = extract_forecast_residuals(data=self.values)
            self.forecast_residuals = features

            analysis_record = format_forecast_residuals(features)
            self.history_analysis.append(analysis_record)

            logger.info("Forecast residuals extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_forecast_residuals: {e}")
            return 0.0

    async def extract_data_quality(self, **kwargs) -> float:
        """Extract data quality features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for data quality extraction")
                return 0.0

            features = extract_data_quality(data=self.values)
            self.data_quality = features

            analysis_record = format_data_quality(features)
            self.history_analysis.append(analysis_record)

            logger.info("Data quality extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_data_quality: {e}")
            return 0.0

    async def extract_event_summary(self, **kwargs) -> float:
        """Extract event summary features from time series data."""
        try:
            if not self.values or len(self.values) < 2:
                logger.warning("Insufficient data for event summary extraction")
                return 0.0

            features = extract_event_summary(data=self.values)
            self.event_summary = features

            analysis_record = format_event_summary(features)
            self.history_analysis.append(analysis_record)

            logger.info("Event summary extraction completed")
            return 0.0
        except Exception as e:
            logger.error(f"Error in extract_event_summary: {e}")
            return 0.0
