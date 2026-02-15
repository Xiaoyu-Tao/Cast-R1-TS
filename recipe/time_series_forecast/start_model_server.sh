#!/bin/bash
# =============================================================================
# Unified Time Series Prediction Service Startup Script
# =============================================================================
#
# This script starts the unified prediction service that loads all available
# models (Chronos2, PatchTST, iTransformer) on a dedicated GPU.
#
# Run this BEFORE starting the training script.
#
# Usage:
#   ./start_model_server.sh           # Uses GPU 3 by default, port 8994
#   ./start_model_server.sh 2         # Uses GPU 2
#   ./start_model_server.sh 3 8995    # Uses GPU 3, port 8995
#
# =============================================================================

# Configuration
GPU_ID=${1:-0}          # Default: GPU 3
PORT=${2:-8994}         # Default: port 8994
HOST="0.0.0.0"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the project root directory (two levels up from script dir)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Add project root to PYTHONPATH so that imports like 'recipe.time_series_forecast...' work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "============================================================"
echo "  Unified Time Series Prediction Service"
echo "============================================================"
echo "  GPU:  $GPU_ID"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo "============================================================"
echo ""
echo "  Available Models:"
echo "    - Chronos2:     Foundation model for time series"
echo "    - PatchTST:     Patch-based Transformer"
echo "    - iTransformer: Inverted Transformer"
echo ""
echo "  Models will be loaded if checkpoint exists."
echo "============================================================"

# Set CUDA device and start the server
cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES=$GPU_ID python model_server.py --host $HOST --port $PORT --device cuda

# =============================================================================
# Alternative: run with nohup in background
# =============================================================================
# Uncomment the lines below to run in background:
#
# CUDA_VISIBLE_DEVICES=$GPU_ID nohup python model_server.py --host $HOST --port $PORT --device cuda > model_server.log 2>&1 &
# echo "Server started in background. PID: $!"
# echo "Log file: model_server.log"
# echo ""
# echo "To check status: curl http://localhost:$PORT/health"
# echo "To see models:   curl http://localhost:$PORT/models"
