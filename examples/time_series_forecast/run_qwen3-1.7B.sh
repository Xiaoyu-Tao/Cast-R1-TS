set -x

export VLLM_USE_V1=1
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1

# Set PYTHONPATH to find arft module
export PYTHONPATH=

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$HOME/project/AgentRFT/recipe/time_series_forecast/base.yaml"

DATASET_PATH=

PROJECT_NAME='TimeSeriesForecast'
EXP_NAME=

python3 -m arft.main_agent_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files= \
    data.val_files= \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path= \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path=$CONFIG_PATH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 $@
