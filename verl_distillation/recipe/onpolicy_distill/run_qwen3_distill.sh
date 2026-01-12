#!/bin/bash
# On-policy Distillation: distill from a teacher model (e.g., Qwen3-1.7B) to a student model
# with extended vocabulary (e.g., recommendation pretrained model with item tokens).
#
# Usage:
#   export BASE_MODEL=/path/to/student_model
#   export TEACHER_MODEL=/path/to/teacher_model
#   export DATASET_PARQUET=/path/to/train.parquet
#   bash run_qwen3_1.7b_distill.sh [hostfile]

set -x
HOME=$(pwd)
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")

# tmp_hostfile_dir 需要保留,框架需要
HOSTFILE="${1:-/etc/mpi/hostfile}"
NODES=$(wc -l < $HOSTFILE)

if [ ! -d "$HOME/tmp_hostfile_dir" ]; then
    mkdir -p "$HOME/tmp_hostfile_dir"
fi
if [ ! -d "$HOME/timeline_dir" ]; then
    mkdir -p "$HOME/timeline_dir"
fi
cat $HOSTFILE > "$HOME/tmp_hostfile_dir/hostfile_$timestamp"

N_GPUS_PER_NODE=2

project_name="verl_on_policy_distill"

experiment_name="verl_1.7b_distill_${NODES}_${timestamp}"
CKPT_HOME=${CKPT_HOME:-"$HOME/outputs"}
CKPT_DIR=${CKPT_DIR:-"${CKPT_HOME}/ckpts/${project_name}/${experiment_name}/"}

rollout_mode="async"
rollout_name="sglang" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

export HYDRA_FULL_ERROR=1

# rollout buffer setting
NUM_WORKER=16
CUDA_GRAPH_MAX_BS=64

# ===== Open-source friendly defaults =====
# You MUST set these paths for your own environment.
export BASE_MODEL=${BASE_MODEL:-""}
export TEACHER_MODEL=${TEACHER_MODEL:-""}
export DATASET_PARQUET=${DATASET_PARQUET:-"$(realpath ../output/onpolicy_distillation.parquet)"}

# Logging: default is console only.
# To enable W&B, export WANDB_API_KEY and override trainer.logger:
#   export WANDB_API_KEY="your-key"
#   ... trainer.logger='[console,wandb]' ...
export WANDB_API_KEY=${WANDB_API_KEY:-""}

if [ -z "$BASE_MODEL" ] || [ -z "$TEACHER_MODEL" ] || [ -z "$DATASET_PARQUET" ]; then
  echo "[ERROR] Please set BASE_MODEL / TEACHER_MODEL / DATASET_PARQUET before running."
  echo "  BASE_MODEL=$BASE_MODEL"
  echo "  TEACHER_MODEL=$TEACHER_MODEL"
  echo "  DATASET_PARQUET=$DATASET_PARQUET"
  exit 1
fi

export USE_DYNAMIC_BSZ=True # 是否开启动态batch size, 则无视上述batch_size设置，按token数来分配显卡，避免某张显卡处理的token数过多导致OOM显存溢出
export MAX_TOKENS_PER_GPU=24000  # n*(prompt_len+response_len)

export TRAIN_BATCH_SIZE=32
export LEARNING_RATE=5e-6


export ROLLOUT_N=1  # 每个prompt的CoT采样数量
export BEAM_SIZE_PER_ROLLOUT=1  # 每个CoT的beam search数量
export TEMPERATURE=1.1
export ENABLE_THINK=True  # 是否在user prompt末尾添加/think
export THINK_MODE="auto"
export MAX_RESPONSE_LEN=2048

export DISTILL_ADV_MAX=5.0
export DISTILL_ADV_MIN=-30.0

# ===== Extended vocabulary distillation settings =====
# Token ID threshold: tokens with id >= this value are considered "extended vocab tokens"
# For Qwen3 with OneRec item tokens, 151669 is the start of extended vocabulary.
# Set to empty string or "null" to disable extended vocab handling.
export EXTEND_VOCAB_START_TOKEN=151669
# Whether to mask the entire response if it contains any extended token
export MASK_RESPONSE_IF_HAVE_EXTEND_TOKEN=False

export TRAIN_FILES=$DATASET_PARQUET
export VAL_FILES=$DATASET_PARQUET

echo "Training files: $TRAIN_FILES"
echo "Validation files: $VAL_FILES"


PYTHONUNBUFFERED=1 python3 -m recipe.onpolicy_distill.main_onpolicy_distill --config-name='onpolicy_distill_trainer'\
    +ray_kwargs.ray_init.runtime_env.env_vars.TRACE_GPU_MEM=False \
    +ray_kwargs.ray_init.runtime_env.env_vars.WORK_DIR=$HOME \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="$WANDB_API_KEY" \
    +ray_kwargs.ray_init.runtime_env.env_vars.nosp="1" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_ECE_ENABLE="0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="32" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NVTE_ALLOW_NONDETERMINISTIC_ALGO="1" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NVLS_ENABLE="0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONWARNINGS="ignore" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG="VERSION" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_GID_INDEX="3" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_ASYNC_ERROR_HANDLING="1" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="bond0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_HCA="mlx5" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_PXN_DISABLE="0" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_QPS_PER_CONNECTION="4" \
    +ray_kwargs.ray_init.runtime_env.env_vars.SGLANG_VLM_CACHE_SIZE_MB="512" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TIMESTAMP=$timestamp \
    algorithm.adv_estimator=on_policy_distill \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.max_prompt_length=10240 \
    ++data.enable_think=$ENABLE_THINK \
    ++data.think_mode=$THINK_MODE \
    data.prompt_key=prompt \
    data.image_key=dummy \
    data.video_key=dummy \
    ++data.data_source_key='source' \
    data.reward_fn_key='source' \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=$return_raw_chat \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    +actor_rollout_ref.ref.model.path=$TEACHER_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.ref_log_prob_replace_val=-100 \
    actor_rollout_ref.ref.ref_log_prob_replace_val=-100 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.extend_vocab_start_token=$EXTEND_VOCAB_START_TOKEN \
    actor_rollout_ref.rollout.mask_response_if_have_extend_token=$MASK_RESPONSE_IF_HAVE_EXTEND_TOKEN \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=16384 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=$CUDA_GRAPH_MAX_BS \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=info \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_requests=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_requests_level=2 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=200 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.agent.num_workers=$NUM_WORKER \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    algorithm.use_kl_in_reward=False \
    ++algorithm.distill_adv_max_clip=$DISTILL_ADV_MAX \
    ++algorithm.distill_adv_min_clip=$DISTILL_ADV_MIN \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    trainer.logger='[console]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NODES \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=100 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.rollout_data_dir=$HOME \
    +trainer.validation_data_dir=$HOME \
    +trainer.ray_timeline_dir=$HOME/tmp_hostfile_dir \
    trainer.total_epochs=1 2>&1 | tee $project_name-$experiment_name-$timestamp.log

