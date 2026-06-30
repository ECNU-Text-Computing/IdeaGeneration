#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"

TRAIN_DATA="${TRAIN_DATA:-/path/to/train_data.parquet}"
VAL_DATA="${VAL_DATA:-/path/to/val_data.parquet}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-5000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-5000}"

CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"

ROLLOUT_N="${ROLLOUT_N:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"

USE_EAIG="${USE_EAIG:-true}"
EAIG_WEIGHT="${EAIG_WEIGHT:-0.3}"
EAIG_TOP_P_RATIO="${EAIG_TOP_P_RATIO:-0.25}"

USE_EAIG_SHAPING="${USE_EAIG_SHAPING:-true}"

USE_LENGTH_ANCHOR="${USE_LENGTH_ANCHOR:-true}"
LENGTH_ANCHOR_LAMBDA="${LENGTH_ANCHOR_LAMBDA:-0.5}"

SEMANTIC_WEIGHT="${SEMANTIC_WEIGHT:-0.7}"
USE_SEMANTIC_SHAPING="${USE_SEMANTIC_SHAPING:-true}"

USE_SEMANTIC_SIMILARITY="${USE_SEMANTIC_SIMILARITY:-true}"
EMBEDDING_API_URL="${EMBEDDING_API_URL:-http://127.0.0.1:30015/v1/embeddings}"

USE_OVERVIEW_ONLY="${USE_OVERVIEW_ONLY:-true}"
USE_CONTRASTIVE_GAIN="${USE_CONTRASTIVE_GAIN:-true}"

USE_FORMAT_PENALTY="${USE_FORMAT_PENALTY:-true}"

N_GPUS="${N_GPUS:-8}"
N_NODES="${N_NODES:-1}"
TP_SIZE="${TP_SIZE:-1}"

PROJECT_NAME="${PROJECT_NAME:-idea_gen}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-idea_gen_semantic_${SEMANTIC_WEIGHT}_eaig_${EAIG_WEIGHT}}"

LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"

python3 -u -m verl.trainer.main_idea_gen \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.reward_fn_key='data_source' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    reward_model.enable=False \
    reward_model.reward_manager=entropy \
    +reward_model.reward_kwargs.use_eaig=$USE_EAIG \
    +reward_model.reward_kwargs.eaig_weight=$EAIG_WEIGHT \
    +reward_model.reward_kwargs.eaig_top_p_ratio=$EAIG_TOP_P_RATIO \
    +reward_model.reward_kwargs.use_eaig_shaping=$USE_EAIG_SHAPING \
    +reward_model.reward_kwargs.use_length_anchor=$USE_LENGTH_ANCHOR \
    +reward_model.reward_kwargs.length_anchor_lambda=$LENGTH_ANCHOR_LAMBDA \
    +reward_model.reward_kwargs.semantic_weight=$SEMANTIC_WEIGHT \
    +reward_model.reward_kwargs.use_semantic_shaping=$USE_SEMANTIC_SHAPING \
    +reward_model.reward_kwargs.use_semantic_similarity=$USE_SEMANTIC_SIMILARITY \
    +reward_model.reward_kwargs.embedding_api_url=$EMBEDDING_API_URL \
    +reward_model.reward_kwargs.use_overview_only=$USE_OVERVIEW_ONLY \
    +reward_model.reward_kwargs.use_contrastive_gain=$USE_CONTRASTIVE_GAIN \
    +reward_model.reward_kwargs.use_format_penalty=$USE_FORMAT_PENALTY \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=2000 \
    trainer.total_epochs=2 \
    trainer.rollout_data_dir="${OUTPUT_DIR:-./output}/${EXPERIMENT_NAME}/rollout_data" \
    trainer.default_local_dir="${OUTPUT_DIR:-./output}/${EXPERIMENT_NAME}" \
    "$@"
