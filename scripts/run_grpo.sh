#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to the prepared training Parquet file}"
: "${VAL_PARQUET:?Set VAL_PARQUET to the PrimeVul validation Parquet file}"
: "${TAXONOMY_JSON:?Set TAXONOMY_JSON to the frozen CWE taxonomy JSON}"
: "${CPG_SNAPSHOT_DIR:?Set CPG_SNAPSHOT_DIR to the precomputed Joern snapshots}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a persistent checkpoint directory}"
: "${MODEL_PATH:?Set MODEL_PATH to the supervised-SFT checkpoint}"

HIERARCHY_REWARD_MODE="${HIERARCHY_REWARD_MODE:-alpha}"
MAX_CPG_QUERIES="${MAX_CPG_QUERIES:-3}"

python -m vulnrl.run_verl \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size=4 \
  data.max_prompt_length=8192 \
  data.max_response_length=8192 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.ppo_clip_ratio=0.2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.agent.default_agent_loop=vulnrl_cwe_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl_agent_loops.yaml \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.project_name=cwe-active-forensics \
  trainer.logger='[console,wandb]' \
  +vulnrl.taxonomy_path="${TAXONOMY_JSON}" \
  +vulnrl.snapshot_dir="${CPG_SNAPSHOT_DIR}" \
  +vulnrl.max_cpg_queries="${MAX_CPG_QUERIES}" \
  +vulnrl.hierarchy_reward_mode="${HIERARCHY_REWARD_MODE}" \
  trainer.total_training_steps="${TRAINING_STEPS:-1000}" \
  trainer.resume_mode=disable
