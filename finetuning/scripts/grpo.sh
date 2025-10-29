export OUTPUT_PATH="work_dirs/grpo"
export EXP_NAME="grpo"
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT_PATH}/log.txt"


# make sure the output directory exists
mkdir -p ${OUTPUT_PATH}

set -x

export PYTHONUNBUFFERED=1

# Disable Ray cluster discovery to force local cluster
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_ADDRESS=""
export RAY_CLIENT_MODE=""

# Force VLLM to use local Ray cluster
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Clear any Ray temp files
rm -rf /tmp/ray/* 2>/dev/null || true

MODEL_PATH="IDEA-Research/Rex-Omni"  # replace it with your local file path

python3 -m verl.trainer.main \
    config=verl/configs/config.yaml \
    data.config_path="configs/grpo.py" \
    data.format_prompt=verl/configs/r1v_format.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    worker.actor.global_batch_size=64 \
    data.rollout_batch_size=64 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.n=8 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.save_checkpoint_path=${OUTPUT_PATH} \
    trainer.save_freq=100 \
