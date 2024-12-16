#!/bin/bash

## NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00          # Max run time
#SBATCH --mem=40G                # Memory allocation
#SBATCH --output=./job_output_%x.%j  # Standard output log
#SBATCH --ntasks=1           # Number of tasks
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=8        # CPUs per task
#SBATCH --gpus=1                # Number of GPUs
#SBATCH --partition=gpu          # GPU partition

# Activate Environment
source ~/.bashrc
conda activate ML

export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'
export TORCH_DISTRIBUTED_DEBUG=DETAIL

data_path=./mock_datasets
ann_path=./mock_clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=1

mkdir -p logs

run_training() {
    local ita_type=$1
    local gpu_id=$2
    local optimizer=$3
    local port=$((4820 + gpu_id))
    local output_dir="output/${ita_type}/${ita_type}_${optimizer}_${data}_g${gamma}_e${epochs}"
    local log_dir="logs/${ita_type}"
    local log_file="${log_dir}/${ita_type}_${optimizer}_training.log"

    mkdir -p "${output_dir}"
    mkdir -p "${log_dir}"

    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=${port} ./bimodal_exps/clip.py \
        --data_path ${data_path} \
        --ann_path ${ann_path} \
        --train_file ${train_file} \
        --train_image_root ${train_image_root} \
        --output_dir ${output_dir} \
        --init_model \
        --use_amp \
        --ita_type ${ita_type} \
        --tau_init 0.01 \
        --opt ${optimizer} \
        --sogclr_gamma ${gamma} \
        --eta_init 0.03 --sched cosine \
        --distributed \
        --epochs ${epochs} > "${log_file}" 2>&1 &
}

run_training sogclraug_linear 0 adamp

wait
