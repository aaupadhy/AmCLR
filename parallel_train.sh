#!/bin/bash

## NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output_%x.%j
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=3
#SBATCH --partition=gpu

# First Executable Line
source ~/.bashrc
conda activate DL_Project

export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=30

mkdir -p logs

run_training() {
    local ita_type=$1
    local gpu_id=$2
    local optimizer=$3
    local port=$((4820 + gpu_id))
    local log_file="logs/${ita_type}_${optimizer}_training.log" 

    CUDA_VISIBLE_DEVICES=${gpu_id} python -m torch.distributed.launch --nproc_per_node=1 --master_port=${port} \
        --use-env ./bimodal_exps/clip.py \
        --data_path ${data_path} \
        --ann_path ${ann_path} \
        --train_file ${train_file} \
        --train_image_root ${train_image_root} \
        --output_dir output/${ita_type}_${optimizer}_${data}_g${gamma}_e${epochs} \
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

# run_training sogclr 0 adamw

# run_training sogclraug 1 adamw

run_training isogclr_new_v2 0 adamw
run_training isogclr_new_v2 1 nadam
run_training isogclr_new_v2 2 nvnovograd
wait
