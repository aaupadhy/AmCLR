#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output_%x.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu

#First Executable Line
env_name=AmCLR
source ~/.bashrc
conda activate ${env_name}

export PYTHONPATH="$PYTHONPATH:./src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=1
ita_type=sogclr

output_dir="output/${ita_type}/${ita_type}_${optimizer}_${data}_g${gamma}_e${epochs}"
log_dir="logs/${ita_type}"
log_file="${log_dir}/${ita_type}_${optimizer}_training.log"
mkdir -p "${output_dir}"
mkdir -p "${log_dir}"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4820   \
    --use-env ./src/clip.py \
    --data_path ${data_path} \
    --ann_path ${ann_path} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --output_dir ${output_dir} \
    --init_model \
    --use_amp \
    --ita_type ${ita_type} \
    --tau_init 0.01 \
    --sogclr_gamma ${gamma} \
    --eta_init 0.03 --sched cosine \
    --distributed \
    --epochs ${epochs} 