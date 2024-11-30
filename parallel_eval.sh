#!/bin/bash

## NECESSARY JOB SPECIFICATIONS
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --output=./job_output_%x.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
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

ita_type="isogclr_new_v2"
declare -a optimizers=("adamw" "nadam" "nvnovograd")

for gpu_id in {0..2}; do
    optimizer=${optimizers[gpu_id]}
    checkpoint_path="./output/${ita_type}_${optimizer}_${data}_g${gamma}_e${epochs}/checkpoint_30.pth"
    output_dir="output/eval_${ita_type}_${optimizer}_${data}_g${gamma}_e${epochs}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python ./bimodal_exps/clip.py \
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
        --no-distributed \
        --epochs ${epochs} \
        --evaluate \
        --checkpoint ${checkpoint_path} \
        --zs_dataset imagenet \
        --zs_datafolder ./datasets/imagenet/val > "logs/eval_${ita_type}_${optimizer}.log" 2>&1 &
done

wait
