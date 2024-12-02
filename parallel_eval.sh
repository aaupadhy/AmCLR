#!/bin/bash
## NECESSARY JOB SPECIFICATIONS
#SBATCH --time=2:00:00
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

# Constants
data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
epochs=30

declare -A models
models=(
    ["sogclr"]="sgdp"
)

# Iterate over the models
for model in "${!models[@]}"; do
    optimizer=${models[$model]}
    gamma=0.8 # Adjust gamma if necessary
    checkpoint_path="./output/${model}/${model}_${optimizer}_${data}_g${gamma}_e${epochs}/checkpoint_30.pth"
    output_dir="output/eval/eval_${model}_${optimizer}_${data}_g${gamma}_e${epochs}"

    echo "Evaluating model: ${model}, optimizer: ${optimizer}"

    CUDA_VISIBLE_DEVICES=7 python ./bimodal_exps/clip.py \
        --data_path ${data_path} \
        --ann_path ${ann_path} \
        --train_file ${train_file} \
        --train_image_root ${train_image_root} \
        --output_dir ${output_dir} \
        --init_model \
        --use_amp \
        --ita_type ${model} \
        --tau_init 0.01 \
        --sogclr_gamma ${gamma} \
        --eta_init 0.03 --sched cosine \
        --no-distributed \
        --epochs ${epochs} \
        --evaluate \
        --checkpoint ${checkpoint_path} \
        --zs_dataset imagenet \
        --zs_datafolder ./datasets/imagenet/val > "logs/eval_logs/eval_${model}_${optimizer}.log" 2>&1 &
done

wait