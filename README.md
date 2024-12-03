# AmCLR & XAmCLR PyTorch Implementation based on SogCLR

In this repo, we show how to train a self-supervised model by using Global Contrastive Loss (GCL) on a widely used bimodal image-text dataset [CC3M](https://ai.google.com/research/ConceptualCaptions/download). Initial experimentations are run on CC3M_mini (100k subset).

### Environment

Setting up a new virtual environment with Conda:
````bash
env_name='AmCLR'
# We have used env_name as "DL_Project"
conda create -n "$env_name" python=3.10
conda activate "$env_name"
pip install -r requirements.txt
````

### Training and Evaluation

1. Download the data: [cc3m_subset_100k.tar.gz](https://drive.google.com/file/d/142zQjlOw0Xw4tKzXMrQjYE6NtGRTeasT/view?usp=drive_link), a 100k subset of the [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) dataset; [mscoco_val.tar.gz](https://drive.google.com/file/d/142tMsnclHTTPpnTXHSeNgTUlBk4She6o/view?usp=drive_link), a 5k subset of the [COCO](https://cocodataset.org/#home) val2014 dataset; [clip_train.tar.gz](https://drive.google.com/file/d/142xxRoMaHxX3BIfCw_1b_G_dgu-02Yq3/view?usp=drive_link), captions of the previous datasets; [imagenet/val.tar](https://drive.google.com/file/d/1NXhfhwFy-nhdABACkodgYqm9pomDKE39/view?usp=sharing), [ImageNet](https://www.image-net.org/challenges/LSVRC/index.php) validation set. The code and data should be structured as follows:
    ```
    .
    +--bimodal_exps (code)
    |
    +--clip_train (captions)
    |  +--cc3m_train_subset.json
    |  +--coco_val.json
    |
    +--datasets (images)
    |  +--cc3m_subset_100k
    |  +--mscoco_val
    |  +--imagnet
    |  |  +-- val
    ```
2. To train a model on cc3m, use `parallel_train.sh`, below is a sample for one type of experiment:
    ```bash
    # Export environment variables
    export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
    export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'
    export TORCH_DISTRIBUTED_DEBUG=DETAIL  

    # Paths and Configurations
    data_path=./datasets
    ann_path=./clip_train
    train_image_root=cc3m_subset_100k/
    data=cc3m
    train_file=${data}_train_subset.json
    gamma=0.8
    epochs=30

    # Ensure necessary directories exist
    mkdir -p logs

    # Function to run training
    run_training() {
        local ita_type=$1
        local gpu_id=$2
        local optimizer=$3
        local port=$((4820 + gpu_id))
        local output_dir="output/${ita_type}/${ita_type}_${optimizer}_${data}_g${gamma}_e${epochs}"
        local log_dir="logs/${ita_type}"
        local log_file="${log_dir}/${ita_type}_${optimizer}_training.log"

        # Ensure output and log directories exist
        mkdir -p "${output_dir}"
        mkdir -p "${log_dir}"

        # Launch training
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

    # Call run_training with different configurations
    run_training sogclraug_linear 3 adamp
    run_training sogclraug_wSelf_linear 4 adamp
    run_training sogclr 5 adamp

    # Wait for all processes to finish
    wait

    ```
3. To test the performance of a model on MSCOCO and ImageNet, use `parallel_eval.sh`, below is a sample for the same:
    ```bash
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
        ["sogclraug_linear"]="adamp"
        ["sogclraug_wSelf_linear"]="adamp"
        ["sogclr"]="adamp"
        
    )
    # Iterate over the models
    for model in "${!models[@]}"; do
        optimizer=${models[$model]}
        gamma=0.8 # Adjust gamma if necessary
        checkpoint_path="./output/${model}/${model}_${optimizer}_${data}_g${gamma}_e${epochs}/checkpoint_30.pth"
        output_dir="output/eval/eval_${model}_${optimizer}_${data}_g${gamma}_e${epochs}"

        echo "Evaluating model: ${model}, optimizer: ${optimizer}"

        CUDA_VISIBLE_DEVICES=4 python ./bimodal_exps/clip.py \
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
    ```

## Reference
If you find this tutorial helpful, please cite:
```
@inproceedings{qiu2023not,
  title={Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization},
  author={Zi-Hao Qiu, Quanqi Hu, Zhuoning Yuan, Denny Zhou, Lijun Zhang, and Tianbao Yang},
  booktitle={International Conference on Machine Learning},
  pages={TBD},
  year={2023},
  organization={PMLR}
}
```
