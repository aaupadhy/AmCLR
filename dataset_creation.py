#!git clone https://github.com/aaupadhy/iSogCLR.git

# !export PYTHONPATH="$PYTHONPATH:./iSogCLR/bimodal_exps"
# !export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'
# !mkdir checkpoints

# !gdown 142xxRoMaHxX3BIfCw_1b_G_dgu-02Yq3    # clip_train.tar.gz
# !gdown 142zQjlOw0Xw4tKzXMrQjYE6NtGRTeasT    # cc3m_subset_100k.tar.gz
# !gdown 142tMsnclHTTPpnTXHSeNgTUlBk4She6o    # ms_coco_val.tar.gz
# !gdown 1NXhfhwFy-nhdABACkodgYqm9pomDKE39    # val.tar

# !mkdir datasets
# !mkdir -p datasets/imagenet
# !tar xf clip_train.tar.gz
# !tar xf cc3m_subset_100k.tar.gz -C datasets
# !tar xf mscoco_val.tar.gz -C datasets
# !tar xf val.tar -C datasets/imagenet

# !pip install -r requirements.txt    # there may be pip warnings/ errors, should be fine to ignore them