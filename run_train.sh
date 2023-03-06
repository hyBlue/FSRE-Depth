gpu=$1
config=$2

master_port=29501
# data_path=/home/notebook/data/group/rongbin/datasets/Kitti
python -m torch.distributed.launch --nproc_per_node ${gpu} --master_port ${master_port} train_ddp.py \
    --config=${config}