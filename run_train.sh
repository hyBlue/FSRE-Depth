gpu=$1
config=$2

log_dir=/home/notebook/data/group/rongbin/proj_misc/Github/FSRE-Depth/results

master_port=29502
# data_path=/home/notebook/data/group/rongbin/datasets/Kitti
python -m torch.distributed.launch --nproc_per_node ${gpu} --master_port ${master_port} train_ddp.py \
    --config=${config} \
    --log_dir=${log_dir}