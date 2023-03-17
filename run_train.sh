proj_misc=/home/notebook/data/group/rongbin/proj_misc/Github/FSRE-Depth

if [[ $1 == "test" ]]; then
    echo "In test mode, arguments is hard-coded in the script ..."
    gpu=1
    config=Roiformer_res50_192
    log_dir=${proj_misc}/test_results
elif [[ $1 == "prod" ]]; then
    echo "In prod mode ..."
    gpu=$2
    config=$3
    log_dir=${proj_misc}/results
else
    echo "Mode should be either test or prod"
    exit 1
fi

master_port=29502
# data_path=/home/notebook/data/group/rongbin/datasets/Kitti
python -m torch.distributed.launch --nproc_per_node ${gpu} --master_port ${master_port} train_ddp.py \
    --config=${config} \
    --log_dir=${log_dir}