kitti_data_path=/home/notebook/data/group/rongbin/datasets/Kitti
proj_misc=/home/notebook/data/group/rongbin/proj_misc/Github/FSRE-Depth

tag=03-16-2023
config=Roiformer_res50_192
which_weight=weights_19_best
# model_dir=/home/notebook/data/group/rongbin/proj_misc/Github/FSRE-Depth/results/03-06-2023/Roiformer_res18_192/models/weights_18_best
model_dir=${proj_misc}/results/${tag}/${config}/models/${which_weight}

python evaluate_depth.py --load_weights_folder=${model_dir} --config=${config}