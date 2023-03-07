model_dir=/home/notebook/data/group/rongbin/proj_misc/Github/FSRE-Depth/results/03-06-2023/Roiformer_res18_192/models/weights_18_best
kitti_data_path=/home/notebook/data/group/rongbin/datasets/Kitti

python evaluate_depth.py --load_weights_folder=${model_dir} --config=Roiformer_res18_192