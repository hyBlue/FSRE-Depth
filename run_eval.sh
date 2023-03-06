model_dir=tmp/02-24-2023/full_res18_192x640/models/weights_19_best
kitti_data_path=/home/notebook/data/group/rongbin/datasets/Kitti

python evaluate_depth.py --load_weights_folder ${model_dir} --data_path ${kitti_data_path}