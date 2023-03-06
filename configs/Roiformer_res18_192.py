from easydict import EasyDict as edict


cfg = edict()

# MODEL
cfg.batch_size = 4
cfg.height = 192
cfg.width = 640
cfg.max_depth = 100.0
cfg.num_layers = 18

#cfg.base_seed = 42

cfg.data_path = './data/Kitti'

# Optimization 
cfg.learning_rate = 1.5e-4  
cfg.num_epochs = 20
cfg.lr_decay = [10,15]
cfg.decay_rate = 0.1 

   
# Semantic loss 
cfg.sgt = 0.1
# cfg.sbl = 0.0
 
# Special arguments for RoiFormer fusion model 
cfg.fusion_type = 'roiformer'  
cfg.num_heads = [2,2,4,8,2] 
cfg.fusion_layers = [3,2,1]
cfg.fusion_chs = [32, 64, 128, 256, 256]  
cfg.num_att_points = [16,32,16,8,16]
cfg.num_att_layers = 2 
cfg.anchor_max = 0.7
cfg.anchor_min = 0.3
 
# Depth and Semantic decoder
# cfg.num_ch_dec = [16, 32, 64, 128, 256]  