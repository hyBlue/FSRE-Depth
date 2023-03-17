from networks.depth_decoder import DepthDecoder
from networks.roiformer.attention import *
from networks.seg_decoder import SegDecoder
from utils.depth_utils import *
from .attention_utils import *

class RoiFormer(nn.Module):
    def __init__(self, num_ch_enc=None, opt=None):
        super(RoiFormer, self).__init__()

        print("Roi Former")
        #print(num_ch_enc)
        self.scales = opt.scales
        self.opt = opt

        self.num_heads = opt.num_heads
        in_channels_list = opt.fusion_chs

        num_output_channels = 1
        self.num_local_att = 1
        self.num_global_att = 2

        #self.pos_enc = FixedPositionEmbedding(128)

        self.depth_decoder = DepthDecoder(num_ch_enc, num_output_channels=num_output_channels,
                                          scales=opt.scales,
                                          opt=self.opt)
        if opt.semantic_distil:
            self.seg_decoder = SegDecoder(num_ch_enc, num_output_channels=19, scales=[0])


        att_local = {}
        for i in opt.fusion_layers:
            att_local[str(i)] = nn.ModuleList([LocalTransformer(in_channels=in_channels_list[i],
                                                num_head=self.num_heads[i], 
                                                with_sem = opt.semantic_distil, 
                                                batch_norm=True,
                                                num_att_points=opt.num_att_points[i], 
                                                num_att_layers = opt.num_att_layers,
                                                min_a=opt.anchor_min,
                                                max_a=opt.anchor_max,
                                                adaptive_attn=self.opt.adaptive_attn
                                                ) for k in range(self.num_local_att)])
        self.att_local = nn.ModuleDict(att_local)

    def forward(self, input_features):

        depth_outputs = {}
        seg_outputs = {}
        x = input_features[-1]
        x_d = None
        x_s = None
        for i in range(4, -1, -1):
            if x_d is None:
                x_d = self.depth_decoder.decoder[-2 * i + 8](x)               
            else:
                x_d = self.depth_decoder.decoder[-2 * i + 8](x_d)

            if self.opt.semantic_distil:
                if x_s is None:
                    x_s = self.seg_decoder.decoder[-2 * i + 8](x)              
                else:
                    x_s = self.seg_decoder.decoder[-2 * i + 8](x_s)

            x_d = [upsample(x_d)]

            if self.opt.semantic_distil:
                x_s = [upsample(x_s)]

            if i > 0:
                x_d += [input_features[i - 1]]
                if self.opt.semantic_distil:
                    x_s += [input_features[i - 1]]
                    
            x_d = torch.cat(x_d, 1)
            x_d = self.depth_decoder.decoder[-2 * i + 9](x_d)

            if self.opt.semantic_distil:
                x_s = torch.cat(x_s, 1)
                x_s = self.seg_decoder.decoder[-2 * i + 9](x_s)

            #print(x_d.shape, x_s.shape)
            if (i - 1) in self.opt.fusion_layers:
                for layer in self.att_local[str(i - 1)]:
                    if self.opt.semantic_distil:
                        x_d, x_s = layer([x_d, x_s])
                    else:
                        x_d = layer([x_d])

            if self.opt.sgt:
                depth_outputs[('d_feature', i)] = x_d
                seg_outputs[('s_feature', i)] = x_s

            if i in self.scales:
                outs = self.depth_decoder.decoder[10 + i](x_d)
                depth_outputs[("disp", i)] = torch.sigmoid(outs[:, :1, :, :])
                if i == 0 and self.opt.semantic_distil:
                    outs = self.seg_decoder.decoder[10 + i](x_s)
                    seg_outputs[("seg_logits", i)] = outs[:, :19, :, :]

        return depth_outputs, seg_outputs
