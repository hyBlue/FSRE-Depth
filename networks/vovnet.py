# Citation ?
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = ["VoVNet"]

_NORM = False

SAVED_STATE_DICT_KEY_PREFIX = 'backbone.bottom_up.'
SAVED_STATE_DICT_KEY_PREFIX_NO_ESE = 'backbone.'

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE': True,
    "dw": False
}

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False
}

VoVNet19_slim_no_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    "eSE": False,
    "dw": False
}

VoVNet39_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw": False
}

VoVNet57_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "eSE": True,
    "dw": False
}

VoVNet99_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
    "V-19-slim": VoVNet19_slim_no_eSE
}


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": nn.SyncBatchNorm,
            "nnSyncBN": nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def dw_conv3x3(in_channels, out_channels, module_name, postfix,
               stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/dw_conv3x3'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=out_channels,
                   bias=False)),
        ('{}_{}/pw_conv1x1'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=1,
                   stride=1,
                   padding=0,
                   groups=1,
                   bias=False)),
        ('{}_{}/pw_norm'.format(module_name, postfix), get_norm(_NORM, out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


def conv3x3(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", get_norm(_NORM, out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        # self.inplace = inplace
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        # return F.relu6(x + 3.0, inplace=self.inplace) / 6.0
        return self.relu6(x + 3.0) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        # Use torch.mean instead of AdaptiveAvgPool2d to let model conversion to tflite works.
        # x = self.avg_pool(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):
    def __init__(
            self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch,
                                    "{}_reduction".format(module_name), "0")))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(
                    nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(
                    nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i)))
                )
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat"))
        )

        if SE:
            self.ese = eSEModule(concat_ch)
        else:
            self.ese = None

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        # Here, self.concat is actually not Concat, but use 1x1 conv to do feature aggregation.
        xt = self.concat(x)

        if self.ese is not None:
            xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(
            self,
            in_ch,
            stage_ch,  # stage conv channel
            concat_ch,  # also the stage out channel
            block_per_stage,
            layer_per_block,
            stage_num, SE=False,
            depthwise=False):

        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise)
        )
        for i in range(block_per_stage - 1):
            # if i != block_per_stage - 2:  # last block
            #     SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True, depthwise=depthwise
                ),
            )


class VoVNetBackbone(nn.Module):
    def __init__(self, input_ch, vovnet_conv_body: str='V-19-slim-eSE', norm='BN'):
        """
        Args:
            input_ch (int): the number of input channel.
            vovnet_conv_body (str): the string to denote the config of vovnet body.
            norm (str): the type of normalization used.
        """
        super(VoVNetBackbone, self).__init__()

        global _NORM
        _NORM = norm

        stage_specs = _STAGE_SPECS[vovnet_conv_body]

        stem_ch = stage_specs["stem"]  # Default: [64, 64, 128]
        config_stage_ch = stage_specs["stage_conv_ch"]  # Default: [64, 80, 96, 112]
        config_concat_ch = stage_specs["stage_out_ch"]  # Default: [112, 256, 384, 512]
        block_per_stage = stage_specs["block_per_stage"]  # Default: [1, 1, 1, 1]
        layer_per_block = stage_specs["layer_per_block"]  # Default: 3
        self.SE = stage_specs["eSE"]

        depthwise = stage_specs["dw"]

        # Needed by backbone interface. The output channels of all stages.
        self.channels = self.get_stage_out_channels(stage_specs)

        # Stem module
        conv_type = dw_conv3x3 if depthwise else conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.add_module("stem", nn.Sequential((OrderedDict(stem))))
        current_stirde = 4
        # Here, comparing to vovnet in original paper, we combine the last layer of stem with the pooling
        # at the beginning of stage 2. Hence, out_feature_stride of stem becomes 4.
        self._out_feature_strides = {"stem": current_stirde, "stage2": current_stirde}
        # TODO(rbli): seems that self._out_feature_channels is not needed. Remove it.
        self._out_feature_channels = {"stem": stem_ch[2]}

        in_ch_list = self.channels[:-1]
        # OSA stages
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    self.SE,
                    depthwise,
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(current_stirde * 2)

        # initialize weights
        self._initialize_weights()
        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(freeze_at=-1)

        # Copied from MobileNetV2Backbone.
        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    @staticmethod
    def get_stage_out_channels(stage_specs):
        return [stage_specs['stem'][-1]] + stage_specs['stage_out_ch']

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return

        raise NotImplementedError

        # for stage_index in range(freeze_at):
        #     if stage_index == 0:
        #         m = self.stem  # stage 0 is the stem
        #     else:
        #         m = getattr(self, "stage" + str(stage_index + 1))
        #     for p in m.parameters():
        #         p.requires_grad = False
        #         FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        outputs.append(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            outputs.append(x)

        return outputs

    # Needed by interface.
    def add_layer(self):
        raise NotImplementedError('VovNet backbone not implemented this method.')

    # Needed by interface.
    def init_backbone(self, path):
        saved_stat_dict = torch.load(path)

        state_dict = OrderedDict()

        if self.SE:
            for key in saved_stat_dict:
                assert key.startswith(SAVED_STATE_DICT_KEY_PREFIX)
                new_key = key[len(SAVED_STATE_DICT_KEY_PREFIX):]
                state_dict[new_key] = saved_stat_dict[key]
        else:
            for key in saved_stat_dict['model']:
                if key.startswith(SAVED_STATE_DICT_KEY_PREFIX_NO_ESE + 'head'):
                    continue
                new_key = key[len(SAVED_STATE_DICT_KEY_PREFIX_NO_ESE):]
                state_dict[new_key] = saved_stat_dict['model'][key]

        self.load_state_dict(state_dict)


class VovNet(nn.Module):
    def __init__(self, num_classes, input_ch, vovnet_conv_body: str='V-19-slim-eSE', norm='BN'):
        super().__init__()
        self.backbone = VoVNetBackbone(input_ch, vovnet_conv_body, norm)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)[-1]
        # print(f'backbone_out.shape {x.size()}')
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = VovNet(1, 3, 'V-19-slim-dw-eSE')
    # checkpoint = "faster_rcnn_V_19_eSE_slim_dw_FPNLite_ms_3x.pth"
    # model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    # print(model)
    layer0 = model.backbone.stem[:7]
    layer1 = nn.Sequential(model.backbone.stem[7:],model.backbone.stage2)
    layer2 = model.backbone.stage3
    layer3 = model.backbone.stage4
    layer4 = model.backbone.stage5
    x = torch.rand(1, 3, 192, 640)
    features = []
    features.append(layer0(x))
    features.append(layer1(features[-1]))
    features.append(layer2(features[-1]))
    features.append(layer3(features[-1]))
    features.append(layer4(features[-1]))
    for feature in features:
        print(feature.shape)
    

