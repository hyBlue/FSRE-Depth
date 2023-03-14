import sys

sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from networks.vovnet import VovNet

class VovNetMultiImageInput(nn.Module):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, num_input_images=1):
        super(VovNetMultiImageInput, self).__init__()
        
        self.backbone = VovNet(1, 3, "V-19-slim-eSE").backbone

        self.backbone.stem[0] = nn.Conv2d(3 * num_input_images, 64, kernel_size=3, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def vovnet_multiimage_input(pretrained=False, num_input_images=1):
    """Constructs a Mbnetv2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """

    model = VovNetMultiImageInput(num_input_images)

    if pretrained:
        checkpoint = "vovnet19_ese_slim_detectron2.pth"
        loads = torch.load(checkpoint)
        updated_loads = {}
        for (k, v) in loads.items():
            new_key = k.replace('.bottom_up', '')
            updated_loads[new_key] = v
        
        updated_loads['backbone.stem.stem_1/conv.weight'] = torch.cat(
            [updated_loads['backbone.stem.stem_1/conv.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(updated_loads, strict=False)

    return model

class VovNetEncoder(nn.Module):
    def __init__(self, num_layers=18, pretrained=True, num_input_images=1):
        super(VovNetEncoder, self).__init__()

        if pretrained:
            print('load pretrained from imagenet, vovnet')
        else:
            print('train starts from the scratch')

        self.num_ch_enc = np.array([64, 112, 256, 384, 512])
        if num_input_images > 1:
            encoder = vovnet_multiimage_input(pretrained, num_input_images)  
        else:
            encoder = VovNet(1, 3, "V-19-slim-eSE")
        
            if pretrained:
                checkpoint = "vovnet19_ese_slim_detectron2.pth"
                loads = torch.load(checkpoint)
                updated_loads = {}
                for (k, v) in loads.items():
                    new_key = k.replace('.bottom_up', '')
                    updated_loads[new_key] = v
                encoder.load_state_dict(updated_loads, strict=False)
            else:
                self.init_weight()

        # Choose layers such that the feature maps' size are same with the resnet encoder
        self.layer0 = encoder.backbone.stem[:6]
        self.layer1 = nn.Sequential(encoder.backbone.stem[6:], encoder.backbone.stage2)
        self.layer2 = encoder.backbone.stage3
        self.layer3 = encoder.backbone.stage4
        self.layer4 = encoder.backbone.stage5

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225
        features.append(self.layer0(x))
        features.append(self.layer1(features[-1]))
        features.append(self.layer2(features[-1]))
        features.append(self.layer3(features[-1]))
        features.append(self.layer4(features[-1]))

        return features

if __name__ == "__main__":
    x = torch.rand(1, 3, 192, 640)
    model = VovNetEncoder(num_layers=50, pretrained=True, num_input_images=1)
    features = model(x)

    for feature in features:
        print(feature.shape)
    # print(model) 

