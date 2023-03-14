import sys

sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class Mbnetv2MultiImageInput(nn.Module):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, num_input_images=1):
        super(Mbnetv2MultiImageInput, self).__init__()
        
        self.features = models.mobilenet_v2().features

        self.features[0][0] = nn.Conv2d(num_input_images * 3, 32, kernel_size=3, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def mbnetv2_multiimage_input(pretrained=False, num_input_images=1):
    """Constructs a Mbnetv2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """

    model = Mbnetv2MultiImageInput(num_input_images)

    if pretrained:
        loaded = models.mobilenet_v2("DEFAULT").state_dict() # weights = "DEFAULT"
        
        loaded['features.0.0.weight'] = torch.cat(
            [loaded['features.0.0.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded, strict=False)

    return model

class Mbnetv2Encoder(nn.Module):
    def __init__(self, num_layers=18, pretrained=True, num_input_images=1):
        super(Mbnetv2Encoder, self).__init__()

        if pretrained:
            print('load pretrained from imagenet, mbnetv2')
        else:
            print('train starts from the scratch')

        self.num_ch_enc = np.array([32, 24, 32, 64, 160])
        if num_input_images > 1:
            encoder = mbnetv2_multiimage_input(pretrained, num_input_images)
        else:
            if pretrained:
                encoder = models.mobilenet_v2("DEFAULT") # weights="DEFAULT"
                # encoder = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
            else:
                encoder = models.mobilenet_v2()
        if not pretrained:
            self.init_weight()

        # Choose layers such that the feature maps' size are same with the resnet encoder
        self.layer0 = encoder.features[0]
        self.layer1 = encoder.features[1:4]
        self.layer2 = encoder.features[4:7]
        self.layer3 = encoder.features[7:11]
        self.layer4 = encoder.features[11:16]

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
    x = torch.rand(1, 6, 192, 640)
    model = Mbnetv2Encoder(num_layers=50, pretrained=True, num_input_images=2)
    features = model(x)

    for feature in features:
        print(feature.shape)
