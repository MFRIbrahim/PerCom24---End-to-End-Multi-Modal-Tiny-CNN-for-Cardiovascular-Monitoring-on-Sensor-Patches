import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional
from torchvision.ops import Conv2dNormActivation, SqueezeExcitation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    """
    This class is based on the torchvision MobileNetv3 implementation: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
    """
    def __init__(self, c_in, kernel, c_exp, c_out, use_se, activation, stride, dilation, width_mult=1.0) -> None:
        super().__init__()


        self.c_in = self.adjust_channels(c_in, width_mult)
        self.kernel = kernel
        self.c_exp = self.adjust_channels(c_exp, width_mult)
        self.c_out = self.adjust_channels(c_out, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.width_mult = width_mult


        self.use_res_connect = self.stride == 1 and self.c_in == self.c_out

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if self.use_hs else nn.ReLU
        norm_layer = nn.BatchNorm2d

        # expand
        if self.c_exp != self.c_in:
            layers.append(
                Conv2dNormActivation(
                    self.c_in,
                    self.c_exp,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if self.dilation > 1 else self.stride
        layers.append(
            Conv2dNormActivation(
                self.c_exp,
                self.c_exp,
                kernel_size=self.kernel,
                stride=stride,
                dilation=self.dilation,
                groups=self.c_exp,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if self.use_se:
            squeeze_channels = _make_divisible(self.c_exp // 4, 8)
            layers.append(SqueezeExcitation(self.c_exp, squeeze_channels, scale_activation=nn.Hardsigmoid))

        # project
        layers.append(
            Conv2dNormActivation(
                self.c_exp, self.c_out, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = self.c_out
        # self._is_cn = stride > 1


    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        # print(input.shape)
        # print(result.shape)
        # print("====")
        if self.use_res_connect:
            result += input
        return result


class TinyCNN(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        structure = [(16, (1,3), 16, 16, False, "RE", 1, 1, 1.0),
                     (16, (1,3), 72, 24, False, "RE", 1, 1, 1.0),
                     (24, (1,3), 88, 24, False, "RE", 1, 1, 1.0),
                     (24, (1,5), 96, 40, False, "HS", 1, 1, 1.0),]
        
        layers = []

        self.size = conf[4]
        self.modality = conf[5]

        self.shift1 = nn.Parameter(torch.tensor(conf[0]))
        self.scale1 = nn.Parameter(torch.tensor(conf[1]))

        self.shift2 = nn.Parameter(torch.tensor(conf[2]))
        self.scale2 = nn.Parameter(torch.tensor(conf[3]))
        
        if self.size == 6000: 
            layers.append(
                Conv2dNormActivation(
                    1,
                    16,
                    kernel_size=(1,95),
                    stride=(1,94),
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.Hardswish,
                )
            )
        
        if self.size == 256: 
            layers.append(
                Conv2dNormActivation(
                    1,
                    16,
                    kernel_size=(1,5),
                    stride=(1,4),
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.Hardswish,
                )
            )

        if self.size == 128: 
            layers.append(
                Conv2dNormActivation(
                    1,
                    16,
                    kernel_size=(1,3),
                    stride=(1,2),
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.Hardswish,
                )
            )

        if self.size == 64: 
            layers.append(
                Conv2dNormActivation(
                    1,
                    16,
                    kernel_size=(1,3),
                    stride=(1,1),
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.Hardswish,
                )
            )

        for layer in structure:
            layers.append(InvertedResidual(*layer))

        self.backbone = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(40, 2)

    def forward(self, x):
        if self.modality == "affine":
            x[:, :, :, :self.size] += self.shift1
            x[:, :, :, :self.size] *= self.scale1
            x[:, :, :, self.size:] += self.shift2
            x[:, :, :, self.size:] *= self.scale2
        elif self.modality == "ecg":
            x = x[:, :, :, :self.size]
        elif self.modality == "pcg":
            x = x[:, :, :, self.size:]
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
