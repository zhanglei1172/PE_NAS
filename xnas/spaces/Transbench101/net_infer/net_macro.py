import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchsummary import summary

from xnas.spaces.Transbench101.primitives import GenerativeDecoder, Sequential, SequentialJigsaw, Stem, StemJigsaw

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from xnas.spaces.Transbench101.net_infer.cell_micro import ResNetBasicblock, MicroCell
from xnas.spaces.Transbench101.net_ops.cell_ops import ReLUConvBN
# from models.utils import merge_list


class MacroNet(nn.Module):
    """Adapted from torchvision/models/resnet.py"""

    def __init__(self, net_code,C=64, input_dim=(224, 224), dataset=None, in_channels=3, use_small_model=True, n_classes=10):
        super(MacroNet, self).__init__()
        if dataset == "jigsaw":
            self.num_classes = 1000
        elif dataset == "class_object":
            self.num_classes = 100
        elif dataset == "class_scene":
            self.num_classes = 63
        else:
            self.num_classes = n_classes
        self.base_channel = C
        self.in_channels = in_channels
        self.use_small_model = use_small_model
        self._read_net_code(net_code)
        self.feature_dim = [input_dim[0] // 4, input_dim[1] // 4]
        self.stem = self._get_stem_for_task(dataset)

        self.layers = []
        for i, layer_type in enumerate(self.macro_code):
            layer_type = int(layer_type)  # channel change: [2, 4]; stride change: [3, 4]
            target_channel = self.base_channel * 2 if layer_type % 2 == 0 else self.base_channel
            stride = 2 if layer_type > 2 else 1
            self.feature_dim = [self.feature_dim[0] // stride, self.feature_dim[1] // stride]
            layer = self._make_layer(self.cell, target_channel, 2, stride, True, True)
            self.add_module(f"layer{i}", layer)
            self.layers.append(f"layer{i}")

        self.head = self._get_decoder_for_task(dataset, self.base_channel)

    def weights(self):
        return self.parameters()
    def forward(self, x):
        x = self.stem(x)

        for i, layer_name in enumerate(self.layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

        x = self.head(x)
        return x

    def _make_layer(self, cell, planes, num_blocks, stride=1, affine=True, track_running_stats=True):
        layers = [cell(self.micro_code, self.base_channel, planes, stride, affine, track_running_stats)]
        self.base_channel = planes * cell.expansion
        for _ in range(1, num_blocks):
            layers.append(cell(self.micro_code, self.base_channel, planes, 1, affine, track_running_stats))
        return nn.Sequential(*layers)

    def _read_net_code(self, net_code):
        net_code_list = net_code.split('-')
        self.base_channel = int(net_code_list[0])
        self.macro_code = net_code_list[1]
        if net_code_list[-1] == 'basic':
            self.micro_code = 'basic'
            self.cell = ResNetBasicblock
        else:
            self.micro_code = [''] + net_code_list[2].split('_')
            self.cell = MicroCell

    def get_hash(self,):
        return tuple(map(int,self.macro_code.split('')))

    def forward_before_global_avg_pool(self, x):
        outputs = []
        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)

        self.forward(x)

        assert len(outputs) == 1
        return outputs[0]

    def _get_decoder_for_task(self, task, n_channels): #TODO: Remove harcoding
        if task == "jigsaw":
            return  SequentialJigsaw(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(n_channels * 9, self.num_classes)
                    )
        elif task in ["class_object", "class_scene"]:
            return Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(n_channels, self.num_classes)
                    )
        elif task == "autoencoder":
            if self.use_small_model:
                return GenerativeDecoder((64, 32), (256, 2048)) # Short
            else:
                return GenerativeDecoder((512, 32), (512, 2048)) # Full TNB

        else:
            return Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(n_channels, self.num_classes)
                    )
        # self._kaiming_init()
    def _get_stem_for_task(self, task):
        if task == "jigsaw":
            return StemJigsaw(C_out=self.base_channel)
        elif task in ["class_object", "class_scene"]:
            return Stem(C_out=self.base_channel)
        elif task == "autoencoder":
            return Stem(C_out=self.base_channel)
        else:
            return Stem(C_in=self.in_channels, C_out=self.base_channel)


if __name__ == "__main__":
    # net = MacroNet("64-41414-3_33_333", structure='backbone').cuda()
    net = MacroNet("64-41414-basic", structure='backbone').cuda()
    print(net)
    summary(net, (3, 256, 256))
