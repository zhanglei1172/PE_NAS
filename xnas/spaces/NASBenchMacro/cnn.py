import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

from xnas.core.query_metrics import Metric

candidate_OP = ['id', 'ir_3x3_t3', 'ir_5x5_t6']
OPS = OrderedDict()
OPS['id'] = lambda inp, oup, stride: Identity(inp=inp, oup=oup, stride=stride)
OPS['ir_3x3_t3'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=3, stride=stride, k=3)
OPS['ir_5x5_t6'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=6, stride=stride, k=5)


class Identity(nn.Module):
    def __init__(self, inp, oup, stride, supernet=True):
        super(Identity, self).__init__()
        if supernet:
            self.affine = False
        else:
            self.affine = True
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup, affine=self.affine),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, supernet=True, use_se=False, **kwargs):
        super(InvertedResidual, self).__init__()
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        hidden_dim = round(inp * t)
        if t == 1:
            self.feature = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim, affine=self.affine),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, affine=self.affine)
            )
        else:
            self.feature = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim, affine=self.affine),
                activation(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim, affine=self.affine),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, affine=self.affine),
            )
        self.use_shortcut = inp == oup and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.feature(x) + x
        return self.feature(x)


class NBM(nn.Module):
    def __init__(self, num_classes=10, stages=[2, 3, 3], init_channels=32, supernet=True):
        super(NBM, self).__init__()
        self.supernet = supernet
        self.stages = stages
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        channels = init_channels
        self.choice_block = nn.ModuleList([])
        for stage in stages:
            for idx in range(stage):
                layer_cb = nn.ModuleList([])
                for i in candidate_OP:
                    if idx == 0:
                        layer_cb.append(OPS[i](channels, channels * 2, 2))
                    else:
                        layer_cb.append(OPS[i](channels, channels, 1))
                if idx == 0:
                    channels *= 2
                self.choice_block.append(layer_cb)

        self.out = nn.Sequential(
            nn.Conv2d(channels, 1280, kernel_size=1, bias=False, stride=1),
            nn.BatchNorm2d(1280, affine=self.affine),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(1280, num_classes)

    def weights(self):
        return self.parameters()

    def forward(self, x, choice):
        # if self.supernet == True:
        #     choice = np.random.randint(3, size=8)
        x = self.stem(x)
        for i, j in enumerate(choice):
            x = self.choice_block[i][j](x)
        x = self.out(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out
    def forward_with_features(self,x, choice):
        x = self.stem(x)
        cur_stages = 1
        out_features = []
        for i, j in enumerate(choice):
            x = self.choice_block[i][j](x)
            if cur_stages in self.stages:
                out_features.append(x)
            cur_stages += 1
        x = self.out(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out_features, out
    
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

    def get_type(self,):
        return "nasbenchmacro"

    def sample_random_architecture(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        xnas object accordingly.
        """
        op_indices = np.random.randint(3, size=(8))
        return tuple(op_indices)
    
    def get_all_architecture(self,dataset_api=None):
        import itertools
        return list(itertools.product(*[range(3) for i in range(8)]))

    def query(
        self,
        arch_hash,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results from nasbench 201
        """
        assert isinstance(metric, Metric)
        arch_str = ''.join(map(str,arch_hash))
        if metric == Metric.ALL:
            raise NotImplementedError()
        if metric != Metric.RAW and metric != Metric.ALL:
            assert dataset in [
                "cifar10",
                # "cifar100",
                # "ImageNet16-120",
            ], "Unknown dataset: {}".format(dataset)
        if dataset_api is None:
            raise NotImplementedError("Must pass in dataset_api to query nasbenchmacro")
        

        dataset_api = dataset_api[arch_str]
        if metric == Metric.VAL_ACCURACY:
            val_acc = dataset_api['mean_acc']
            return val_acc
        elif metric == Metric.FLOPS:
            flops = dataset_api['flops']
            return flops
        elif metric == Metric.PARAMETERS:
            params = dataset_api['params']
            return params
        else:
            return -1


class Infer_NBM(nn.Module):
    def __init__(self, arch_hash, num_classes=10, stages=[2, 3, 3], init_channels=32):
        super(Infer_NBM, self).__init__()
        self.stages = stages
        self.affine = True
        self.arch_hash = arch_hash
        _arch = list(arch_hash)
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        channels = init_channels
        self.choice_block = nn.ModuleList([])
        for stage in stages:
            layer_cb = nn.ModuleList([])
            for idx in range(stage):
                i = candidate_OP[_arch.pop(0)]
                if idx == 0:
                    layer_cb.append(OPS[i](channels, channels * 2, 2))
                    channels *= 2
                else:
                    layer_cb.append(OPS[i](channels, channels, 1))
            self.choice_block.append(layer_cb)

        self.out = nn.Sequential(
            nn.Conv2d(channels, 1280, kernel_size=1, bias=False, stride=1),
            nn.BatchNorm2d(1280, affine=self.affine),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(1280, num_classes)

    def weights(self):
        return self.parameters()

    def forward(self, x):
        # if self.supernet == True:
        #     choice = np.random.randint(3, size=8)
        x = self.stem(x)
        for i in range(len(self.choice_block)):
            for j in range(len(self.choice_block[i])):
                x = self.choice_block[i][j](x)
        x = self.out(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out
    
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

    def get_type(self,):
        return "nasbenchmacro"
    
    def get_hash(self,):
        return self.arch_hash

    def sample_random_architecture(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        xnas object accordingly.
        """
        op_indices = np.random.randint(3, size=(8))
        return tuple(op_indices)

    def get_all_architecture(self,dataset_api=None):
        import itertools
        return list(itertools.product(*[range(3) for i in range(8)]))

    # def _initialize_weights(self):
    #     for name, m in self.named_modules():
    #         if isinstance(m, nn.Conv2d):
    #             if 'first' in name:
    #                 nn.init.normal_(m.weight, 0, 0.01)
    #             else:
    #                 nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             if m.weight is not None:
    #                 nn.init.constant_(m.weight, 1)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.0001)
    #             nn.init.constant_(m.running_mean, 0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.0001)
    #             nn.init.constant_(m.running_mean, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)



def _NBMacro_sup_train():
    return NBM(
        supernet=True
    )

def _NBMacro_child_train():
    return NBM(
        supernet=False
    )
