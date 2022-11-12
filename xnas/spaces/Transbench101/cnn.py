import torch
import torch.nn as nn
import numpy as np
from xnas.core.query_metrics import Metric

from xnas.spaces.Transbench101.net_infer.net_macro import MacroNet
from xnas.spaces.Transbench101.net_ops.cell_ops import OPS, ReLUConvBN
from xnas.spaces.Transbench101.primitives import GenerativeDecoder, Sequential, SequentialJigsaw, Stem, StemJigsaw

PRIMITIVES = ['0' ,'1', '2', '3']
class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, affine, track_running_stats)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in filter(lambda x:x[0]!=0, zip(weights, self._ops)))

class MultiSearchCell(nn.Module):
    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, inputs, weights): 
        for layer in self.layers:
            inputs = layer(inputs, weights)
        return inputs

class SearchMicroCell(nn.Module):
    expansion = 1

    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        """
        initialize a cell
        Args:
            cell_code: ['', '1', '13', '302']
            C_in: in channel
            C_out: out channel
            stride: 1 or 2
        """
        super(SearchMicroCell, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)

        self.node_num = 4
        self.edges = nn.ModuleList()
        self.nodes = list(range(self.node_num))  # e.g. [0, 1, 2, 3]
        # assert self.nodes == list(map(len, cell_code))
        self.from_nodes = [list(range(i)) for i in self.nodes]  # e.g. [[], [0], [0, 1], [0, 1, 2]]
        self.from_ops = [list(range(n * (n - 1) // 2, n * (n - 1) // 2 + n))
                         for n in range(self.node_num)]  # e.g. [[], [0], [1, 2], [3, 4, 5]]
        self.stride = stride

        for node in self.nodes:
            for from_node in self.from_nodes[node]:
                if from_node == 0:
                    edge = MixedOp(C_in, C_out, self.stride, affine, track_running_stats)
                else:
                    edge = MixedOp(C_out, C_out, 1, affine, track_running_stats)
                self.edges.append(edge)

        # self.cell_code = cell_code
        self.C_in = C_in
        self.C_out = C_out
    def weights(self):
        return self.parameters()
    def forward(self, inputs, weights): # weights: (6, 4)
        node_features = [inputs]
        # compute the out features for each nodes
        for node_idx in self.nodes:
            if node_idx == 0:
                continue
            node_feature_list = [self.edges[from_op](node_features[from_node], weights[from_op]) for from_op, from_node in
                                 zip(self.from_ops[node_idx], self.from_nodes[node_idx])]
            # for i, nf in enumerate(node_feature_list):
            #     print(node_idx, self.from_nodes[node_idx][i], self.cell_code[node_idx][i], nf.shape)
            node_feature = torch.stack(node_feature_list).sum(0)
            # print(f"node_idx {node_idx} output:{node_feature.shape}\n")
            node_features.append(node_feature)
        return node_features[-1]

class SPOS_Trans101_micro(nn.Module):
    def __init__(self, dataset=None, C=64, input_dim=(224,224),in_channels=3,use_small_model=True, n_classes=10) -> None:
        super().__init__()
        self.macro_code = '41414'
        self.base_channel = C
        self.in_channels = in_channels
        self.use_small_model = use_small_model
        # self.n_modules = 5 # short: 3
        self.feature_dim = [input_dim[0] // 4, input_dim[1] // 4]
        self.all_edges = 6
        self.spos_all_edge_num = 5*self.all_edges # 5 layers 6 edges
        if dataset == "jigsaw":
            self.num_classes = 1000
        elif dataset == "class_object":
            self.num_classes = 100
        elif dataset == "class_scene":
            self.num_classes = 63
        else:
            self.num_classes = n_classes
        self.stem = self._get_stem_for_task(dataset)

        self.layers = []
        for i, layer_type in enumerate(self.macro_code):
            layer_type = int(layer_type)  # channel change: {2, 4}; stride change: {3, 4}
            target_channel = self.base_channel * 2 if layer_type % 2 == 0 else self.base_channel
            stride = 2 if layer_type > 2 else 1
            self.feature_dim = [self.feature_dim[0] // stride, self.feature_dim[1] // stride]
            layer = self._make_layer(SearchMicroCell, target_channel, 2, stride, True, True)
            self.add_module(f"layer{i}", layer)
            
            self.layers.append(f"layer{i}")
        # self.global_pooling = nn.AdaptiveAvgPool2d((1, 1)) if structure in ['drop_last', 'full'] else None
        # self.classifier = nn.Linear(self.base_channel, self.num_classes) if structure in ['full'] else None

        self.head = self._get_decoder_for_task(dataset, self.base_channel)

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
        
    def forward(self, x, choice): # choice: [30]
        x = self.stem(x)
        # repeat
        _it = 0
        if len(choice) < self.spos_all_edge_num:
            choice = np.tile(choice, self.spos_all_edge_num//len(choice))
            
        weights = np.eye(len(PRIMITIVES))[choice]
        for i, layer_name in enumerate(self.layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x, weights[_it:_it+self.all_edges])
            _it += self.all_edges
        assert _it == self.spos_all_edge_num
        # x = self.last_conv(x)
        x = self.head(x)
        return x
    
    def weights(self):
        return self.parameters()
    
    def _make_layer(self, cell, planes, num_blocks, stride=1, affine=True, track_running_stats=True):
        layers = [cell(self.base_channel, planes, stride, affine, track_running_stats)]
        self.base_channel = planes * cell.expansion
        for _ in range(1, num_blocks):
            layers.append(cell(self.base_channel, planes, 1, affine, track_running_stats))
        return MultiSearchCell(*layers)

    def _get_stem_for_task(self, task):
        if task == "jigsaw":
            return StemJigsaw(C_out=self.base_channel)
        elif task in ["class_object", "class_scene"]:
            return Stem(C_out=self.base_channel)
        elif task == "autoencoder":
            return Stem(C_out=self.base_channel)
        else:
            return Stem(C_in=self.in_channels, C_out=self.base_channel)

    def sample_random_architecture(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        xnas object accordingly.
        """
        op_indices = np.random.randint(4, size=(6))
        return tuple(op_indices)
        # self.set_op_indices(op_indices)

    def get_all_architecture(self,dataset_api=None):
        import itertools
        return list(itertools.product(*[range(4) for i in range(6)]))

    def get_type(self,):
        return "transbench101_micro"

    def query(self, arch_hash, metric=None, dataset=None, path=None, epoch=-1, full_lc=False, dataset_api=None):
        """
        Query results from transbench 101
        """
        assert isinstance(metric, Metric)
        if metric == Metric.ALL:
            raise NotImplementedError()
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query transbench101')
        
        arch_str = '64-41414-{}_{}{}_{}{}{}'.format(*arch_hash)
          
        query_results = dataset_api['api']
        task = dataset_api['task']
                
        
        if task in ['class_scene', 'class_object', 'jigsaw']:
            
            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_top1',
                Metric.VAL_ACCURACY: 'valid_top1',
                Metric.TEST_ACCURACY: 'test_top1',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }

        elif task == 'room_layout':
            
            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_neg_loss',
                Metric.VAL_ACCURACY: 'valid_neg_loss',
                Metric.TEST_ACCURACY: 'test_neg_loss',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }
            
        elif task == 'segmentsemantic':
            
            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_acc',
                Metric.VAL_ACCURACY: 'valid_acc',
                Metric.TEST_ACCURACY: 'test_acc',
                Metric.TRAIN_LOSS: 'train_loss',
                Metric.VAL_LOSS: 'valid_loss',
                Metric.TEST_LOSS: 'test_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }    
        
        else: # ['normal', 'autoencoder']
            
            metric_to_tb101 = {
                Metric.TRAIN_ACCURACY: 'train_ssim',
                Metric.VAL_ACCURACY: 'valid_ssim',
                Metric.TEST_ACCURACY: 'test_ssim',
                Metric.TRAIN_LOSS: 'train_l1_loss',
                Metric.VAL_LOSS: 'valid_l1_loss',
                Metric.TEST_LOSS: 'test_l1_loss',
                Metric.TRAIN_TIME: 'time_elapsed',
            }
        

        
        if metric == Metric.RAW:
            # return all data
            return query_results.get_arch_result(arch_str).query_all_results()[task]


        if metric == Metric.HP:
            # return hyperparameter info
            return query_results[dataset]['cost_info']
        elif metric == Metric.TRAIN_TIME:
            return query_results.get_single_metric(arch_str, task, metric_to_tb101[metric])


        if full_lc and epoch == -1:
            return query_results[dataset][metric_to_tb101[metric]]
        elif full_lc and epoch != -1:
            return query_results[dataset][metric_to_tb101[metric]][:epoch]
        else:
            return query_results.get_single_metric(arch_str, task, metric_to_tb101[metric])



# build API

def _SPOS_trans101_micro_CNN(): 
    from xnas.core.config import cfg
    return SPOS_Trans101_micro(
        dataset=cfg.LOADER.DATASET,
        input_dim=(224, 224),
        C=cfg.TRAIN.CHANNELS
        # num_classes=cfg.LOADER.NUM_CLASSES,
    )

def _infer_trans101_macro_CNN(arch_hash): # (41414)
    from xnas.core.config import cfg
    return MacroNet(
        '64-{}-basic'.format(''.join(map(str,arch_hash))), 
        dataset=cfg.LOADER.DATASET,
        input_dim=(224, 224),
        C=cfg.TRAIN.CHANNELS
        # resize=cfg.SPOS.RESIZE,
        # num_classes=cfg.LOADER.NUM_CLASSES,
    )

def _infer_trans101_micro_CNN(arch_hash): # ()
    from xnas.core.config import cfg
    return MacroNet(
        '64-41414-{}_{}{}_{}{}{}'.format(*arch_hash),
        dataset=cfg.LOADER.DATASET,
        input_dim=(224, 224),
        C=cfg.TRAIN.CHANNELS
        # num_classes=cfg.LOADER.NUM_CLASSES,
    )

