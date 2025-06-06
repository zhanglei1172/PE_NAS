import numpy as np
from collections import namedtuple
from xnas.core.query_metrics import Metric

from xnas.spaces.DARTS.ops import *
import xnas.spaces.DARTS.genos as gt

NUM_VERTICES = 4
NUM_OPS = 7

class DartsCell(nn.Module):
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, basic_op_list):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.basic_op_list = basic_op_list

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ReluConvBn(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ReluConvBn(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, C, stride, self.basic_op_list)
                self.dag[i].append(op)

    def forward(self, s0, s1, sample):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        w_dag = darts_weight_unpack(sample, self.n_nodes)
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)
        s_out = torch.cat(states[2:], 1)
        return s_out


class DartsCNN(nn.Module):

    def __init__(self, C=16, n_classes=10, n_layers=8, n_nodes=4, basic_op_list=[]):
        super().__init__()
        stem_multiplier = 3
        self.C_in = 3  # 3
        self.C = C  # 16
        self.n_classes = n_classes  # 10
        self.n_layers = n_layers  # 8
        self.n_nodes = n_nodes  # 4
        self.basic_op_list = DARTS_SPACE if len(basic_op_list) == 0 else basic_op_list
        self.non_op_idx = get_op_index(self.basic_op_list, NON_PARAMETER_OP)
        self.para_op_idx = get_op_index(self.basic_op_list, PARAMETER_OP)
        self.none_idx = 7
        C_cur = stem_multiplier * C  # 3 * 16 = 48
        self.stem = nn.Sequential(
            nn.Conv2d(self.C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C
        # 48   48   16
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = DartsCell(n_nodes, C_pp, C_p, C_cur,
                             reduction_p, reduction, self.basic_op_list)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_p, n_classes)
        # number of edges per cell
        self.num_edges = sum(list(range(2, self.n_nodes + 2)))
        self.num_ops = len(self.basic_op_list)
        # whole edges
        self.all_edges = 2 * self.num_edges
        self.norm_node_index = self._node_index(n_nodes, input_nodes=2, start_index=0)
        self.reduce_node_index = self._node_index(n_nodes, input_nodes=2, start_index=self.num_edges)

    def weights(self):
        return self.parameters()
    
    def forward(self, x, sample):
        s0 = s1 = self.stem(x)
        # sample = np.array(sample)
        if len(sample.shape) == 1:
            sample = np.eye(8)[sample]

        for i, cell in enumerate(self.cells):
            if (sample.shape[0]) == self.all_edges:
                weights = sample[self.num_edges:] if cell.reduction else sample[0:self.num_edges]
            elif (sample.shape[0]) == self.num_edges:
                weights = sample
            else:
                idx = i*self.num_edges
                weights = sample[idx:idx+self.num_edges]
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.classifier(out)
        return logits

    def genotype(self, theta):
        Genotype = namedtuple(
            'Genotype', 'normal normal_concat reduce reduce_concat')
        theta_norm = darts_weight_unpack(
            theta[0:self.num_edges], self.n_nodes)
        theta_reduce = darts_weight_unpack(
            theta[self.num_edges:], self.n_nodes)
        gene_normal = parse_from_numpy(
            theta_norm, k=2, basic_op_list=self.basic_op_list)
        gene_reduce = parse_from_numpy(
            theta_reduce, k=2, basic_op_list=self.basic_op_list)
        concat = range(2, 2+self.n_nodes)  # concat all intermediate nodes
        return Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)

    def genotype_to_onehot_sample(self, genotype):
        sample = np.zeros([self.all_edges, len(self.basic_op_list)])
        norm_gene = genotype[0]
        reduce_gene = genotype[2]
        num_select = list(range(2, 2+self.n_nodes))
        for j, _gene in enumerate([norm_gene, reduce_gene]):
            for i, node in enumerate(_gene):
                for op in node:
                    op_name = op[0]
                    op_id = op[1]
                    if i == 0:
                        true_id = op_id + j * self.num_edges
                    else:
                        if i == 1:
                            _temp = num_select[0]
                        else:
                            _temp = sum(num_select[0:i])
                        true_id = op_id + _temp + j * self.num_edges
                    sample[true_id, self.basic_op_list.index(op_name)] = 1
        for i in range(self.all_edges):
            if np.sum(sample[i, :]) == 0:
                sample[i, len(self.basic_op_list)-1] = 1
        return sample

    def _node_index(self, n_nodes, input_nodes=2, start_index=0):
        node_index = []
        start_index = start_index
        end_index = input_nodes + start_index
        for i in range(n_nodes):
            node_index.append(list(range(start_index, end_index)))
            start_index = end_index
            end_index += input_nodes + i + 1
        return node_index

    def query(
        self,arch_hash,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
    ):
        """
        Query results from nasbench 301
        """
        if dataset_api is None:
            raise NotImplementedError('Must pass in dataset_api to query NAS-Bench-301')

        metric_to_nb301 = {
            Metric.TRAIN_LOSS: "train_losses",
            Metric.VAL_ACCURACY: "val_accuracies",
            Metric.TEST_ACCURACY: "val_accuracies",
            Metric.TRAIN_TIME: "runtime",
        }

        assert not epoch or epoch in [-1, 100]
        # assert metric in [Metric.VAL_ACCURACY, Metric.RAW]
        genotype = convert_darts_hash_to_genotype(arch_hash)
        if metric == Metric.VAL_ACCURACY:
            val_acc = dataset_api["nb301_model"][0].predict(
                config=genotype, representation="genotype"
            )
            return val_acc
        elif metric == Metric.TRAIN_TIME:
            runtime = dataset_api["nb301_model"][1].predict(
                config=genotype, representation="genotype"
            )
            return runtime
        elif full_lc and epoch == -1:
            return dataset_api["nb301_data"][arch_hash][metric_to_nb301[metric]]
        elif full_lc and epoch != -1:
            return dataset_api["nb301_data"][arch_hash][metric_to_nb301[metric]][:epoch]
        else:
            return -1

    def sample_random_architecture(self, dataset_api=None, load_labeled=False):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        (in_node_idx, op_idx)
        """

        # if load_labeled == True:
        #     return self.sample_random_labeled_architecture()

        compact = [[], []]
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(NUM_OPS), NUM_VERTICES)

            nodes_in_normal = np.sort(np.random.choice(range(i + 2), 2, replace=False))
            nodes_in_reduce = np.sort(np.random.choice(range(i + 2), 2, replace=False))

            compact[0].extend(
                [(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])]
            )
            compact[1].extend(
                [(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])]
            )

        return (tuple(compact[0]), tuple(compact[1]))

    def get_type(self,):
        return "nasbench301"

    def load_labeled_architecture(self, dataset_api=None):
        """
        This is meant to be called by a new DartsSearchSpace() object
        (one that has not already been discretized).
        It samples a random architecture from the nasbench301 training data,
        and updates the graph object to match the architecture.
        """
        index = np.random.choice(len(dataset_api["nb301_arches"]))
        compact = dataset_api["nb301_arches"][index]
        self.load_labeled = True
        return compact
    
    def get_all_architecture(self,dataset_api=None):
        return dataset_api["nb301_arches"]

# Augmented DARTS
def convert_darts_hash_to_genotype(compact):
    """Converts the compact representation to a Genotype"""
    genotype = []

    for i in range(2):
        cell = compact[i]
        genotype.append([])

        for j in range(8):
            genotype[i].append((gt.PRIMITIVES[cell[j][1]], cell[j][0]))

    return gt.Genotype(
        normal=genotype[0],
        normal_concat=[2, 3, 4, 5],
        reduce=genotype[1],
        reduce_concat=[2, 3, 4, 5],
    )

class AugmentCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(AugmentCell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReluConvBn(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReluConvBn(C_prev, C, 1, 1, 0)
        
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path_(h1, drop_prob, self.training)
                if not isinstance(op2, Identity):
                    h2 = drop_path_(h2, drop_prob, self.training)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0.0
        
        self.genotype = gt.from_str(genotype)

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = AugmentCell(self.genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
            if i == 2*layers//3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
    
    def weights(self):
        return self.parameters()

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2*self._layers//3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits, logits_aux

    def forward_with_features(self, input):
        features = []
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i in [int(self._layers//3-1), int(2*self._layers//3-1), int(self._layers-1)]:
                features.append(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return features, logits

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
    
    def get_hash(self,):
        hash_arch = [[],[]]
        normal = self.genotype.normal
        reduce = self.genotype.reduce

        op_names, indices = zip(*normal)
        hash_arch[0] = tuple(zip(indices,(gt.PRIMITIVES.index(op) for op in op_names)))
        op_names, indices = zip(*reduce)
        hash_arch[1] = tuple(zip(indices,(gt.PRIMITIVES.index(op) for op in op_names)))
        return tuple(hash_arch)

class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.genotype = gt.from_str(genotype)
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = AugmentCell(self.genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)
    
    def weights(self):
        return self.parameters()

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

    def forward_with_features(self, input):
        logits_aux = None
        features = []
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i in [int(self._layers//3-1), int(2*self._layers//3-1), int(self._layers-1)]:
                features.append(s1)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return features, logits, logits_aux

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

    def get_hash(self,):
        hash_arch = [[],[]]
        normal = self.genotype.normal
        reduce = self.genotype.reduce

        op_names, indices = zip(*normal)
        hash_arch[0] = tuple(zip(indices,(gt.PRIMITIVES.index(op) for op in op_names)))
        op_names, indices = zip(*reduce)
        hash_arch[1] = tuple(zip(indices,(gt.PRIMITIVES.index(op) for op in op_names)))
        return tuple(hash_arch)

# build API

def _DartsCNN():
    from xnas.core.config import cfg
    return DartsCNN(
        C=cfg.SPACE.CHANNELS,
        n_classes=cfg.LOADER.NUM_CLASSES,
        n_layers=cfg.SPACE.LAYERS,
        n_nodes=cfg.SPACE.NODES,
        basic_op_list=cfg.SPACE.BASIC_OP)

def _infer_DartsCNN():
    from xnas.core.config import cfg
    if cfg.LOADER.DATASET in ['cifar10', 'cifar100', 'imagenet16']:
        return NetworkCIFAR(
            C=cfg.TRAIN.CHANNELS,
            num_classes=cfg.LOADER.NUM_CLASSES,
            layers=cfg.TRAIN.LAYERS,
            auxiliary=cfg.TRAIN.AUX_WEIGHT > 0,
            genotype=cfg.TRAIN.GENOTYPE,
        )
    elif cfg.LOADER.DATASET == 'imagenet':
        return NetworkImageNet(
            C=cfg.TRAIN.CHANNELS,
            num_classes=cfg.LOADER.NUM_CLASSES,
            layers=cfg.TRAIN.LAYERS,
            auxiliary=cfg.TRAIN.AUX_WEIGHT > 0,
            genotype=cfg.TRAIN.GENOTYPE,
        )
    else:
        print("dataset not support.")
    exit(1)
