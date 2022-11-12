"""Samplers for Single Path One-Shot Search Space."""

import math
import pickle
import numpy as np
from copy import deepcopy
from collections import deque
import torch
import torch.nn.functional as F

from xnas.algorithms.RMINAS.sampler.sampling import hash2genostr
from xnas.algorithms.zero_cost_pe.utils.measures.model_stats import get_model_stats
from xnas.algorithms.zero_cost_pe.utils.pe import find_measures
from xnas.spaces.DARTS.cnn import NetworkCIFAR, NetworkImageNet, convert_darts_hash_to_genotype
from xnas.spaces.NASBench1Shot1.cnn import _infer_NASbench1shot1_1, _infer_NASbench1shot1_2, _infer_NASbench1shot1_3
from xnas.spaces.NASBench201.utils import dict2config, get_cell_based_tiny_net
from xnas.spaces.NASBenchMacro.cnn import Infer_NBM


class RAND():
    """Random choice"""
    def __init__(self, num_choice, layers=None):
        self.num_choice = num_choice
        self.child_len = layers
        if layers is None:
            assert isinstance(num_choice,list)
        self.history = []
        
    def record(self, child, value):
        self.history.append({"child":child, "value":value})
    
    def suggest(self):
        if self.child_len is None:
            return list(np.random.randint(_) for _ in self.num_choice)
        else:
            return list(np.random.randint(self.num_choice, size=self.child_len))
    
    def final_best(self):
        best_child = min(self.history, key=lambda i:i["value"])
        return best_child['child'], best_child['value']

class RAND_ZC():
    """Random choice"""
    def __init__(self, model, cfg, train_loader, thres_ratio=0.2, warmup_num=100):
        self.model = model
        self.scores = []
        self.warmup_num = warmup_num
        self.thres_ratio = thres_ratio
        method_type = cfg.SEARCH.method_type.split('zc_spos_')[-1]
        # self.num_choice = num_choice
        # self.child_len = layers
        # if layers is None:
            # assert isinstance(num_choice,list)
        self.method_type = method_type
        self.history = []
        self.config = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataload = "random"
        self.num_imgs_or_batches = 1
        num_classes_dic = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}
        self.num_classes = None
        if self.config.LOADER.DATASET in num_classes_dic:
            self.num_classes = num_classes_dic[self.config.LOADER.DATASET]
        self.train_loader = train_loader
        self._pre_process()
        
    def _pre_process(self,):
        if self.config.SPACE.NAME == 'nasbench301':
            with open("./res_301.pkl", "rb") as f:
                self.nb_data = pickle.load(f)
        elif self.config.SPACE.NAME == 'nasbench201':
            with open("./res_201.pkl", "rb") as f:
                self.nb_data = pickle.load(f)
        elif self.config.SPACE.NAME == 'nasbenchmacro':
            with open("./res_macro.pkl", "rb") as f:
                self.nb_data = pickle.load(f)
        else:
            raise NotImplementedError
        
        
    def get_score(self,test_arch):
        raise NotImplementedError
        if self.method_type not in ('combine_rank_3','vote','intersection'):
            
            score = self.get_zc_score(test_arch, self.method_type)
        else:
            l2_norm = self.get_zc_score(test_arch, 'l2_norm')
            nwot = self.get_zc_score(test_arch, 'nwot')
            zen = self.get_zc_score(test_arch, 'zen')
            
        
    def get_zc_score(self, test_arch, method_type):
        if self.config.SPACE.NAME == 'nasbench201':
            # get arch
            arch_str = hash2genostr(test_arch)
            arch_config = {
                'name': 'infer.tiny', 
                'C': 16, 'N': 5, 
                'arch_str':arch_str, 
                'num_classes': self.config.LOADER.NUM_CLASSES}
            net_config = dict2config(arch_config, None)
            network = get_cell_based_tiny_net(net_config)#.cuda()
        elif self.config.SPACE.NAME == 'nasbench301':
            genotype = convert_darts_hash_to_genotype(test_arch)
            # self.config.TRAIN.GENOTYPE = 
            # network = _infer_DartsCNN()
            if self.config.LOADER.DATASET in ['cifar10', 'cifar100', 'imagenet16']:
                network = NetworkCIFAR(
                    C=32,
                    num_classes=self.config.LOADER.NUM_CLASSES,
                    layers=8,
                    auxiliary=False,
                    genotype=str(genotype),
                )
            elif self.config.LOADER.DATASET == 'imagenet':
                network = NetworkImageNet(
                    C=32,
                    num_classes=self.config.LOADER.NUM_CLASSES,
                    layers=8,
                    auxiliary=False,
                    genotype=str(genotype),
                )
        elif self.config.SPACE.NAME == 'nasbenchmacro':
            network = Infer_NBM(arch_hash=test_arch)#.cuda()
        elif self.config.SPACE.NAME == 'nasbench1shot1_1':
            network = _infer_NASbench1shot1_1(arch_hash=test_arch)#.cuda()
        elif self.config.SPACE.NAME == 'nasbench1shot1_2':
            network = _infer_NASbench1shot1_2(arch_hash=test_arch)#.cuda()
        elif self.config.SPACE.NAME == 'nasbench1shot1_3':
            network = _infer_NASbench1shot1_3(arch_hash=test_arch)#.cuda()
        # set up loss function
        if self.config.LOADER.DATASET in ['class_object', 'class_scene']:
            raise NotImplementedError
            # loss_fn = SoftmaxCrossEntropyWithLogits()
        elif self.config.LOADER.DATASET == 'autoencoder':
            loss_fn = torch.nn.L1Loss()
        else:
            loss_fn = F.cross_entropy

        network = network.to(self.device)
        # return network, loss_fn

        # todo: rearrange the if statements so that nb101/201/darts can use this code as well:
        # try: # useful when launching bash scripts            
        if method_type in ['flops', 'params']:
            """
            This code is from
            https://github.com/microsoft/archai/blob/5fc5e5aa63f3ac51a384b41e32eee4e7e5da2481/archai/common/trainer.py#L201
            """
            data_iterator = iter(self.train_loader)
            x, target = next(data_iterator)
            x_shape = list(x.shape)
            x_shape[0] = 1 # to prevent overflow errors with large batch size we will use a batch size of 1
            model_stats = get_model_stats(network, input_tensor_shape=x_shape, clone_model=True)

            # important to do to avoid overflow
            mega_flops = float(model_stats.Flops)/1e6
            mega_params = float(model_stats.parameters)/1e6

            if method_type == 'params':
                score = mega_params
            elif method_type == 'flops':
                score = mega_flops

        else:
            score = find_measures(
                network,
                self.train_loader,
                (self.dataload, self.num_imgs_or_batches, self.num_classes),
                self.device,
                loss_fn=loss_fn,
                measure_names=[method_type],
            )
        # except: # useful when launching bash scripts
        #     print('find_measures failed')
        #     score = -1e8

        # some of the values need to be flipped
        if math.isnan(score):
            score = -1e8

        if (
            "nasbench101" in self.config.SPACE.NAME
            and method_type == "jacov"
        ):
            score = -score
        elif "darts" in self.config.SPACE.NAME and method_type in [
            "fisher",
            "grad_norm",
            "synflow",
            "snip",
        ]:
            score = -score
        score = -score
        torch.cuda.empty_cache()
        self.scores.append(score)
        if len(self.scores) == self.warmup_num:
            self.thres = np.quantile(self.scores, self.thres_ratio)
        return score # minimize
        
    def record(self, child, value):
        self.history.append({"child":child, "value":value})
    
    def suggest(self):
        # if self.child_len is None:
        while True:
            # arch = list(np.random.randint(_) for _ in self.num_choice)
            if self.method_type == 'all':
                arch = self.model.sample_random_architecture()
            else:
                arch_idx = np.random.randint(len(self.nb_data["kept_arch"][self.method_type]))
                arch = self.nb_data["kept_arch"][self.method_type][arch_idx]
            if self.config.SPACE.NAME == 'nasbench301':
                # def to_301_array(a):
                arch_array = np.zeros((28, 8))
                c = 0
                for i, t in enumerate(arch[0]):
                    if i  % 2 == 0:
                        c += [0,2,3,4][i//2]
                    arch_array[c+t[0], t[1]] = 1
                c = 14
                for i, t in enumerate(arch[1]):
                    if i  % 2 == 0:
                        c += [0,2,3,4][i//2]
                    arch_array[c+t[0], t[1]] = 1
                arch_array[:,-1] = 1-arch_array.sum(-1)
                arch_array = np.roll(arch_array, 1, axis=1)
                return arch_array.argmax(-1)
            else:
                return arch
            # score = self.get_zc_score(arch)
            # if len(self.scores) >= self.warmup_num and score <= self.thres:
            #     return arch
        # else:
        #     while True:
        #         # arch = list(np.random.randint(self.num_choice, size=self.child_len))
        #         arch = self.model.sample_random_architecture()
        #         score = self.get_zc_score(arch)
        #         if len(self.scores) >= self.warmup_num and score <= self.thres:
        #             return arch
    
    def final_best(self):
        best_child = min(self.history, key=lambda i:i["value"])
        return best_child['child'], best_child['value']


class Fair():
    """Fair choice"""
    def __init__(self, num_choice, layers):
        self.num_choice = num_choice
        self.child_len = layers
        self.history = []
        
    def record(self, child, value):
        self.history.append({"child":child, "value":value})
    
    def suggest(self):
        
        return np.array([np.random.permutation(self.num_choice) for _ in range(self.child_len)]).T
    
    def final_best(self):
        best_child = min(self.history, key=lambda i:i["value"])
        return best_child['child'], best_child['value']


class REA():
    """Regularized Evolution Algorithm"""
    def __init__(self, num_choice, layers, population_size=20, better=min):
        self.num_choice = num_choice
        self.population_size = population_size
        self.child_len = layers
        self.better = better
        self.population = deque()
        self.history = []
        # init population
        self.init_pop = np.random.randint(
            self.num_choice, size=(self.population_size, self.child_len)
        )

    def _get_mutated_parent(self):
        parent = self.better(self.population, key=lambda i:i["value"])  # default: min(error)
        return self._mutate(parent['child'])

    def _mutate(self, parent):
        parent = deepcopy(parent)
        idx = np.random.randint(0, len(parent))
        prev_value, new_value = parent[idx], parent[idx]
        while new_value == prev_value:
            new_value = np.random.randint(self.num_choice)
        parent[idx] = new_value
        return parent

    def record(self, child, value):
        self.history.append({"child":child, "value":value})
        self.population.append({"child":child, "value":value})
        if len(self.population) > self.population_size:
            self.population.popleft()

    def suggest(self):
        if len(self.history) < self.population_size:
            return list(self.init_pop[len(self.history)])
        else:
            return self._get_mutated_parent()
    
    def final_best(self):
        best_child = self.better(self.history, key=lambda i:i["value"])
        return best_child['child'], best_child['value']

def rank(l):
    return np.argsort(np.argsort(-np.array(l)))+1

class RAND_space():
    """Random choice"""
    def __init__(self, archs, cfg, train_loader, thres_ratio=0.2, warmup_num=100):
        self.scores = []
        self.warmup_num = warmup_num
        self.thres_ratio = thres_ratio
        method_type = cfg.SEARCH.method_type.split('zc_spos_')[-1]
        # self.num_choice = num_choice
        # self.child_len = layers
        # if layers is None:
            # assert isinstance(num_choice,list)
        self.method_type = method_type
        self.history = []
        self.config = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataload = "random"
        self.num_imgs_or_batches = 1
        num_classes_dic = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}
        self.num_classes = None
        if self.config.LOADER.DATASET in num_classes_dic:
            self.num_classes = num_classes_dic[self.config.LOADER.DATASET]
        self.train_loader = train_loader
        self.space = archs
        
        
    def record(self, child, value):
        self.history.append({"child":child, "value":value})
    
    def suggest(self):
        # if self.child_len is None:
        while True:
            arch_idx = np.random.randint(len(self.space))
            arch = self.space[arch_idx]
            if self.config.SPACE.NAME == 'nasbench301':
                # def to_301_array(a):
                arch_array = np.zeros((28, 8))
                c = 0
                for i, t in enumerate(arch[0]):
                    if i  % 2 == 0:
                        c += [0,2,3,4][i//2]
                    arch_array[c+t[0], t[1]] = 1
                c = 14
                for i, t in enumerate(arch[1]):
                    if i  % 2 == 0:
                        c += [0,2,3,4][i//2]
                    arch_array[c+t[0], t[1]] = 1
                arch_array[:,-1] = 1-arch_array.sum(-1)
                arch_array = np.roll(arch_array, 1, axis=1)
                return arch_array.argmax(-1)
            else:
                return arch
    
    def final_best(self):
        best_child = min(self.history, key=lambda i:i["value"])
        return best_child['child'], best_child['value']