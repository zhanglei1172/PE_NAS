"""
This contains implementations of:
synflow, grad_norm, fisher, and grasp, and variants of jacov and snip 
based on https://github.com/mohsaied/zero-cost-nas
Note that zerocost_v1.py contains the original implementations
of jacov and snip. Particularly, the original jacov implementation tends to
perform better than the one in this file.
"""

import json
import random, os
import numpy as np
import torch
import torch.nn.functional as F
import math
from xnas.algorithms.RMINAS.sampler.sampling import array2genostr, hash2genostr
from xnas.evaluations.pe_evaluation import PE
from xnas.core.builder import space_builder
from xnas.core.utils import get_project_root
from xnas.datasets.loader import get_train_val_loaders
import xnas.logger.logging as logging
import xnas.models.nasbench2 as nas201_arch
import xnas.models.nasbench1 as nas101_arch
from xnas.models import nasbench1_spec
from xnas.algorithms.zero_cost_pe.utils import pe
from xnas.algorithms.zero_cost_pe.utils.measures.model_stats import get_model_stats
from xnas.spaces.DARTS.cnn import NetworkCIFAR, NetworkImageNet, _infer_DartsCNN, convert_darts_hash_to_genotype
from xnas.spaces.NASBench1Shot1.cnn import _infer_NASbench1shot1_1, _infer_NASbench1shot1_2, _infer_NASbench1shot1_3
from xnas.spaces.NASBench201.utils import dict2config, get_cell_based_tiny_net
from xnas.spaces.NASBenchMacro.cnn import Infer_NBM

logger = logging.get_logger(__name__)


class ZeroCost(PE):
    def __init__(self, config, method_type="jacov"):
        # available zero-cost method types: 'jacov', 'snip', 'synflow', 'grad_norm', 'fisher', 'grasp'

        self.dataload = "random"
        self.num_imgs_or_batches = 1
        self.method_type = method_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.data = "{}/data".format(get_project_root())
        self.config = config
        num_classes_dic = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}
        self.num_classes = None
        if self.config.LOADER.DATASET in num_classes_dic:
            self.num_classes = num_classes_dic[self.config.LOADER.DATASET]

    def pre_process(self, **kwargs):
        self.train_loader, _, _, _, _ = get_train_val_loaders(self.config, mode="train")

    def query(self, xtest, info=None):

        test_set_scores = []
        count = 0
        for test_arch in xtest:
            count += 1
            logger.info("zero cost: {} of {}".format(count, len(xtest)))
            
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

            # todo: rearrange the if statements so that nb101/201/darts can use this code as well:
            # try: # useful when launching bash scripts            
            if self.method_type in ['flops', 'params']:
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

                if self.method_type == 'params':
                    score = mega_params
                elif self.method_type == 'flops':
                    score = mega_flops

            else:

                score = pe.find_measures(
                    network,
                    self.train_loader,
                    (self.dataload, self.num_imgs_or_batches, self.num_classes),
                    self.device,
                    loss_fn=loss_fn,
                    measure_names=[self.method_type],
                )
            # except: # useful when launching bash scripts
            #     print('find_measures failed')
            #     score = -1e8

            # some of the values need to be flipped
            if math.isnan(score):
                score = -1e8

            if (
                "nasbench101" in self.config.SPACE.NAME
                and self.method_type == "jacov"
            ):
                score = -score
            elif "darts" in self.config.SPACE.NAME and self.method_type in [
                "fisher",
                "grad_norm",
                "synflow",
                "snip",
            ]:
                score = -score
            if os.environ.get('DEBUG'):
                zc_api = get_zc_benchmark_api(self.config.SPACE.NAME, self.config.LOADER.DATASET)
                gt_zc_score = zc_api[str(test_arch.get_hash())][self.method_type]['score']
                assert np.allclose(score, gt_zc_score)
            test_set_scores.append(score)
            torch.cuda.empty_cache()

        return np.array(test_set_scores)

def get_zc_benchmark_api(search_space, dataset):

    datafile_path = os.path.join("./data", f"zc_{search_space}.json")
    with open(datafile_path) as f:
        data = json.load(f)

    return data[dataset]