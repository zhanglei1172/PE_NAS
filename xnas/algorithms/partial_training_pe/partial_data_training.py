


import glob
import os
import re
import numpy as np
import torch
import torch.nn as nn
from xnas.algorithms.RMINAS.sampler.sampling import hash2genostr
from xnas.core.query_metrics import Metric
from xnas.core.utils import get_project_root
from xnas.evaluations.pe_evaluation import PE
from xnas.logger.checkpoint import _DIR_NAME, change_name_prefix_to, get_checkpoint_dir, get_checkpoint_name
from xnas.datasets.loader import get_train_val_loaders
from xnas.runner.criterion import criterion_builder
from xnas.runner.optimizer import optimizer_builder
from xnas.runner.scheduler import lr_scheduler_builder
from xnas.runner.trainer import PE_Trainer
from xnas.spaces.DARTS.cnn import convert_darts_hash_to_genotype
from xnas.spaces.DARTS.cnn import NetworkCIFAR, NetworkImageNet, _infer_DartsCNN, convert_darts_hash_to_genotype
from xnas.spaces.NASBench1Shot1.cnn import _infer_NASbench1shot1_1, _infer_NASbench1shot1_2, _infer_NASbench1shot1_3
from xnas.spaces.NASBench201.utils import dict2config, get_cell_based_tiny_net
from xnas.spaces.NASBenchMacro.cnn import Infer_NBM

class Partial_data_training_PE(PE):
    def __init__(self, config):
        self.config = config
        # self.trainer = trainer
        config.data = "{}/data".format(get_project_root())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def pre_process(self, **kwargs):
        """
        This is called at the start of the NAS algorithm,
        before any architectures have been queried
        """
        self.train_loader, self.test_loader, _, _, _ = get_train_val_loaders(self.config, mode="train", **kwargs)
        # if self.config.SEARCH.ALL_NOT_BN:
        #     self.set_bn(False)
        # if self.config.SEARCH.AUTO_RESUME:
        #     start_epoch = self.trainer.loading()
        # else:
        #     start_epoch = 0            
        # self.trainer.tran_supernet(start_epoch)
        # # think whether recalibrate bn
        # self.train_times = self.trainer.epoch_times

    def set_bn(self, track_running_stats=False):
        device = next(self.trainer.model.parameters()).device
        for m in self.trainer.model.modules():
            to_replace_dict = {}
            for name, sub_m in m.named_children():
                if isinstance(sub_m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    if sub_m.track_running_stats != track_running_stats:
                        param = {"num_features":sub_m.num_features,
                         "eps": sub_m.eps,
                         "momentum":sub_m.momentum,
                         "affine":sub_m.affine,
                         "track_running_stats":track_running_stats,
                         "device": device}
                        bn_m = sub_m.__class__(**param)
                        
                        if param["affine"]:
                            # load weight
                            bn_m.weight.data.copy_(sub_m.weight.data)
                            bn_m.bias.data.copy_(sub_m.bias.data)
                            # load requires_grad
                            bn_m.weight.requires_grad = sub_m.weight.requires_grad
                            bn_m.bias.requires_grad = sub_m.bias.requires_grad
                        to_replace_dict[name] = bn_m
            m._modules.update(to_replace_dict)

    def fit(self, xtrain, ytrain, train_info=None, verbose=0):
        pass

    def query(self, xtest, info=None, eval_batch_size=None):
        # global _NAME_PREFIX
        test_set_scores = []
        count = 0
        for i,test_arch in enumerate(xtest):

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

            criterion = criterion_builder().to(self.device)
            network = network.to(self.device)
            optimizer = optimizer_builder("SGD", network.parameters())
            lr_scheduler = lr_scheduler_builder(optimizer)
            trainer = PE_Trainer(
                model=network,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
            )

            trainer.tran_net()
            top1_err = trainer.evaluate_epoch()
            test_set_scores.append(-top1_err)
        return np.array(test_set_scores)
    

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the pe needs
        extra info to train/query, such as a partial learning curve,
        or hyperparameters of the architecture
        """
        reqs = {
            "requires_partial_lc": True,
            "metric": Metric.TEST_ACCURACY,
            "requires_hyperparameters": False,
            "hyperparams": {},
            "unlabeled": False,
            "unlabeled_factor": 0,
        }
        return reqs