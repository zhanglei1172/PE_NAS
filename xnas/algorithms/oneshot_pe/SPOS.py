


import os
import numpy as np
import torch
import torch.nn as nn
from xnas.algorithms.RMINAS.sampler.sampling import hash2genostr
from xnas.core.utils import get_project_root
from xnas.evaluations.pe_evaluation import PE
from xnas.logger.checkpoint import _DIR_NAME, get_checkpoint_name
from xnas.runner.trainer import PE_Trainer
from xnas.spaces.DARTS.cnn import convert_darts_hash_to_genotype


class SPOS_PE(PE):
    def __init__(self, config, trainer: PE_Trainer):
        self.config = config
        self.trainer = trainer
        config.data = "{}/data".format(get_project_root())


    def pre_process(self, train_net=True, **kwargs):
        """
        This is called at the start of the NAS algorithm,
        before any architectures have been queried
        """
        if self.config.SEARCH.ALL_NOT_BN:
            self.set_bn(False)
        print(self.trainer.model)
        if self.config.SEARCH.AUTO_RESUME:
            start_epoch = self.trainer.loading()
        else:
            start_epoch = 0
        if train_net:       
            self.trainer.tran_net(start_epoch)
        # think whether recalibrate bn
        self.train_times = self.trainer.epoch_times

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

    # def __call__(self, archs):
    #     """
    #     Evaluate, i.e. do a forward pass for every image datapoint, the
    #     one-shot model for every architecture in archs.
    #         params:
    #             archs: torch.Tensor where each element is an architecture encoding

    #         return:
    #             torch.Tensor with the predictions for every arch in archs
    #     """
    #     prediction = []
    #     for arch in archs:
    #         # we have to iterate through all the architectures in the
    #         # mini-batch
    #         self.model.optimizer.set_alphas_from_path(arch)
    #         # NOTE: evaluation on the 25k validation data for now. provide a test
    #         # dataloader to evaluate on the test data
    #         val_acc = self.model.evaluate_oneshot(dataloader=None)
    #         prediction.append(val_acc)
    #     print("Predictions:")
    #     print(prediction)

    #     return prediction

    def fit(self, xtrain, ytrain, train_info=None, verbose=0):
        pass

    def query(self, xtest, info=None, eval_batch_size=None):
        test_set_scores = []
        count = 0
        for test_arch in xtest:
            if self.config.SPACE.NAME == 'nasbench201':
                # get arch
                sample = list(test_arch)
            elif self.config.SPACE.NAME == 'nasbench301':
                normal = test_arch[0]
                in_node_idxs, op_idxs = zip(*normal)
                normal_sample = np.zeros((14, 8))
                cum = 0
                for _i in range(0,len(in_node_idxs), 2):
                    
                    row = in_node_idxs[_i] + cum
                    normal_sample[row, op_idxs[_i]] = 1
                    row = in_node_idxs[_i+1] + cum
                    normal_sample[row, op_idxs[_i+1]] = 1
                    cum += _i//2 + 2 # 0, 2, 2+3, 2+3+4
                reduce = test_arch[1]
                in_node_idxs, op_idxs = zip(*reduce)
                reduce_sample = np.zeros((14, 8))
                cum = 0
                for _i in range(0,len(in_node_idxs), 2):
                    
                    row = in_node_idxs[_i] + cum
                    reduce_sample[row, op_idxs[_i]] = 1
                    row = in_node_idxs[_i+1] + cum
                    reduce_sample[row, op_idxs[_i+1]] = 1
                    cum += _i//2 + 2 # 0, 2, 2+3, 2+3+4
                    
                sample = np.concatenate([normal_sample, reduce_sample], axis=0)
                sample[:, -1] = 1 - np.sum(sample, axis=-1)
                
            elif self.config.SPACE.NAME == 'nasbenchmacro':
                sample = list(test_arch)
            elif self.config.SPACE.NAME in ['nasbench1shot1_1','nasbench1shot1_2','nasbench1shot1_3']:
                sample = list(test_arch)
            
            top1_err = self.trainer.evaluate_epoch(sample)
            test_set_scores.append(-top1_err)
        return np.array(test_set_scores)
    
    def query_every_epoch(self, xtest, info=None, eval_batch_size=None):
        # self.config.OUT_DIR
        y_preds = []
        for epoch in range(0, self.config.OPTIM.MAX_EPOCH+1, self.config.SAVE_PERIOD):
            file_name = get_checkpoint_name(epoch, '/'.join(self.config.SEARCH.WEIGHTS.split('/')[:-1]))
            self.config.SEARCH.WEIGHTS = file_name
            self.trainer.loading()
            y_pred = self.query(xtest, info, eval_batch_size)
            y_preds.append(y_pred)
        return np.array(y_preds)