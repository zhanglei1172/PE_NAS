"""Basic training recorders.

NOTE:
    If you want to add trainers inherited based on specific methods, 
    please place them in their search code. 

    Only the basic trainers are kept here.

"""

import gc
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from xnas.datasets.loader import get_train_val_loaders
import xnas.logger.timer as timer
import xnas.logger.meter as meter
import xnas.logger.logging as logging
import xnas.logger.checkpoint as checkpoint
from xnas.core.config import cfg

from torch.utils.tensorboard import SummaryWriter


__all__ = ["Trainer", "DartsTrainer", "OneShotTrainer", "KDTrainer"]


logger = logging.get_logger(__name__)


class Recorder():
    """Data recorder."""
    def __init__(self):
        self.full_timer = None
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tb"))
    
    def start(self):
        # recording full time
        self.full_timer = timer.Timer()
        self.full_timer.tic()
        
    def finish(self):
        # stop full time recording
        assert self.full_timer is not None, "not start yet."
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.full_timer.toc()
        logger.info("Overall time cost: {}".format(str(self.full_timer.total_time)))
        gc.collect()
        self.full_timer = None


class Trainer(Recorder):
    """Basic trainer."""
    def __init__(self, model, criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_meter = meter.TrainMeter(len(self.train_loader))
        self.test_meter = meter.TestMeter(len(self.test_loader))
        self.best_err = 23*3*3*3
        self.epoch_times = []

    def train_epoch(self, cur_epoch, rank=0, save=True, **kwargs):
        self.model.train()
        lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        st_time = time.time()
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            preds = self.model(inputs)
            if isinstance(preds, tuple):
                preds = preds[0]
            loss = self.criterion(preds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()

            # Compute the errors
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS)
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalar('train/loss', loss, cur_step)
            self.writer.add_scalar('train/top1_error', top1_err, cur_step)
            self.writer.add_scalar('train/top5_error', top5_err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.epoch_times.append(time.time() - st_time)
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        self.lr_scheduler.step()
        # Saving checkpoint
        if save and (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)

    @torch.no_grad()
    def test_epoch(self, cur_epoch, rank=0):
        self.model.eval()
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            preds = self.model(inputs)
            if isinstance(preds, tuple):
                preds = preds[0]
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            
            self.test_meter.iter_toc()
            self.test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
            self.test_meter.log_iter_stats(cur_epoch, cur_iter)
            self.test_meter.iter_tic()
        top1_err = self.test_meter.mb_top1_err.get_win_avg()
        self.writer.add_scalar('val/top1_error', self.test_meter.mb_top1_err.get_win_avg(), cur_epoch)
        self.writer.add_scalar('val/top5_error', self.test_meter.mb_top5_err.get_win_avg(), cur_epoch)
        # Log epoch stats
        self.test_meter.log_epoch_stats(cur_epoch)
        self.test_meter.reset()
        # Saving best model
        if self.best_err > top1_err:
            self.best_err = top1_err
            self.saving(cur_epoch, best=True)

    def resume(self, best=False):
        """Resume from previous checkpoints.
        may not loaded if there is no checkpoints.
        """
        if cfg.SEARCH.WEIGHTS:
            if os.path.exists(cfg.SEARCH.WEIGHTS):
                ckpt_epoch, ckpt_dict = checkpoint.load_checkpoint(cfg.SEARCH.WEIGHTS, self.model)
                return ckpt_epoch, ckpt_dict
            else:
                return -2, -2
        elif checkpoint.has_checkpoint():
            last_checkpoint = checkpoint.get_last_checkpoint(best=best)
            ckpt_epoch, ckpt_dict = checkpoint.load_checkpoint(last_checkpoint, self.model)
            return ckpt_epoch, ckpt_dict
        else:
            return -2, -2

    def saving(self, epoch, best=False, ckpt_dir=None):
        """Save to checkpoint."""
        _kwdict = {}
        if self.optimizer is not None:
            _kwdict['optimizer'] = self.optimizer
        if self.lr_scheduler is not None:
            _kwdict['lr_scheduler'] = self.lr_scheduler
        if hasattr(self, "epoch_times"):
            _kwdict['epoch_times'] = self.epoch_times
        _kwdict["tran_loader_indices"] = self.train_loader.sampler.indices
        _kwdict["test_loader_indices"] = self.test_loader.sampler.indices
        checkpoint_file = checkpoint.save_checkpoint(
            model=self.model, 
            epoch=epoch, 
            checkpoint_dir=ckpt_dir,
            best=best,
            **_kwdict
        )
        info_str = "Saving checkpoint to: {}".format(checkpoint_file)
        if best:
            info_str = "[Best] " + info_str
        logger.info(info_str)

    def loading(self):
        """Load from checkpoint."""
        ckpt_epoch, ckpt_dict = self.resume()
        if ckpt_epoch != -2:
            logger.info("Resume checkpoint from epoch: {}".format(ckpt_epoch+1))
            if self.optimizer is not None:
                self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
            if ckpt_dict.get("epoch_times"):
                self.epoch_times = ckpt_dict.get("epoch_times")
            if ckpt_dict.get("tran_loader_indices") and len(self.train_loader.sampler.indices) == len(ckpt_dict["tran_loader_indices"]):
                self.train_loader.sampler.indices = ckpt_dict["tran_loader_indices"]
            if ckpt_dict.get("test_loader_indices") and len(self.train_loader.sampler.indices) == len(ckpt_dict["tran_loader_indices"]):
                self.test_loader.sampler.indices = ckpt_dict["test_loader_indices"]
            return ckpt_epoch + 1
        else:
            return 0
            
class PE_Trainer(Trainer):
    """Basic trainer."""
    def __init__(self, model, criterion, optimizer, lr_scheduler,train_loader=None, test_loader=None):
        Recorder.__init__(self)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if train_loader is None and train_loader is None:
            self.train_loader, self.test_loader, _ = self.build_search_dataloaders(
                    cfg
                )
        else:
            self.train_loader, self.test_loader = train_loader, test_loader
        self.train_meter = meter.TrainMeter(len(self.train_loader))
        self.test_meter = meter.TestMeter(len(self.test_loader))
        self.best_err = 23*3*3*3
        self.epoch_times = []
        self.evaluate_meter = meter.TestMeter(len(self.test_loader))
        
    def tran_net(self, start_epoch=0, end_epoch=-1, save=True):
        # start training
        end_epoch = cfg.OPTIM.MAX_EPOCH if end_epoch == -1 else end_epoch
        self.start()
        for cur_epoch in range(start_epoch, end_epoch):
            # train epoch
            if cur_epoch == 0:
                # top1_err = self.test_epoch(cur_epoch=-1)
                if save:
                    self.saving(-1, best=False)
            top1_err = self.train_epoch(cur_epoch, save=save)
            # test epoch
            # if (cur_epoch+1) % cfg.EVAL_PERIOD == 0 or (cur_epoch+1) == cfg.OPTIM.MAX_EPOCH:
            #     top1_err = self.test_epoch(cur_epoch)
        self.finish()

    
    @staticmethod
    def build_search_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = get_train_val_loaders(
            config, mode="train"
        )
        return train_queue, valid_queue, _  # test_queue is not used in search currently

    @staticmethod
    def build_eval_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = get_train_val_loaders(
            config, mode="val"
        )
        return train_queue, valid_queue, test_queue

    @torch.no_grad()
    def evaluate_epoch(self, rank=0):
        """Return performance of the given sample (subnet)"""
        self.model.eval()
        # choice = self.evaluate_sampler.suggest()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            preds = self.model(inputs)
            if isinstance(preds, tuple):
                preds = preds[0]
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            self.evaluate_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        top1_err = self.evaluate_meter.mb_top1_err.get_global_avg()
        # self.evaluate_sampler.record(choice, top1_err)
        self.evaluate_meter.reset()
        return top1_err

class DartsTrainer(Trainer):
    """Basic DARTS-like network trainer.
    
    Methods:
        train_epoch: train epoch for DARTS, containing two-phase optimization
        darts_resume: resume w_optimizer and a_optimizer from ckpt_dict
    """
    def __init__(self, darts_controller, architect, criterion, w_optim, a_optim, lr_scheduler, train_loader, valid_loader):
        super().__init__(
            model=darts_controller,
            criterion=criterion,
            optimizer=w_optim,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            test_loader=valid_loader,
        )
        self.architect = architect
        self.a_optimizer = a_optim
        self.valid_loader = valid_loader    # DARTS uses valid_loader as both valid & test sets.

    def train_epoch(self, cur_epoch, alpha_step=True, rank=0, **kwargs):
        self.model.train()
        lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        valid_loader_iter = iter(self.valid_loader)  # using valid_loader during darts optimization
        for cur_iter, (trn_X, trn_y) in enumerate(self.train_loader):
            trn_X, trn_y = trn_X.to(rank), trn_y.to(rank, non_blocking=True)
            # hook for two-phase optimizing
            if alpha_step:
                try:
                    (val_X, val_y) = next(valid_loader_iter)
                except StopIteration:
                    valid_loader_iter = iter(self.valid_loader)
                    (val_X, val_y) = next(valid_loader_iter)
                val_X, val_y = val_X.to(rank), val_y.to(rank, non_blocking=True)
                
                # phase 2. architect step (alpha)
                self.a_optimizer.zero_grad()
                if cfg.DARTS.UNROLLED:
                    self.architect.unrolled_backward(
                        trn_X, trn_y, val_X, val_y, lr, 
                        self.optimizer, unrolled=cfg.DARTS.UNROLLED)
                else:
                    aloss = self.model.loss(val_X, val_y)
                    aloss.backward()
                self.a_optimizer.step()
                
            # phase 1. net weights step (w)            
            preds = self.model(trn_X)
            loss = self.criterion(preds, trn_y)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()

            # Compute the errors
            top1_err, top5_err = meter.topk_errors(preds, trn_y, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(top1_err, top5_err, loss, lr, trn_X.size(0) * cfg.NUM_GPUS)
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalar('train/loss', loss, cur_step)
            self.writer.add_scalar('train/top1_error', top1_err, cur_step)
            self.writer.add_scalar('train/top5_error', top5_err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        self.lr_scheduler.step()
        # saving model
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)

    def saving(self, epoch, best=False):
        """Save to checkpoint."""
        checkpoint_file = checkpoint.save_checkpoint(
            model=self.model, 
            epoch=epoch, 
            best=best,
            w_optim=self.optimizer, 
            a_optim=self.a_optimizer, 
            lr_scheduler=self.lr_scheduler,
        )
        logger.info("[Best: {}] saving checkpoint to: {}".format(best, checkpoint_file))

    def darts_loading(self):
        ckpt_epoch, ckpt_dict = self.resume()
        if ckpt_epoch != -1:
            logger.info("Resume checkpoint from epoch: {}".format(ckpt_epoch+1))
            self.optimizer.load_state_dict(ckpt_dict['w_optim'])
            self.a_optimizer.load_state_dict(ckpt_dict['a_optim'])
            self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
            return ckpt_epoch + 1
        else:
            return 0


class OneShotTrainer(Trainer):
    def __init__(self, supernet, criterion, optimizer, lr_scheduler, train_loader, test_loader, sample_type='epoch'):
        super().__init__(
            model=supernet, 
            criterion=criterion, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            train_loader=train_loader, 
            test_loader=test_loader)
        self.iter_sampler = None
        self.sample_type = sample_type
        assert self.sample_type in ['epoch', 'iter']
        self.evaluate_meter = meter.TestMeter(len(self.test_loader))
    
    def register_iter_sample(self, sampler):
        self.iter_sampler = sampler

    def train_epoch(self, cur_epoch, sample=None, rank=0, **kwargs):
        """Sample path from supernet and train it."""
        self.model.train()
        lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            # sample subnet
            self.optimizer.zero_grad()
            if sample is None and self.sample_type == 'iter':
                ssample = self.iter_sampler.suggest()
            else:
                ssample = sample
            ssample = np.array(ssample)
            if len(ssample.shape) == 2:
                for s in ssample:
                    preds = self.model(inputs, s)
                    loss = self.criterion(preds, labels)
                    
                    loss.backward()
            else:
                preds = self.model(inputs, ssample)
                loss = self.criterion(preds, labels)
                loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()

            # Compute the errors
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            if self.iter_sampler and self.sample_type == 'iter':
                self.iter_sampler.record(ssample, top1_err)     # use top1_err as evaluation
            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS)
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalar('train/loss', loss, cur_step)
            self.writer.add_scalar('train/top1_error', top1_err, cur_step)
            self.writer.add_scalar('train/top5_error', top5_err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        self.lr_scheduler.step()
        # Saving checkpoint
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)
        return top1_err

    @torch.no_grad()
    def test_epoch(self, cur_epoch, sample=None, rank=0):
        self.model.eval()
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            # sample subnet
            if sample is None and self.sample_type == 'iter':
                ssample = self.iter_sampler.suggest()
            else:
                ssample = sample
            ssample = np.array(ssample)
            if len(ssample.shape) == 2:
                for s in ssample:
                    preds = self.model(inputs, s)
                    break
            else:
                preds = self.model(inputs, ssample)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            if self.iter_sampler and self.sample_type == 'iter':
                self.iter_sampler.record(ssample, top1_err)     # use top1_err as evaluation
            self.test_meter.iter_toc()
            self.test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
            self.test_meter.log_iter_stats(cur_epoch, cur_iter)
            self.test_meter.iter_tic()
        top1_err = self.test_meter.mb_top1_err.get_win_avg()
        self.writer.add_scalar('val/top1_error', self.test_meter.mb_top1_err.get_win_avg(), cur_epoch)
        self.writer.add_scalar('val/top5_error', self.test_meter.mb_top5_err.get_win_avg(), cur_epoch)
        # Log epoch stats
        self.test_meter.log_epoch_stats(cur_epoch)
        self.test_meter.reset()
        # Saving best model
        if self.best_err > top1_err:
            self.best_err = top1_err
            self.saving(cur_epoch, best=True)
        return top1_err
            
    @torch.no_grad()
    def evaluate_epoch(self, sample, rank=0):
        """Return performance of the given sample (subnet)"""
        self.model.eval()
        # choice = self.evaluate_sampler.suggest()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            preds = self.model(inputs, sample)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            self.evaluate_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        top1_err = self.evaluate_meter.mb_top1_err.get_global_avg()
        # self.evaluate_sampler.record(choice, top1_err)
        self.evaluate_meter.reset()
        return top1_err

class OneShot_PE_Trainer(PE_Trainer):
    def __init__(self, supernet, criterion, optimizer, lr_scheduler, sample_type='epoch'):
        super().__init__(
            model=supernet, 
            criterion=criterion, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler)
        self.iter_sampler = None
        self.sample_type = sample_type
        assert self.sample_type in ['epoch', 'iter']
        self.evaluate_meter = meter.TestMeter(len(self.test_loader))
    
    def register_iter_sample(self, sampler):
        self.iter_sampler = sampler

    def train_epoch(self, cur_epoch, sample=None, rank=0, **kwargs):
        """Sample path from supernet and train it."""
        self.model.train()
        lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        st_time = time.time()
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            # sample subnet
            self.optimizer.zero_grad()
            if sample is None and self.sample_type == 'iter':
                ssample = self.iter_sampler.suggest()
            else:
                ssample = sample
            ssample = np.array(ssample)
            if len(ssample.shape) == 2:
                for s in ssample:
                    preds = self.model(inputs, s)
                    loss = self.criterion(preds, labels) # TODO loss /= num ?
                    
                    loss.backward()
            else:
                preds = self.model(inputs, ssample)
                loss = self.criterion(preds, labels)
                loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()

            # Compute the errors
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            if self.iter_sampler and self.sample_type == 'iter':
                self.iter_sampler.record(ssample, top1_err)     # use top1_err as evaluation
            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS)
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalar('train/loss', loss, cur_step)
            self.writer.add_scalar('train/top1_error', top1_err, cur_step)
            self.writer.add_scalar('train/top5_error', top5_err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        self.epoch_times.append(time.time() - st_time)
        self.lr_scheduler.step()
        # Saving checkpoint
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)
        return top1_err

    @torch.no_grad()
    def test_epoch(self, cur_epoch, sample=None, rank=0):
        self.model.eval()
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            # sample subnet
            if sample is None and self.sample_type == 'iter':
                ssample = self.iter_sampler.suggest()
            else:
                ssample = sample
            ssample = np.array(ssample)
            if len(ssample.shape) == 2:
                for s in ssample:
                    preds = self.model(inputs, s)
                    break
            else:
                preds = self.model(inputs, ssample)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            if self.iter_sampler and self.sample_type == 'iter':
                self.iter_sampler.record(ssample, top1_err)     # use top1_err as evaluation
            self.test_meter.iter_toc()
            self.test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
            self.test_meter.log_iter_stats(cur_epoch, cur_iter)
            self.test_meter.iter_tic()
        # top1_err = self.test_meter.mb_top1_err.get_win_avg()
        top1_err = self.test_meter.get_epoch_stats(0)["top1_err"]
        self.writer.add_scalar('val/top1_error', self.test_meter.mb_top1_err.get_win_avg(), cur_epoch)
        self.writer.add_scalar('val/top5_error', self.test_meter.mb_top5_err.get_win_avg(), cur_epoch)
        # Log epoch stats
        self.test_meter.log_epoch_stats(cur_epoch)
        self.test_meter.reset()
        # Saving best model
        if self.best_err > top1_err:
            self.best_err = top1_err
            self.saving(cur_epoch, best=True)
        return top1_err
            
    @torch.no_grad()
    def evaluate_epoch(self, sample, rank=0):
        """Return performance of the given sample (subnet)"""
        self.model.eval()
        # choice = self.evaluate_sampler.suggest()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            preds = self.model(inputs, sample)
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            self.evaluate_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        # top1_err = self.evaluate_meter.mb_top1_err.get_win_avg()
        top1_err = self.evaluate_meter.get_epoch_stats(0)["top1_err"]
        # self.evaluate_sampler.record(choice, top1_err)
        self.evaluate_meter.reset()
        return top1_err


class KDTrainer(Trainer):
    def __init__(self, model, teacher_model, kd_ratio, criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super().__init__(model, criterion, optimizer, lr_scheduler, train_loader, test_loader)
        self.teacher_model = teacher_model
        self.kd_ratio = kd_ratio
        from xnas.runner.criterion import CrossEntropyLoss_soft_target
        self.celoss_st = CrossEntropyLoss_soft_target
    
    def train_epoch(self, cur_epoch, rank=0, **kwargs):
        self.model.train()
        lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        self.writer.add_scalar('train/lr', lr, cur_step)
        self.train_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank, non_blocking=True)
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            
            # Knowledge Distillation from teacher model
            if (self.teacher_model is not None) and (self.kd_ratio > 0.):
                self.teacher_model.train()
                with torch.no_grad():
                    soft_logits = self.teacher_model(inputs).detach()
                    soft_label = F.softmax(soft_logits, dim=1)
                kd_loss = self.celoss_st(preds, soft_label)
                loss += kd_loss * self.kd_ratio
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.weights(), cfg.OPTIM.GRAD_CLIP)
            self.optimizer.step()

            # Compute the errors
            top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
            loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS)
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            self.writer.add_scalar('train/loss', loss, cur_step)
            self.writer.add_scalar('train/top1_error', top1_err, cur_step)
            self.writer.add_scalar('train/top5_error', top5_err, cur_step)
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        self.lr_scheduler.step()
        # Saving checkpoint
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(cur_epoch)
