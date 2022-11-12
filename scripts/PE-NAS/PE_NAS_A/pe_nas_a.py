import os, sys
import pickle
import time
import numpy as np
import gc
from xnas.algorithms.SPOS import RAND, RAND_ZC, Fair, RAND_space
from xnas.algorithms.oneshot_pe.SPOS import SPOS_PE
from xnas.algorithms.partial_training_pe.partial_training import Partial_training_PE
from xnas.algorithms.zero_cost_pe.zero_cost import ZeroCost
from xnas.core.builder import SUPPORTED_SPACES
from xnas.evaluations.pe_evaluation import PEEvaluator
# from xnas.algorithms.oneshot
# from xnas.core.builder import arch_space_builder
import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *
from xnas.runner.trainer import OneShot_PE_Trainer
from xnas.search_sapces.get_dataset_api import get_dataset_api
from xnas.algorithms.RMINAS.sampler.sampling import hash2genostr, nb201genostr2array
from sklearn.ensemble import RandomForestClassifier

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)



def rank(l):
    return np.argsort(np.argsort(-np.array(l))) + 1


def arch2vec(space, arch):
    if space == 'nasbenchmacro':
        return list(arch)
    elif space == 'nasbench201':
        return list(arch)
    elif space == 'nasbench301':
        arch_array = np.zeros((28, 8))
        c = 0
        for i, t in enumerate(arch[0]):
            if i % 2 == 0:
                c += [0, 2, 3, 4][i // 2]
            arch_array[c + t[0], t[1]] = 1
        c = 14
        for i, t in enumerate(arch[1]):
            if i % 2 == 0:
                c += [0, 2, 3, 4][i // 2]
            arch_array[c + t[0], t[1]] = 1
        arch_array[:, -1] = 1 - arch_array.sum(-1)
        arch_array = np.roll(arch_array, 1, axis=1)
        return arch_array.argmax(-1)


def main():
    device = setup_env()
    # evaluator = evaluator_builder()
    # init sampler
    if cfg.SPACE.NAME == 'nasbench201':
        num_choice, layers = 5, 6
        model = SUPPORTED_SPACES["spos_nb201"]().to(device)
    elif cfg.SPACE.NAME == 'nasbenchmacro':
        num_choice, layers = 3, 8
        model = SUPPORTED_SPACES["nasbenchmacro"]().to(device)
    elif cfg.SPACE.NAME == 'nasbench301':
        num_choice, layers = 8, 14 * 2  # 14 edges 8 cell (6 normal + 2 reduce)
        model = SUPPORTED_SPACES["darts"]().to(device)
        # raise NotImplementedError
    elif cfg.SPACE.NAME == 'transbench101_micro':
        num_choice, layers = 4, 6
        model = SUPPORTED_SPACES["transbench101_micro"]().to(device)
    elif cfg.SPACE.NAME in [
            "nasbench1shot1_1", "nasbench1shot1_2", "nasbench1shot1_3"
    ]:
        model = SUPPORTED_SPACES[cfg.SPACE.NAME]().to(device)
        num_choice, layers = model.category, None

    dataset_api = get_dataset_api(cfg.SPACE.NAME, cfg.LOADER.DATASET, cfg)

    criterion = criterion_builder().to(device)
    optimizer = optimizer_builder("SGD", model.parameters())
    lr_scheduler = lr_scheduler_builder(optimizer)

    archs = list(model.get_all_architecture(dataset_api))

    ####### zero cost PE ########
    file_p = os.path.join(cfg.OUT_DIR, 'zc_arch.pkl')
    if os.path.exists(file_p):
        with open(os.path.join(cfg.OUT_DIR, 'zc_arch.pkl'), 'rb') as f:
            res = pickle.load(f)
        archs = res["space"]
        scores = res["scores"]
        zc_st_time, zc_ed_time = res["time"]
    else:
        num_workers = cfg.LOADER.NUM_WORKERS
        cfg.LOADER.NUM_WORKERS = 0
        zc_st_time = time.time()
        l2_pe = ZeroCost(cfg, method_type='l2_norm')
        nwot_pe = ZeroCost(cfg, method_type='nwot')
        zen_pe = ZeroCost(cfg, method_type='zen')
        l2_pe.pre_process()
        nwot_pe.train_loader = l2_pe.train_loader
        zen_pe.train_loader = l2_pe.train_loader
        # nwot_pe.pre_process()
        # zen_pe.pre_process()
        scores = []
        # for arch in archs:
        l2_score = l2_pe.query(archs)
        nwot_score = nwot_pe.query(archs)
        zen_score = zen_pe.query(archs)
        # scores.append([l2_score, nwot_score, zen_score])
        scores = np.stack([l2_score, nwot_score, zen_score], axis=1)

        zc_ed_time = time.time()
        del l2_pe, zen_pe, nwot_pe
        gc.collect()
        cfg.LOADER.NUM_WORKERS = num_workers

        with open(os.path.join(cfg.OUT_DIR, 'zc_arch.pkl'), 'wb') as f:
            # pickle.dump({"space":archs, "time":[zc_st_time, zc_ed_time]}, f)
            pickle.dump({"space":archs, "scores":scores, "time":[zc_st_time, zc_ed_time]}, f)
    logger.info("zero-cost st:{}, ed:{}, dur:{}".format(zc_st_time, zc_ed_time, zc_ed_time-zc_st_time))
    # sys.exit()
    # thres = np.quantile(scores, q=0.8, axis=0)
    thres = np.quantile(scores, q=0.8, axis=0, keepdims=True)
    # mask = (scores > thres)
    mask = (scores > thres).sum(axis=1)>=2
    reduced_arch = []
    for i, arch in enumerate(archs):
        if mask[i]:
            reduced_arch.append(arch)
    archs = reduced_arch
    # print(1)
    ####### One shot PE ########
    file_p = os.path.join(cfg.OUT_DIR, 'os_arch.pkl')
    if os.path.exists(file_p):
        with open(os.path.join(cfg.OUT_DIR, 'os_arch.pkl'),
                  'rb') as f:
            res = pickle.load(f)
        archs = res["space"]
        scores = res['scores']
        os_st_time, os_ed_time = res["time"]
    else:
        os_st_time = time.time()
        trainer = OneShot_PE_Trainer(supernet=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     lr_scheduler=lr_scheduler,
                                     sample_type='iter')
        # train_sampler = RAND_space(archs, cfg, trainer.train_loader)
        train_sampler = RAND(num_choice, layers)

        trainer.register_iter_sample(train_sampler)

        # arch_search_space = arch_space_builder()
        oneshot_pe = SPOS_PE(cfg, trainer)
        oneshot_pe.pre_process()
        # for arch in archs:
        scores = oneshot_pe.query(archs)

        os_ed_time = time.time()
        del oneshot_pe
        gc.collect()
        res = {
            "space": archs,
            "scores": scores,
            "time": [os_st_time, os_ed_time]
        }
        with open(os.path.join(cfg.OUT_DIR, 'os_arch.pkl'),
                  'wb') as f:
            # pickle.dump({"space":archs, "time":[os_st_time, os_ed_time]}, f)
            pickle.dump(res, f)
    logger.info("one-shot st:{}, ed:{}, dur:{}".format(
        os_st_time, os_ed_time, os_ed_time - os_st_time))

    thres = np.quantile(scores, q=0.8, axis=0)
    mask = (scores > thres)
    reduced_arch = []
    for i, arch in enumerate(archs):
        if mask[i]:
            reduced_arch.append(arch)
    archs = reduced_arch
    # sys.exit()
    ###### Partial training PE ########
    reduce_ratios_list = [0, 0.75, 0.41, 0.46, 0.13, 0.26, 0.23, 0.05, 0.34, 0.07, 0.36, 0.0, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0] # thres
    for i, ratio in enumerate(reduce_ratios_list[1:]):
        epoch = i+1
    # epoch = 1
        file_p = os.path.join(cfg.OUT_DIR,
                            'pt_arch_{}.pkl'.format(epoch))
        if os.path.exists(file_p):
            with open(
                    os.path.join(cfg.OUT_DIR,
                                'pt_arch_{}.pkl'.format(epoch)), 'rb') as f:
                res = pickle.load(f)
            archs = res["space"]
            scores = res["scores"]
            pt_st_time, pt_ed_time = res["time"]
        else:
            pt_st_time = time.time()
            pt_pe = Partial_training_PE(cfg)
            pt_pe.pre_process()
            scores = pt_pe.query(archs, end_epoch=epoch)
            pt_ed_time = time.time()
            res = {
                "space": archs,
                "scores": scores,
                "time": [pt_st_time, pt_ed_time]
            }
            with open(
                    os.path.join(cfg.OUT_DIR,
                                'pt_arch_{}.pkl'.format(epoch)), 'wb') as f:
                pickle.dump(res, f)
        logger.info("partial training st:{}, ed:{}, dur:{}".format(
            pt_st_time, pt_ed_time, pt_ed_time - pt_st_time))
        if ratio > 0:
            thres = np.quantile(scores, q=ratio, axis=0)
            mask = (scores > thres)
            reduced_arch = []
            for i, arch in enumerate(archs):
                if mask[i]:
                    reduced_arch.append(arch)
            archs = reduced_arch
        if len(archs) <= 1:
            break

if __name__ == "__main__":
    main()
