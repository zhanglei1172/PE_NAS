import os, sys
import pickle
import time
import numpy as np
import gc
from xnas.algorithms.SPOS import RAND, RAND_ZC, REA, Fair, RAND_space
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


nb_models_path = os.path.join("data", "nb_models")
nb301_model_path = os.path.join(nb_models_path, "xgb_v1.0")
nb301_runtime_path = os.path.join(nb_models_path, "lgb_runtime_v1.0")
data_path = os.path.join("data", "nb301_full_training.pickle")



def main():
    device = setup_env()


    if cfg.SPACE.NAME == 'nasbenchmacro':
        num_choice, layers = 3, 8
        model = SUPPORTED_SPACES["nasbenchmacro"]().to(device)
    else:
        raise NotImplementedError

    dataset_api = get_dataset_api(cfg.SPACE.NAME, cfg.LOADER.DATASET, cfg)



    ####### One shot PE ########
    file_p = os.path.join(cfg.OUT_DIR, 'zc_arch.pkl')
    if os.path.exists(file_p):
        with open(os.path.join(cfg.OUT_DIR, 'zc_arch.pkl'), 'rb') as f:
            res = pickle.load(f)
        best_arch = res["best_arch"]
        zc_st_time, zc_ed_time = res["time"]
    else:
        zc_st_time = time.time()
        pe = ZeroCost(cfg, method_type=cfg.SEARCH.method_type)
        pe_eval = PEEvaluator(pe,config=cfg)
        pe_eval.adapt_search_space(model, load_labeled=False, dataset_api=dataset_api)

        pe.pre_process()

        evaluate_sampler = REA(3, 8)
        for cycle in range(500):  # NOTE: this should be a hyperparameter
            sample = evaluate_sampler.suggest()
            top1_err = -pe.query([sample]).item()
            evaluate_sampler.record(sample, top1_err)
        best_arch, best_top1err = evaluate_sampler.final_best()
        logger.info("Best arch: {} \nTop1 error: {}".format(best_arch, best_top1err))
        logger.info("Best arch: {} gt: {}".format(best_arch, dataset_api[''.join(map(str, best_arch))]))
        zc_ed_time = time.time()
        del pe
        gc.collect()
        res = {
            "best_arch": best_arch,
            "best_acc": dataset_api[''.join(map(str, best_arch))],
            "time": [zc_st_time, zc_ed_time]
        }
        with open(os.path.join(cfg.OUT_DIR
                               , 'zc_arch.pkl'),
                  'wb') as f:
            pickle.dump(res, f)
            
    logger.info("zero-cost st:{}, ed:{}, dur:{}".format(
        zc_st_time, zc_ed_time, zc_ed_time - zc_st_time))

    




if __name__ == "__main__":
    main()
