from xnas.algorithms.SPOS import RAND, Fair
from xnas.algorithms.oneshot_pe.SPOS import SPOS_PE
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


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main():
    device = setup_env()
    # evaluator = evaluator_builder()
    # [train_loader, valid_loader] = construct_loader()
    # model = space_builder().to(device)
    # init sampler
    if cfg.SPACE.NAME == 'nasbench201':
        num_choice, layers = 5, 6
        model = SUPPORTED_SPACES["spos_nb201"]().to(device)
    elif cfg.SPACE.NAME == 'nasbenchmacro':
        num_choice, layers = 3, 8
        model = SUPPORTED_SPACES["nasbenchmacro"]().to(device)
    elif cfg.SPACE.NAME == 'nasbench301':
        num_choice, layers = 8, 14*2 # 14 edges 8 cell (6 normal + 2 reduce)
        model = SUPPORTED_SPACES["darts"]().to(device)
        # raise NotImplementedError
    elif cfg.SPACE.NAME == 'transbench101_micro':
        num_choice, layers = 4, 6
        model = SUPPORTED_SPACES["transbench101_micro"]().to(device)
    elif cfg.SPACE.NAME in ["nasbench1shot1_1", "nasbench1shot1_2", "nasbench1shot1_3"]:
        model = SUPPORTED_SPACES[cfg.SPACE.NAME]().to(device)
        num_choice, layers = model.category, None
    
    dataset_api = get_dataset_api(cfg.SPACE.NAME, cfg.LOADER.DATASET, cfg)
    
    criterion = criterion_builder().to(device)
    optimizer = optimizer_builder("SGD", model.parameters())
    lr_scheduler = lr_scheduler_builder(optimizer)


    if cfg.SEARCH.method_type == 'SPOS':
        train_sampler = RAND(num_choice, layers)
    elif cfg.SEARCH.method_type == 'FairNAS':
        train_sampler = Fair(num_choice, layers)

    trainer = OneShot_PE_Trainer(
        supernet=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        sample_type='iter'
    )
    trainer.register_iter_sample(train_sampler)
    
    # arch_search_space = arch_space_builder()
    pe = SPOS_PE(cfg, trainer)
    
    pe_eval = PEEvaluator(pe, config=cfg)
    pe_eval.adapt_search_space(model, load_labeled=False, dataset_api=dataset_api)
    # evaluate the predictor
    pe_eval.evaluate()

if __name__ == "__main__":
    main()
