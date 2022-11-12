from xnas.evaluations.pe_evaluation import PEEvaluator
from xnas.algorithms.zero_cost_pe.zero_cost import ZeroCost
# from xnas.core.builder import arch_space_builder
import xnas.core.config as config
import xnas.logger.logging as logging
from xnas.core.config import cfg
from xnas.core.builder import *
from xnas.search_sapces.get_dataset_api import get_dataset_api


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main():
    device = setup_env()
    # evaluator = evaluator_builder()
    model = space_builder().to(device)
    
    dataset_api = get_dataset_api(cfg.SPACE.NAME, cfg.LOADER.DATASET, cfg)
    # arch_search_space = arch_space_builder()
    pe = ZeroCost(cfg, method_type=cfg.SEARCH.method_type)
    pe_eval = PEEvaluator(pe,config=cfg)
    pe_eval.adapt_search_space(model, load_labeled=False, dataset_api=dataset_api)
    # evaluate the predictor
    pe_eval.evaluate()

if __name__ == "__main__":
    main()
