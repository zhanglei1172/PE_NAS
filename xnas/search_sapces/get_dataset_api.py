import os
import pickle

import numpy as np

from xnas.core.utils import get_project_root
from xnas.search_sapces.utils_asr import from_folder


"""
This file loads any dataset files or api's needed by the Trainer or PredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""

def get_transbench101_api(dataset=None, config=None):
    datafile_path = os.path.join(get_project_root(), "data", "transnas-bench_v10141024.pth")
    assert os.path.exists(datafile_path), f"Could not fine {datafile_path}. Please download transnas-bench_v10141024.pth\
 from https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101"
    from xnas.spaces.Transbench101.api import TransNASBenchAPI

    # from xnas.search_spaces import TransNASBenchAPI
    api = TransNASBenchAPI(datafile_path)
    return {'api': api, 'task': dataset}


def get_nasbench101_api(dataset=None, config=None):
    # load nasbench101
    import xnas.utils.nb101_api as api

    nb101_datapath = os.path.join(get_project_root(), "data", "nasbench_only108.pkl")
    assert os.path.exists(nb101_datapath), f"Could not find {nb101_datapath}. Please download nasbench_only108.pk \
from https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa"

    nb101_data = api.NASBench(nb101_datapath)
    return {"api": api, "nb101_data": nb101_data}


def get_nasbench201_api(dataset=None, config=None):
    """
    Load the NAS-Bench-201 data
    """
    datafiles = {
        'cifar10': 'nb201_cifar10_full_training.pickle',
        'cifar100': 'nb201_cifar100_full_training.pickle',
        'ImageNet16-120': 'nb201_ImageNet16_full_training.pickle'
    }

    datafile_path = os.path.join(get_project_root(), 'data', datafiles[dataset])
    assert os.path.exists(datafile_path), f'Could not find {datafile_path}. Please download {datafiles[dataset]} from \
https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa'

    with open(datafile_path, 'rb') as f:
        data = pickle.load(f)

    return {"nb201_data": data}


def get_darts_api(dataset=None, config=None):
    # Paths to v1.0 model files and data file.
    # path './data/nb301models/data/nb_models'
    nb_models_path = os.path.join(config.BENCHMARK.NB301PATH, "data", "nb_models")
    nb301_model_path=os.path.join(nb_models_path, "xgb_v1.0")
    nb301_runtime_path=os.path.join(nb_models_path, "lgb_runtime_v1.0")
    data_path = os.path.join(get_project_root(), "data", "nb301_full_training.pickle")

    models_not_found_msg = "Please download v1.0 models from \
https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510"

    # Verify the model and data files exist
    assert os.path.exists(nb_models_path), f"Could not find {nb_models_path}. {models_not_found_msg}"
    assert os.path.exists(nb301_model_path), f"Could not find {nb301_model_path}. {models_not_found_msg}"
    assert os.path.exists(nb301_runtime_path), f"Could not find {nb301_runtime_path}. {models_not_found_msg}"
    assert os.path.isfile(data_path), f"Could not find {data_path}. Please download nb301_full_training.pickle from\
        https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa?usp=sharing"

    # Load the nb301 performance and runtime models
    try:
        import nasbench301
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('No module named \'nasbench301\'. \
            Please install nasbench301 from https://github.com/crwhite14/nasbench301')

    performance_model = nasbench301.load_ensemble(nb301_model_path)
    runtime_model = nasbench301.load_ensemble(nb301_runtime_path)

    with open(data_path, "rb") as f:
        nb301_data = pickle.load(f)
        nb301_arches = list(nb301_data.keys())

    nb301_model = [performance_model, runtime_model]

    return {
        "nb301_data": nb301_data,
        "nb301_arches": nb301_arches,
        "nb301_model": nb301_model,
    }


def get_nlp_api(dataset=None, config=None):
    nb_model_path = os.path.join(get_project_root(), "data", "nbnlp_v01")
    nb_nlp_data_path = os.path.join(get_project_root(), "data", "nb_nlp.pickle")

    data_not_found_msg = "Please download the files from https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa"

    assert os.path.exists(nb_model_path), f"Could not find {nb_model_path}. {data_not_found_msg}"
    assert os.path.exists(nb_nlp_data_path), f"Could not find {nb_nlp_data_path}. {data_not_found_msg}"

    # Load the NAS-Bench-NLP data
    with open(nb_nlp_data_path, "rb") as f:
        nlp_data = pickle.load(f)
    nlp_arches = list(nlp_data.keys())

    # Load the NAS-Bench-NLP11 performance model
    try:
        import nasbench301
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('No module named \'nasbench301\'. \
            Please install nasbench301 from https://github.com/crwhite14/nasbench301')

    performance_model = nasbench301.load_ensemble(nb_model_path)

    return {
        "nlp_data": nlp_data,
        "nlp_arches": nlp_arches,
        "nlp_model":performance_model,
    }


def get_asr_api(dataset=None, config=None):
    # Load the NAS-Bench-ASR data
    d = from_folder(os.path.join(get_project_root(), 'data'),
                    include_static_info=True)

    return {
        'asr_data': d,
    }

def get_natsbenchsize_api(dataset=None, config=None):
    from nats_bench import create

    # Create the API for size search space
    api = create(None, 'sss', fast_mode=True, verbose=True)
    return api

def get_1shot1_api(dataset=None, config=None):
    # from xnas.search_space.cellbased_1shot1_ops import 
    # from xnas.spaces.NASBench1Shot1.ops import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3, OUTPUT_NODE

    from nasbench import api

    # nasbench1shot1_path = 'data/nasbench_full.tfrecord'
    # nasbench = api.NASBench(nasbench1shot1_path)
    nb101_datapath = os.path.join(get_project_root(), "data", "nasbench_only108.tfrecord")
    assert os.path.exists(nb101_datapath), f"Could not find {nb101_datapath}. Please download nasbench_only108.tfrecord"

    nb101_data = api.NASBench(nb101_datapath) 
    return {"nb101_data": nb101_data, 'api':api}
    # current_best = np.argmax(theta, axis=1)
    # config = ConfigSpace.Configuration(
    #     search_space.search_space.get_configuration_space(), vector=current_best)
    # adjacency_matrix, node_list = search_space.search_space.convert_config_to_nasbench_format(
    #     config)
    # node_list = [INPUT, *node_list, OUTPUT] if search_space.search_space.search_space_number == 3 else [
    #     INPUT, *node_list, CONV1X1, OUTPUT]
    # adjacency_list = adjacency_matrix.astype(np.int).tolist()
    # model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
    # nasbench_data = nasbench.query(model_spec, epochs=108)
    

def get_dataset_api(search_space=None, dataset=None, config=None):

    if search_space == "nasbench101":
        return get_nasbench101_api(dataset=dataset, config=config)

    elif search_space == "nasbench201":
        return get_nasbench201_api(dataset=dataset, config=config)

    elif search_space in ("darts", "nasbench301"):
        return get_darts_api(dataset=dataset, config=config)

    elif search_space == "nlp":
        return get_nlp_api(dataset=dataset, config=config)

    elif search_space in ['transbench101', 'transbench101_micro', 'transbench101_macro']:
        return get_transbench101_api(dataset=dataset, config=config)

    elif search_space == "asr":
        return get_asr_api(dataset=dataset, config=config)

    elif search_space == 'natsbenchsize':
        return get_natsbenchsize_api(dataset=dataset, config=config)

    elif search_space == "test":
        return None
    
    elif search_space == "nasbenchmacro":
        from xnas.evaluations.NASBenchMacro.evaluate import data
        return data

    elif search_space in ["nasbench1shot1_1","nasbench1shot1_2","nasbench1shot1_3"]:
        return get_1shot1_api(dataset=dataset, config=config)
    else:
        print(search_space)
        raise NotImplementedError()

