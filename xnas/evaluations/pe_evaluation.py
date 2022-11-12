import codecs
import time
import json
import os
import numpy as np
import copy
import torch
from scipy import stats
from sklearn import metrics
import math
from xnas.core.query_metrics import Metric

import xnas.logger.logging as logging
# from xnas.search_spaces.core.query_metrics import Metric
# from xnas.utils import generate_kfold, cross_validation


class PE:
    def __init__(self, ss_type=None, encoding_type=None):
        self.ss_type = ss_type
        self.encoding_type = encoding_type

    def set_ss_type(self, ss_type):
        self.ss_type = ss_type

    def pre_process(self, **kwargs):
        """
        This is called at the start of the NAS algorithm,
        before any architectures have been queried
        """
        pass

    def pre_compute(self, xtrain, xtest, unlabeled=None):
        """
        This method is used to make batch predictions
        more efficient. Perform a computation on the train/test
        set once (e.g., calculate the Jacobian covariance)
        and then use it for all train_sizes.
        """
        pass

    def fit(self, xtrain, ytrain, info=None):
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """
        pass

    def query(self, xtest, info):
        """
        This can be called any number of times during the NAS algorithm.
        inputs: list of architectures,
                info about the architectures (e.g., training data up to 20 epochs)
        output: predictions for the architectures
        """
        pass

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the pe needs
        extra info to train/query, such as a partial learning curve,
        or hyperparameters of the architecture
        """
        reqs = {
            "requires_partial_lc": False,
            "metric": None,
            "requires_hyperparameters": False,
            "hyperparams": {},
            "unlabeled": False,
            "unlabeled_factor": 0,
        }
        return reqs

    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

    def get_hyperparams(self):
        if hasattr(self, "hyperparams"):
            return self.hyperparams
        else:
            # TODO: set hyperparams (at least to None) for all predictors
            print("no hyperparams set")
            return None

    def reset_hyperparams(self):
        self.hyperparams = None

    def get_hpo_wrapper(self):
        if hasattr(self, "hpo_wrapper"):
            return self.hpo_wrapper
        else:
            # TODO: set hpo_wrapper to a boolean for all predictors
            return None


logger = logging.get_logger(__name__)


class PEEvaluator(object):
    """
    This class will evaluate a chosen pe based on
    correlation and rank correlation metrics, for the given
    initialization times and query times.
    """
    def __init__(self, pe, config=None):

        self.pe = pe
        self.config = config
        self.experiment_type = config.experiment_type

        self.test_arch_num = config.test_arch_num
        self.train_size_single = config.train_size_single
        self.train_size_list = config.train_size_list
        self.fidelity_single = config.fidelity_single
        self.fidelity_list = config.fidelity_list
        self.max_hpo_time = config.max_hpo_time
        self.oneshot_epochs = config.oneshot_epochs if hasattr(
            config, "oneshot_epochs") else None

        self.dataset = config.LOADER.DATASET
        self.metric = Metric.VAL_ACCURACY
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = [config]

        # mutation parameters
        self.uniform_random = config.uniform_random
        self.mutate_pool = 10
        self.num_arches_to_mutate = 5
        self.max_mutation_rate = 3
        self.all_arch_sample = False


    def adapt_search_space(self,
                           search_space,
                           load_labeled,
                           scope=None,
                           dataset_api=None):
        self.search_space = search_space
        # self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.pe.set_ss_type(self.search_space.get_type())
        self.load_labeled = load_labeled
        self.dataset_api = dataset_api

        if self.search_space.get_type() in ("nasbench201", "nasbenchmacro", "nasbench301", "nasbench1shot1_1", "nasbench1shot1_2", "nasbench1shot1_3"):
            if hasattr(self.config, "all_arch") and self.config.all_arch:
                self.all_arch_sample = True
            
        # nasbench101 does not have full learning curves or hyperparameters
        if self.search_space.get_type() in ("nasbench101", "nasbenchmacro","transbench101_micro","nasbench1shot1_1", "nasbench1shot1_2","nasbench1shot1_3", "nasbench1shot1",'nasbench301'):
            self.full_lc = False
            self.hyperparameters = False
        elif self.search_space.get_type() in [
                "nasbench201", "nlp", "transbench101", "asr"
        ]:
            self.full_lc = True
            self.hyperparameters = True
        else:
            raise NotImplementedError(
                "This search space is not yet implemented in PredictorEvaluator."
            )

    def get_full_arch_info(self, arch_hash):
        """
        Given an arch, return the accuracy, train_time,
        and also a dict of extra info if required by the pe
        """
        info_dict = {}
        accuracy = self.search_space.query(arch_hash,
                                           metric=self.metric,
                                           dataset=self.dataset,
                                           dataset_api=self.dataset_api)
        train_time = self.search_space.query(arch_hash,
                                             metric=Metric.TRAIN_TIME,
                                             dataset=self.dataset,
                                             dataset_api=self.dataset_api)
        data_reqs = self.pe.get_data_reqs()
        if data_reqs["requires_partial_lc"] and self.full_lc:
            # add partial learning curve if applicable
            assert self.full_lc, "This pe requires learning curve info"
            if type(data_reqs["metric"]) is list:
                for metric_i in data_reqs["metric"]:
                    metric_lc = self.search_space.query(
                        arch_hash,
                        metric=metric_i,
                        full_lc=True,
                        dataset=self.dataset,
                        dataset_api=self.dataset_api,
                    )
                    info_dict[f"{metric_i.name}_lc"] = metric_lc

            else:
                lc = self.search_space.query(
                    arch_hash,
                    metric=data_reqs["metric"],
                    full_lc=True,
                    dataset=self.dataset,
                    dataset_api=self.dataset_api,
                )
                info_dict["lc"] = lc
            if data_reqs["requires_hyperparameters"]:
                assert (self.hyperparameters
                        ), "This pe requires querying arch hyperparams"
                for hp in data_reqs["hyperparams"]:
                    info_dict[hp] = self.search_space.query(
                        arch_hash,
                        Metric.HP,
                        dataset=self.dataset,
                        dataset_api=self.dataset_api)[hp]
        return accuracy, train_time, info_dict

    def load_arch(self, load_labeled=False, data_size=10, arch_hash_map={}):
        """
        There are two ways to load an architecture.
        load_labeled=False: sample a random architecture from the search space.
        This works on NAS benchmarks where we can query any architecture (nasbench101/201/301)
        load_labeled=True: sample a random architecture from a set of evaluated architectures.
        When we only have data on a subset of the search space (e.g., the set of 5k DARTS
        architectures that have the full training info).

        After we load an architecture, query the final val accuracy.
        If the pe requires extra info such as partial learning curve info, query that too.
        """
        xdata = []
        ydata = []
        info = []
        train_times = []
        while len(xdata) < data_size:
            if not load_labeled:
                arch_hash = self.search_space.sample_random_architecture(
                    dataset_api=self.dataset_api)
            else:
                arch_hash = self.search_space.load_labeled_architecture(
                    dataset_api=self.dataset_api)

            # arch_hash = arch_hash
            arch = arch_hash
            if False:  # removing this for consistency, for now
                continue
            else:
                arch_hash_map[arch_hash] = True

            accuracy, train_time, info_dict = self.get_full_arch_info(
                arch_hash)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        return [xdata, ydata, info, train_times], arch_hash_map

    def load_mutated_test(self, data_size=10, arch_hash_map={}):
        """
        Load a test set not uniformly at random, but by picking some random
        architectures and then mutation the best ones. This better emulates
        distributions in local or mutation-based NAS algorithms.
        """
        assert (self.load_labeled == False
                ), "Mutation is only implemented for load_labeled = False"
        xdata = []
        ydata = []
        info = []
        train_times = []

        # step 1: create a large pool of architectures
        while len(xdata) < self.mutate_pool:
            arch = self.search_space.clone()
            arch.sample_random_architecture(dataset_api=self.dataset_api)
            arch_hash = arch.get_hash()
            if arch_hash in arch_hash_map:
                continue
            else:
                arch_hash_map[arch_hash] = True
            accuracy, train_time, info_dict = self.get_full_arch_info(arch)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        # step 2: prune the pool down to the top 5 architectures
        indices = np.flip(np.argsort(ydata))[:self.num_arches_to_mutate]
        xdata = [xdata[i] for i in indices]
        ydata = [ydata[i] for i in indices]
        info = [info[i] for i in indices]
        train_times = [train_times[i] for i in indices]

        # step 3: mutate the top architectures to generate the full list
        while len(xdata) < data_size:
            idx = np.random.choice(self.num_arches_to_mutate)
            arch = xdata[idx].clone()
            mutation_factor = np.random.choice(self.max_mutation_rate) + 1
            for i in range(mutation_factor):
                new_arch = self.search_space.clone()
                new_arch.mutate(arch, dataset_api=self.dataset_api)
                arch = new_arch

            arch_hash = arch.get_hash()
            if arch_hash in arch_hash_map:
                continue
            else:
                arch_hash_map[arch_hash] = True
            accuracy, train_time, info_dict = self.get_full_arch_info(arch)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        return [xdata, ydata, info, train_times], arch_hash_map

    def load_mutated_arch(self, data_size=10, arch_hash_map={}, test_data=[]):
        """
        Load a training set not uniformly at random, but by picking architectures
        from the test set and mutating the best ones. There is still no overlap
        between the training and test sets. This better emulates local or
        mutation-based NAS algorithms.
        """
        assert (self.load_labeled == False
                ), "Mutation is only implemented for load_labeled = False"
        xdata = []
        ydata = []
        info = []
        train_times = []

        while len(xdata) < data_size:
            idx = np.random.choice(len(test_data[0]))
            parent = test_data[0][idx]
            arch = self.search_space.clone()
            arch.mutate(parent, dataset_api=self.dataset_api)
            arch_hash = arch.get_hash()
            if arch_hash in arch_hash_map:
                continue
            else:
                arch_hash_map[arch_hash] = True
            accuracy, train_time, info_dict = self.get_full_arch_info(arch)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        return [xdata, ydata, info, train_times], arch_hash_map

    def single_evaluate(self, train_data, test_data, fidelity,**kwargs):
        """
        Evaluate the pe for a single (train_data / fidelity) pair
        """
        xtrain, ytrain, train_info, train_times = train_data
        xtest, ytest, test_info, _ = test_data
        train_size = len(xtrain)

        data_reqs = self.pe.get_data_reqs()

        logger.info("Fit the pe")
        if data_reqs["requires_partial_lc"]:
            """
            todo: distinguish between predictors that need LC info
            at training time vs test time
            """
            train_info = copy.deepcopy(train_info)
            test_info = copy.deepcopy(test_info)
            for info_dict in train_info:
                lc_related_keys = [
                    key for key in info_dict.keys() if "lc" in key
                ]
                for lc_key in lc_related_keys:
                    info_dict[lc_key] = info_dict[lc_key][:fidelity]

            for info_dict in test_info:
                lc_related_keys = [
                    key for key in info_dict.keys() if "lc" in key
                ]
                for lc_key in lc_related_keys:
                    info_dict[lc_key] = info_dict[lc_key][:fidelity]

        self.pe.reset_hyperparams()
        fit_time_start = time.time()
        cv_score = 0
        if (self.max_hpo_time > 0 and len(xtrain) >= 10
                and self.pe.get_hpo_wrapper()):

            # run cross-validation (for model-based predictors)
            hyperparams, cv_score = self.run_hpo(
                xtrain,
                ytrain,
                train_info,
                start_time=fit_time_start,
                metric="kendalltau",
            )
            self.pe.set_hyperparams(hyperparams)

        self.pe.fit(xtrain, ytrain, train_info)
        hyperparams = self.pe.get_hyperparams()

        fit_time_end = time.time()
        test_pred = self.pe.query(xtest, test_info, **kwargs)
        query_time_end = time.time()

        # If the pe is an ensemble, take the mean
        if len(test_pred.shape) > 1:
            test_pred = np.mean(test_pred, axis=0)

        logger.info("Compute evaluation metrics")
        results_dict = self.compare(ytest, test_pred)
        results_dict["train_size"] = train_size
        results_dict["fidelity"] = fidelity
        results_dict["train_time"] = np.sum(train_times)
        results_dict["fit_time"] = fit_time_end - fit_time_start
        results_dict["query_time"] = (query_time_end -
                                      fit_time_end) / len(xtest)
        if hyperparams:
            for key in hyperparams:
                results_dict["hp_" + key] = hyperparams[key]
        results_dict["cv_score"] = cv_score

        # note: specific code for zero-cost experiments:
        method_type = None
        if hasattr(self.pe, 'method_type'):
            method_type = self.pe.method_type
        print("dataset: {}, pe: {}, spearman {}".format(
            self.dataset, method_type, np.round(results_dict["spearman"], 4)))
        print("full ytest", results_dict["full_ytest"])
        print("full testpred", results_dict["full_testpred"])
        # end specific code for zero-cost experiments.

        # print entire results dict:
        print_string = ""
        for key in results_dict:
            if type(results_dict[key]) not in [str, set, bool]:
                # todo: serialize other types
                print_string += key + ": {}, ".format(
                    np.round(results_dict[key], 4))
        logger.info(print_string)
        self.results.append(results_dict)
        """
        Todo: query_time currently does not include the time taken to train a partial learning curve
        """
    def load_all_arch(self,arch_hash_map={}):
        xdata = []
        ydata = []
        info = []
        train_times = []
        # arch_hash_map = {}

        arch_hashs = self.search_space.get_all_architecture(self.dataset_api)
        for arch_hash in arch_hashs:
            # arch_hash = arch_hash
            arch = arch_hash

            arch_hash_map[arch_hash] = True

            accuracy, train_time, info_dict = self.get_full_arch_info(
                arch_hash)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        return [xdata, ydata, info, train_times], arch_hash_map

    def evaluate(self):
        logger.info("Load the training set")
        max_train_size = self.train_size_single

        if self.experiment_type in ["vary_train_size", "vary_both"]:
            max_train_size = self.train_size_list[-1]


        arch_hash_map = {}
        if self.uniform_random:
            full_train_data, arch_hash_map = self.load_arch(
                load_labeled=self.load_labeled,
                data_size=max_train_size,
                arch_hash_map=arch_hash_map,
            )
        else:
            raise NotImplementedError
        # else:
        #     full_train_data, arch_hash_map = self.load_mutated_arch(
        #         data_size=max_train_size,
        #         arch_hash_map=arch_hash_map,
        #         test_data=test_data,
        #     )
        logger.info("Load the test set")

        # else:
        #     test_data, arch_hash_map = self.load_mutated_test(
        #         data_size=self.test_arch_num)
        if self.all_arch_sample:
            test_data, arch_hash_map = self.load_all_arch(arch_hash_map=arch_hash_map)
        elif self.uniform_random:
            test_data, arch_hash_map = self.load_arch(
                load_labeled=self.load_labeled, data_size=self.test_arch_num, arch_hash_map=arch_hash_map)
        else:
            raise NotImplementedError

        pre_process_st_time = time.time()
        if self.experiment_type == "data":
            
            self.pe.pre_process(train_ratio=self.config.train_ratio)
        else:
            self.pe.pre_process()
            
        pre_process_ed_time = time.time()

        # if the pe requires unlabeled data (e.g. SemiNAS), generate it:
        reqs = self.pe.get_data_reqs()
        unlabeled_data = None
        if reqs["unlabeled"]:
            logger.info("Load unlabeled data")
            unlabeled_size = max_train_size * reqs["unlabeled_factor"]
            [unlabeled_data, _, _, _], _ = self.load_arch(
                load_labeled=self.load_labeled,
                data_size=unlabeled_size,
                arch_hash_map=arch_hash_map,
            )

        # some of the predictors use a pre-computation step to save time in batch experiments:
        pre_compute_st_time = time.time()
        self.pe.pre_compute(full_train_data[0], test_data[0], unlabeled_data)
        pre_compute_ed_time = time.time()

        if hasattr(self.config, "only_train") and self.config.only_train:
            return None
        self.results.append({
            "pe.pre_process_time":
            pre_process_ed_time - pre_process_st_time,
            "pe.pre_compute_time":
            pre_compute_ed_time - pre_compute_st_time,
        })

        if self.experiment_type == "single":
            train_size = self.train_size_single
            fidelity = self.fidelity_single
            self.single_evaluate(full_train_data, test_data, fidelity=fidelity)

        elif self.experiment_type == "vary_train_size":
            fidelity = self.fidelity_single
            for train_size in self.train_size_list:
                train_data = [data[:train_size] for data in full_train_data]
                self.single_evaluate(train_data, test_data, fidelity=fidelity)

        elif self.experiment_type == "vary_fidelity":
            train_size = self.train_size_single
            for fidelity in self.fidelity_list:
                self.single_evaluate(full_train_data,
                                     test_data,
                                     fidelity=fidelity)

        elif self.experiment_type == "vary_both":
            for train_size in self.train_size_list:
                train_data = [data[:train_size] for data in full_train_data]

                for fidelity in self.fidelity_list:
                    self.single_evaluate(train_data,
                                         test_data,
                                         fidelity=fidelity)
        elif self.experiment_type == "vary_epochs":
            train_size = self.train_size_single
            fidelity = self.config.OPTIM.MAX_EPOCH + 1
            self.epochs_evaluate(full_train_data,
                                         test_data,
                                         self.oneshot_epochs,
                                         fidelity=fidelity)
        elif self.experiment_type == "epoch":
            train_size = self.train_size_single
            fidelity = self.config.OPTIM.MAX_EPOCH + 1
            self.single_evaluate(full_train_data,
                                         test_data,
                                         fidelity=fidelity,end_epoch=self.config.OPTIM.MAX_EPOCH)
        elif self.experiment_type == "data":
            train_size = self.train_size_single
            fidelity = self.config.OPTIM.MAX_EPOCH + 1
            self.single_evaluate(full_train_data,
                                         test_data,
                                         fidelity=fidelity)
        else:
            raise NotImplementedError()

        self._log_to_json()
        return self.results

    def compare(self, ytest, test_pred):
        ytest = np.array(ytest)
        test_pred = np.array(test_pred)
        METRICS = [
            "mae",
            "rmse",
            "pearson",
            "spearman",
            "kendalltau",
            "weightedtau",
            "kt_2dec",
            "kt_1dec",
            "precision_10",
            "precision_20",
            "full_ytest",
            "full_testpred",
        ]
        metrics_dict = {}
        # TODO Zen score may be -inf ?
        mask = np.isneginf(test_pred)
        test_pred[mask] = np.min(test_pred[~mask]) - 1
        try:
            metrics_dict["mae"] = np.mean(abs(test_pred - ytest))
            metrics_dict["rmse"] = metrics.mean_squared_error(ytest,
                                                              test_pred,
                                                              squared=False)
            metrics_dict["pearson"] = np.abs(
                np.corrcoef(ytest, test_pred)[1, 0])
            metrics_dict["spearman"] = stats.spearmanr(ytest, test_pred)[0]
            metrics_dict["kendalltau"] = stats.kendalltau(ytest, test_pred)[0]
            metrics_dict["weightedtau"] = stats.weightedtau(ytest, test_pred)[0]
            metrics_dict["kt_2dec"] = stats.kendalltau(
                ytest, np.round(test_pred, decimals=2))[0]
            metrics_dict["kt_1dec"] = stats.kendalltau(
                ytest, np.round(test_pred, decimals=1))[0]
            for k in [10, 20]:
                top_ytest = np.array([
                    y > sorted(ytest)[max(-len(ytest), -k - 1)] for y in ytest
                ])
                top_test_pred = np.array([
                    y > sorted(test_pred)[max(-len(test_pred), -k - 1)]
                    for y in test_pred
                ])
                metrics_dict["precision_{}".format(k)] = (
                    sum(top_ytest & top_test_pred) / k)
            metrics_dict["full_ytest"] = list(ytest)
            metrics_dict["full_testpred"] = list(test_pred)

        except:
            for metric in METRICS:
                metrics_dict[metric] = float("nan")
        if np.isnan(metrics_dict["pearson"]) or not np.isfinite(
                metrics_dict["pearson"]):
            logger.info(
                "Error when computing metrics. ytest and test_pred are:")
            logger.info(ytest)
            logger.info(test_pred)

        return metrics_dict

    def run_hpo(
        self,
        xtrain,
        ytrain,
        train_info,
        start_time,
        metric="kendalltau",
        max_iters=5000,
    ):
        logger.info(f"Starting cross validation")
        n_train = len(xtrain)
        split_indices = generate_kfold(n_train, 3)
        # todo: try to run this without copying the pe
        pe = copy.deepcopy(self.pe)

        best_score = -1e6
        best_hyperparams = None

        t = 0
        while t < max_iters:
            t += 1
            hyperparams = pe.set_random_hyperparams()
            cv_score = cross_validation(xtrain, ytrain, pe, split_indices,
                                        metric)
            if np.isnan(cv_score) or cv_score < 0:
                # todo: this will not work for mae/rmse
                cv_score = 0

            if cv_score > best_score or t == 0:
                best_hyperparams = hyperparams
                best_score = cv_score
                logger.info(
                    f"new best score={cv_score}, hparams = {hyperparams}")

            if (time.time() - start_time
                ) > self.max_hpo_time * (len(xtrain) / 1000) + 20:
                # we always give at least 20 seconds, and the time scales with train_size
                break

        if math.isnan(best_score):
            best_hyperparams = pe.default_hyperparams

        logger.info(f"Finished {t} rounds")
        logger.info(
            f"Best hyperparams = {best_hyperparams} Score = {best_score}")
        self.pe.hyperparams = best_hyperparams

        return best_hyperparams.copy(), best_score

    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.OUT_DIR):
            os.makedirs(self.config.OUT_DIR)
        with codecs.open(os.path.join(self.config.OUT_DIR, "errors.json"),
                         "w",
                         encoding="utf-8") as file:
            for res in self.results:
                for key, value in res.items():
                    if type(value) == np.int32 or type(value) == np.int64:
                        res[key] = int(value)
                    if type(value) == np.float32 or type(value) == np.float64:
                        res[key] = float(value)

            json.dump(self.results, file, separators=(",", ":"))

    def epochs_evaluate(self, train_data, test_data, epochs, fidelity):
        """
        Evaluate the pe for a single (train_data / fidelity) pair
        """
        xtrain, ytrain, train_info, train_times = train_data
        xtest, ytest, test_info, _ = test_data
        train_size = len(xtrain)

        data_reqs = self.pe.get_data_reqs()

        logger.info("Fit the pe")
        if data_reqs["requires_partial_lc"]:
            """
            todo: distinguish between predictors that need LC info
            at training time vs test time
            """
            train_info = copy.deepcopy(train_info)
            test_info = copy.deepcopy(test_info)
            for info_dict in train_info:
                lc_related_keys = [
                    key for key in info_dict.keys() if "lc" in key
                ]
                for lc_key in lc_related_keys:
                    info_dict[lc_key] = info_dict[lc_key][:fidelity]

            for info_dict in test_info:
                lc_related_keys = [
                    key for key in info_dict.keys() if "lc" in key
                ]
                for lc_key in lc_related_keys:
                    info_dict[lc_key] = info_dict[lc_key][:fidelity]

        self.pe.reset_hyperparams()
        fit_time_start = time.time()
        cv_score = 0
        if (self.max_hpo_time > 0 and len(xtrain) >= 10
                and self.pe.get_hpo_wrapper()):

            # run cross-validation (for model-based predictors)
            hyperparams, cv_score = self.run_hpo(
                xtrain,
                ytrain,
                train_info,
                start_time=fit_time_start,
                metric="kendalltau",
            )
            self.pe.set_hyperparams(hyperparams)

        self.pe.fit(xtrain, ytrain, train_info)
        hyperparams = self.pe.get_hyperparams()

        fit_time_end = time.time()
        test_preds = self.pe.query_every_epoch(xtest, test_info, epochs)
        query_time_end = time.time()

        for test_pred in test_preds:
            # If the pe is an ensemble, take the mean
            if len(test_pred.shape) > 1:
                test_pred = np.mean(test_pred, axis=0)

            logger.info("Compute evaluation metrics")
            results_dict = self.compare(ytest, test_pred)
            results_dict["train_size"] = train_size
            results_dict["fidelity"] = fidelity
            results_dict["train_time"] = np.sum(train_times)
            results_dict["fit_time"] = fit_time_end - fit_time_start
            results_dict["query_time"] = (query_time_end -
                                          fit_time_end) / len(xtest)
            if hyperparams:
                for key in hyperparams:
                    results_dict["hp_" + key] = hyperparams[key]
            results_dict["cv_score"] = cv_score

            # note: specific code for zero-cost experiments:
            method_type = None
            if hasattr(self.pe, 'method_type'):
                method_type = self.pe.method_type
            print("dataset: {}, pe: {}, spearman {}".format(
                self.dataset, method_type, np.round(results_dict["spearman"],
                                                    4)))
            print("full ytest", results_dict["full_ytest"])
            print("full testpred", results_dict["full_testpred"])
            # end specific code for zero-cost experiments.

            # print entire results dict:
            print_string = ""
            for key in results_dict:
                if type(results_dict[key]) not in [str, set, bool]:
                    # todo: serialize other types
                    print_string += key + ": {}, ".format(
                        np.round(results_dict[key], 4))
            logger.info(print_string)
            self.results.append(results_dict)
            """
            Todo: query_time currently does not include the time taken to train a partial learning curve
            """
