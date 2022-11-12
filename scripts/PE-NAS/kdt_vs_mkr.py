import seaborn as sns
colors = (sns.color_palette("hls", n_colors=13))
colors[4] = 'black'
colors[5] = 'grey'

import glob
import pickle, os
from matplotlib import pyplot as plt
import json
import pandas as pd
from collections import defaultdict
from scipy.stats import kendalltau, pearsonr
plt.rc('font', family='Times New Roman')
import numpy as np
from ConfigSpace import ConfigurationSpace
import ConfigSpace as CS
from ConfigSpace.conditions import InCondition, LessThanCondition
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
# import matplotlib.pyplot as plt

from xbbo.search_algorithm.de_optimizer import DE


tkw = dict(size=4, width=1.5,labelsize=15)
import itertools



with open('singlespace_dfs_201.pkl','rb') as f:
    precs, kdts, gt, preds, methods = pickle.load(f)

np.random.seed(0)
# methods = []
plt.clf()
fig, axs = plt.subplots(1,2, figsize=(9,3),sharey=True)
ress = []
# kdts = []
precs = []
archs = list(itertools.product(*[range(5) for i in range(6)]))
map_gt = dict(zip(archs, gt))
gt_idxs = np.argsort(gt)


def de(map_gt, map_pe, sample_num, seed):
    configuration_space = ConfigurationSpace(seed)
    for i in range(6):
        configuration_space.add_hyperparameter(CategoricalHyperparameter(str(i), [0,1,2,3,4]))
    # define search space
    cs = configuration_space
    # define black box optimizer
    hpopt = DE(space=cs, rng=seed,llambda=50)
    # ---- Begin BO-loop ----
    for i in range(sample_num):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate 
        d = trial_list[0].config_dict
        l = []
        for i in range(6):
            l.append(d[str(i)])
        obs = -map_pe[tuple(l)]
        # observe
        trial_list[0].add_observe_value(obs)
        hpopt.observe(trial_list=trial_list)

    d = hpopt.trials.get_best()[1]
    l = []
    for i in range(6):
        l.append(d[str(i)])
    return map_gt[tuple(l)]

with open('singlespace_dfs_201.pkl','rb') as f:
    precs, kdts, gt, preds, methods = pickle.load(f)

np.random.seed(0)
# methods = []
plt.clf()
fig, axs = plt.subplots(1,2, sharey=True)
ress = []
# kdts = []
precs = []
archs = list(itertools.product(*[range(5) for i in range(6)]))
map_gt = dict(zip(archs, gt))
gt_idxs = np.argsort(gt)

for i, method in enumerate(methods):
    # prec = precs[i]
    kdt = kdts[i]

    pe = preds[i]
    pe_idxs = np.argsort(pe)
    rank = np.argsort(pe_idxs)
    prec = rank[gt_idxs[-1]] / len(gt) 
    precs.append(prec)
    res = []
    map_pe = dict(zip(archs, pe))
    for r in range(5):
        # res.append(random_search(gt, pe, sample_num=1000))
        seed = np.random.randint(1e5)
        res.append(de(map_gt, map_pe, 1000, seed))

    ress.append(np.mean(res))

    axs[0].scatter(kdt, ress[-1], label=method, color=colors[i], alpha=0.6)

    axs[1].scatter(prec, ress[-1], label=method, color=colors[i], alpha=0.6)

df = pd.DataFrame({'kdts':kdts, 'acc':ress})
sns.regplot(x="kdts", y="acc", data=df, ax=axs[0], scatter=False)
df = pd.DataFrame({'precs':precs, 'acc':ress})
sns.regplot(x="precs", y="acc", data=df, ax=axs[1], scatter=False)
axs[0].set_ylim([84.6, 90.9])

axs[0].set_xlabel(r"kendall's ${\tau}$", fontsize=17, fontweight='bold')
axs[1].set_xlabel('1 - MKR', fontsize=17, fontweight='bold')
axs[0].set_ylabel("Accuracy", fontsize=17,fontweight='bold')
axs[1].set_ylabel(None)
axs[0].tick_params(**tkw)
axs[1].tick_params(**tkw)
axs[0].grid(linestyle='-.', linewidth=1)
axs[1].grid(linestyle='-.', linewidth=1)
axs[0].spines['bottom'].set_linewidth(2)
axs[0].spines['left'].set_linewidth(2)
axs[0].spines['right'].set_linewidth(2)
axs[0].spines['top'].set_linewidth(2)
axs[1].spines['bottom'].set_linewidth(2)
axs[1].spines['left'].set_linewidth(2)
axs[1].spines['right'].set_linewidth(2)
axs[1].spines['top'].set_linewidth(2)
handles, labels = axs[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5,fontsize=14, labelspacing=0.2, handletextpad=0.5, columnspacing=0.2)
# lgd = fig.legend(bbox_to_anchor=(1.5, 0.7), ncol=2,loc='upper center')
plt.tight_layout()
plt.savefig('./kdt_vs_mkr.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
print(kendalltau(kdts,ress))
print(kendalltau(precs, ress))
print(pearsonr(kdts,ress))
print(pearsonr(precs, ress))
