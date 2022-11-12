from glob import glob
import pickle, os
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from xnas.spaces.DARTS.cnn import convert_darts_hash_to_genotype
plt.rc('font', family='Times New Roman')
import numpy as np
from xnas.algorithms.RMINAS.sampler.sampling import hash2genostr, nb201genostr2array

with open('./data/benchres_1.pkl', 'rb') as f:
    benchmark = pickle.load(f)
def rank(l):
    return np.argsort(np.argsort(-np.array(l)))+1

nb_models_path = os.path.join("data", "nb_models")
nb301_model_path=os.path.join(nb_models_path, "xgb_v1.0")
nb301_runtime_path=os.path.join(nb_models_path, "lgb_runtime_v1.0")
data_path = os.path.join("data", "nb301_full_training.pickle")

with open(data_path, "rb") as f:
    nb301_data = pickle.load(f)
with open('data/nb201_cifar10_full_training.pickle', "rb") as f:
    nb201_data = pickle.load(f)
with open('data/nb201_cifar100_full_training.pickle', "rb") as f:
    tmp = (pickle.load(f))
    nb201_data = {k: dict(v, **tmp[k]) for k, v in nb201_data.items()}
with open('data/nb201_ImageNet16_full_training.pickle', "rb") as f:
    tmp = (pickle.load(f))
    nb201_data = {k: dict(v, **tmp[k]) for k, v in nb201_data.items()}
    # nb301_arches = list(nb301_data.keys())
    
nb201_data = {tuple(nb201genostr2array(k).argmax(-1)): v for k, v in nb201_data.items()}
# reduce policy

OUT_DIR = './exp/zc_oneshot_pt_{}_global'
best_pe_index = []

reduce_ratios_list = [0, 0.75, 0.41, 0.46, 0.13, 0.26, 0.23, 0.05, 0.34, 0.07, 0.36, 0.0, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0]
max_epoch = len(reduce_ratios_list)
plt.clf()
tkw = dict(size=4, width=1.5,labelsize=15)
fig, axs = plt.subplots(2,2, sharex=True)
axs = axs.ravel()
axs[2].set_xlabel('PE Stage', fontsize=17)
axs[3].set_xlabel('PE Stage', fontsize=17)
trace_accs = []
for space in ['201']:
    trace_acc = []
    time_list =[0]
    space_sizes = []
    dataset =  'cifar10'
    gt_key = dataset+'_v' if space == '201' else 'gt'
    gt_dict = dict(zip(benchmark["nasbench{}".format(space)][dataset]['arch'], benchmark["nasbench{}".format(space)][dataset][gt_key]))

    # with open((OUT_DIR+'/pred_arch.pkl').format(space),'rb') as f:
    #     res = pickle.load(f)
    # total_arch_num = len(res['space'])
    # gts = []
    # for arch in res['space']:
    #     gts.append(gt_dict[arch])
    # trace_acc.append([min(gts), max(gts)])
    # print(space, 'zero-cost', res['time'][1]-res['time'][0])
    # time_list.append(res['time'][1]-res['time'][0])
    
    with open((OUT_DIR+'/zc_arch.pkl').format(space),'rb') as f:
        res = pickle.load(f)
    total_arch_num = len(res['space'])
    gts = []
    for arch in res['space']:
        gts.append(gt_dict[arch])
    trace_acc.append([min(gts), max(gts)])
    # print(space, 'zero-cost', res['time'][1]-res['time'][0])
    time_list.append(res['time'][1]-res['time'][0])
    # space_size_list.append(0.8)
    with open((OUT_DIR+'/os_arch.pkl').format(space),'rb') as f:
        res = pickle.load(f)

    gts = []
    for arch in res['space']:
        gts.append(gt_dict[arch])
    trace_acc.append([min(gts), max(gts)])
    zc_reduced_num = len(res['space'])
    time_list.append(res['time'][1]-res['time'][0])
    with open((OUT_DIR+'/pt_arch_1.pkl').format(space),'rb') as f:
        res = pickle.load(f)

    arch_scores_dict = {}
    for i, arch in enumerate(res['space']):
        arch_scores_dict[arch] = [res['scores'][i]]

    os_reduced_num = len(res['space'])
    print(space, 'one-shot', res['time'][1]-res['time'][0])
    space_ratios_list = [1]
    # reduce_ratios_list.insert(1, np.round(1-pred_reduced_arch_num/total_arch_num,2))
    reduce_ratios_list.insert(1, np.round(1-zc_reduced_num/total_arch_num,2))
    reduce_ratios_list.insert(2, np.round(1-os_reduced_num/zc_reduced_num,2))
    for r in reduce_ratios_list[1:]:
        space_ratios_list.append((1-r)*space_ratios_list[-1])
    del reduce_ratios_list[1:4]
    for epoch in range(2, max_epoch):
        if not os.path.exists((OUT_DIR+'/pt_arch_{}.pkl').format(space,epoch)):
            break
        with open((OUT_DIR+'/pt_arch_{}.pkl').format(space,epoch),'rb') as f:
            res = pickle.load(f)
        time_list.append(res['time'][1]-res['time'][0])
        space_sizes.append(len(res['space']))
        for i, arch in enumerate(res['space']):
            arch_scores_dict[arch].append(res['scores'][i])
        # if space_sizes[-1] == 1:
        #     break
        

        gts = []
        for arch in res['space']:
            gts.append(gt_dict[arch])
        trace_acc.append([min(gts), max(gts)])
    trace_accs.append(trace_acc)
    print('Space: {} - Arch: {}'.format(space, res['space']))
    if space == '201':
        gt_dict_v = dict(zip(benchmark["nasbench{}".format(space)][dataset]['arch'], benchmark["nasbench{}".format(space)][dataset][dataset+'_v']))
        gt_dict_t = dict(zip(benchmark["nasbench{}".format(space)][dataset]['arch'], benchmark["nasbench{}".format(space)][dataset][dataset+'_t']))
        nb201_res = (res['space'][0], gt_dict_v[res['space'][0]], gt_dict_t[res['space'][0]])
    # print(space, 'patial training_{}epoch'.format(epoch), res['time'][1]-res['time'][0])
    # space_sizes_1 = int(space_sizes[0] / 0.2)
    # space_sizes_0 = int(space_sizes_1 / 0.2)
    # space_sizes = [space_sizes_0, space_sizes_1] + space_sizes
    space_sizes = [total_arch_num,zc_reduced_num] + space_sizes
    best_pe_index.append(space_sizes.index(1))
    time_list = time_list[:best_pe_index[-1]+1]
    space_ratios_list = space_ratios_list[:best_pe_index[-1]+1]
    space_sizes = space_sizes[:best_pe_index[-1]+1]
    # plt.subplot(221)
    axs[0].plot(range(1, len(time_list)), time_list[1:], '-o',linewidth=1.5)
    if space == '301':
        axs[0].set_ylabel('Time cost (S)', fontsize=17)
        # axs[0].set_title(str.capitalize('Time cost'), fontsize=17)
        axs[0].ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
        axs[0].yaxis.get_offset_text().set(size=14)

    # plt.title('time (s)')
    # plt.subplot(222)
    # plt.plot(np.cumsum(time_list))
    # plt.title('accumulate time (s)')
    axs[1].plot(np.cumsum(time_list), '-o',linewidth=1.5)
    if space == '301':
        axs[1].set_ylabel('Accumulate cost (S)', fontsize=17)
        # axs[1].set_title(str.capitalize('Accumulate cost'), fontsize=17)
        axs[1].ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
        axs[1].yaxis.get_offset_text().set(size=14)
    # plt.subplot(223)
    axs[2].plot(space_ratios_list, '-o',linewidth=1.5)
    # plt.plot(space_ratios_list)
    if space == '301':
        axs[2].set_ylabel('Space ratio', fontsize=17)
        # axs[2].set_title(str.capitalize('Space size'), fontsize=17)
    # plt.title('space size (ratio)')
    # plt.subplot(224)
    # plt.plot(space_sizes, label=space)
    axs[3].plot(space_sizes, '-o',linewidth=1.5, label=space)
    if space == '301':
        axs[3].set_ylabel('Arch num', fontsize=17)
        axs[3].ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
        axs[3].yaxis.get_offset_text().set(size=14)
        # axs[3].set_title(str.capitalize('Space size'), fontsize=17)
    # plt.title('space size (arch num)')
    print("累计时间代价(s)：",np.round(np.cumsum(time_list)[best_pe_index[-1]]))
    print("累计时间代价(h)：",np.round(np.cumsum(time_list)[best_pe_index[-1]]/3600, 2))
    print("最终的精度",np.round(trace_acc[best_pe_index[-1]], 2))
    print("gt的精度",np.round(max(benchmark["nasbench{}".format(space)]['cifar10'][gt_key]),2))
    # plt.suptitle(space)
for n in range(4):
    axs[n].tick_params(**tkw)
    axs[n].grid(linestyle='-.', linewidth=1)
    axs[n].spines['bottom'].set_linewidth(2)
    axs[n].spines['left'].set_linewidth(2)
    axs[n].spines['right'].set_linewidth(2)
    axs[n].spines['top'].set_linewidth(2)
    # axs[n].set_aspect(1)
plt.legend(fontsize=14)
plt.tight_layout()
    
# plt.savefig('global_best_1.pdf',bbox_inches='tight')
plt.savefig('out.png',bbox_inches='tight')

plt.clf()



fig, axs = plt.subplots(1, 1, figsize=(8,4), sharex=True)
axs.set_xlabel('Epoch', fontsize=17, fontweight='bold')
# axs[3].set_xlabel('PE Stage', fontsize=17)
trace_acc = np.array(trace_accs)[0, :max(best_pe_index)+1]
# for i, trace_acc in enumerate(trace_accs):
# trace_acc = np.array(trace_acc)
# plt.subplot(131+i)
# axs[i].fill_between(range(1,len(trace_acc)+1),trace_acc[:,0],trace_acc[:,1],facecolor = 'blue', alpha = 0.5)
# plt.fill_between(range(1,len(trace_acc)+1),trace_acc[:,0],trace_acc[:,1],facecolor = 'blue', alpha = 0.5)

def smooth(array):
    # epoch = np.arange(len(array))
    epochs = []
    res = []
    best = np.inf
    for epoch, a in enumerate(array):
        if a < best:
            res.append(a)
            best = a
            epochs.append(epoch+1)
    return epochs, res
    
# i = 0
# for arch in arch_scores_dict:
#     if i % 10 == 0:
        
#         tmp = np.array(arch_scores_dict[arch])
#         axs.plot(range(1, len(tmp)+1), -tmp,'-')
#     i+=1
# i = 0
for arch in arch_scores_dict:
    # if i % 10 == 0:
        
    tmp = -np.array(arch_scores_dict[arch])
    axs.plot(*smooth(tmp),'-')
    # i+=1
    
axs.set_ylabel('Error', fontsize=17, fontweight='bold')
# plt.xlabel('PE stage')
# plt.ylabel("Acc")
# plt.title(("macro", '201', "301")[i])
# axs.set_title(str.capitalize('nasbench{}'.format(("macro", '201', "301")[i])), fontsize=17)
axs.set_xlim([1, 30])
# for arch in arch_scores_dict[:]
# for n in range(3):
axs.tick_params(**tkw)
axs.grid(linestyle='-.', linewidth=1)
axs.spines['bottom'].set_linewidth(2)
axs.spines['left'].set_linewidth(2)
axs.spines['right'].set_linewidth(2)
axs.spines['top'].set_linewidth(2)
    # axs[n].xaxis.set_major_locator(MultipleLocator(5))
plt.tight_layout()
plt.savefig('pt_demo_plot.pdf',bbox_inches='tight')
plt.savefig('out.png',bbox_inches='tight')

print('nasbench201 res: ', nb201_res)
print(convert_darts_hash_to_genotype(res['space'][0]))
    # archs = benchmark["nasbench{}".format(space)]['cifar10']['arch']

    # archs = benchmark["nasbench{}".format(space)]['cifar10']['arch']