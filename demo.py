import matplotlib.pyplot as plt
import re
import numpy as np
import json

# plt.clf()
# with open('./exp/PT_nasbench201_7/errors.json', 'r') as f:
#     res = json.load(f)
# kdts = []
# for j in res[2:]:
#     kdts.append(j['kendalltau'])
# plt.plot(kdts, '-o')
# plt.ylabel('kendalltau')
# plt.xlabel('epoch')
# plt.savefig('out.png')

plt.clf()
with open('./hb.log', 'r') as f:
    res_hb = f.read()
res = np.minimum.accumulate(list(map(float, re.findall('top1_err: ([\d\.]+)', res_hb))))
x = []
y = []
last = np.inf
for i in range(len(res)):
    if res[i] < last:
        x.append(i)
        y.append(res[i])
        last = y[-1]

plt.plot(x, y, '-o', label='hb')

with open('./hb_zc.log', 'r') as f:
    res_hb = f.read()
res2 = np.minimum.accumulate(list(map(float, re.findall('top1_err: ([\d\.]+)', res_hb))))
x = []
y = []
last = np.inf
for i in range(len(res2)):
    if res2[i] < last:
        x.append(i)
        y.append(res2[i])
        last = y[-1]
plt.plot(x, y, '-o', label='zc-hb')

with open('./hb_zc_fix.log', 'r') as f:
    res_hb = f.read()
res2 = np.minimum.accumulate(list(map(float, re.findall('top1_err: ([\d\.]+)', res_hb))))
x = []
y = []
last = np.inf
for i in range(len(res2)):
    if res2[i] < last:
        x.append(i)
        y.append(res2[i])
        last = y[-1]
plt.plot(x, y, '-o', label='zc-hb_fix')

plt.yscale('log')

plt.legend()
plt.savefig('out.png')