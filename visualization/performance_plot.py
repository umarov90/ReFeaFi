import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

os.chdir("/home/user/data/DeepRAG/")
matplotlib.rcParams.update({'font.size': 14})

organism = "human"
deeprag = np.genfromtxt("figures_data/dtv_deeprag_"+organism+".csv", delimiter=',')
ep3 = np.genfromtxt("figures_data/dtv_ep3_"+organism+".csv", delimiter=',')
basenji = np.genfromtxt("figures_data/dtv_basenji7_"+organism+".csv", delimiter=',')
prompredict = np.genfromtxt("figures_data/dtv_prompredict_"+organism+".csv", delimiter=',')


fig, ax = plt.subplots(figsize=(6,4))
ax.plot(deeprag[:,1], deeprag[:,0], '-o', label='ReFeaFi', markersize=0)
ax.plot(ep3[:,1], ep3[:,0], '-o', label='EP3', markersize=0)
ax.plot(prompredict[:,1], prompredict[:,0], '-o', label='PromPredict', markersize=0)
ax.plot(basenji[:,1], basenji[:,0], '-o', label='Basenji', markersize=0)
ax.set(xlabel='FP per million BP', ylabel='Recall',
       title='Human CHR 1 performance')
ax.grid()
ax.legend()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
fig.tight_layout()
plt.savefig("figures/curve_"+organism+".svg")
plt.savefig("figures/curve_"+organism+".png")