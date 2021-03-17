import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

os.chdir("/home/user/data/DeepRAG/")
matplotlib.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(figsize=(6,4))
organisms = ["human", "mouse", "rat", "chicken", "dog", "monkey"]
for organism in organisms:
       deeprag = np.genfromtxt("figures_data/dtv_deeprag_"+organism+".csv", delimiter=',')
       ax.plot(deeprag[:,1], deeprag[:,0], '-o', label=organism, markersize=0)

ax.set(xlabel='FP per million BP', ylabel='Recall',
       title='Genome-wide performance')
ax.grid()
ax.legend()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
# plt.savefig("curve_species.svg")
fig.tight_layout()
plt.savefig("figures/curve_species.png")