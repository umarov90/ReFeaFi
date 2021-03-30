import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir(open("../data_dir").read().strip())
matplotlib.rcParams.update({'font.size': 14})

group = "Enhancers" # Promoters
fig, axs = plt.subplots(figsize=(6,4))
data = np.genfromtxt("figures_data/importance_"+group+".csv", delimiter=',')
data = data[0:1000]

g = sns.lineplot(data=data, ax=axs)
g.set_xticks([0, 500, 1000]) # <--- set the ticks first
g.set_xticklabels(["-500", "+1", "+500"])
axs.axhline(0, ls='-', c="lightgray", lw=2)
axs.set(xlabel='Position', ylabel='Score impact')
g.set(ylim=(0, 0.2))
plt.title("Average effect on score (" + group + ")")
fig.tight_layout()
plt.savefig("figures/importance_"+group+".png")
plt.savefig("figures/importance_"+group+".svg")