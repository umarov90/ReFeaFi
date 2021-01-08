import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import string

fig, axs = plt.subplots(1,1,figsize=(3,3))
#axs = axs.flat

data = []
data.append(np.genfromtxt("a.csv", delimiter=',') / 10)
#data.append(np.genfromtxt("mu.csv", delimiter=','))
#data.append(np.genfromtxt("mc.csv", delimiter=','))
#data.append(np.genfromtxt("md.csv", delimiter=','))
g = sns.lineplot(data=data[0], ax=axs, legend=False, dashes=False)   
g.set_xticks([0, 500, 1000]) # <--- set the ticks first
g.set_xticklabels(["-500", "+1", "+500"])

plt.savefig("imp.svg", transparent=True)