import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

os.chdir("/home/user/data/DeepRAG/")
matplotlib.rcParams.update({'font.size': 14})

e = np.genfromtxt("figures_data/synth.csv", delimiter=',')

fig, ax = plt.subplots(figsize=(6,4))
# ax = sns.regplot(x=e[:,0], y=e[:,1])
r, p = stats.pearsonr(e[:,0], e[:,1])
# r = r * r

sns.regplot(x=e[:,1], y=e[:,0],
            ci=None, label="r = {0:.2f}; p = {1:.2e}".format(r, p)).legend(loc="best")

ax.set(xlabel='Predicted score', ylabel='Measured gene expression')
plt.title("Synthetic promoter score prediction")
fig.tight_layout()
plt.savefig("figures/synth.png")
plt.savefig("figures/synth.svg")