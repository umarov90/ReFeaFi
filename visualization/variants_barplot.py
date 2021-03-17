import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy import genfromtxt
import pandas as pd
matplotlib.use("agg")

os.chdir("/home/user/data/DeepRAG/")
matplotlib.rcParams.update({'font.size': 14})
sns.set(style='ticks')
fig, axs = plt.subplots(1,1,figsize=(6,3))

db = "Clinvar"
x_labels = ["Random", "Low scoring", "High scoring", "CAGE peaks"]
y = genfromtxt("figures_data/" + db + "_overlap.csv")
bp = sns.barplot(x=x_labels, y=y, saturation=0.5, linewidth=1, edgecolor="0.2", ax=axs)
axs.set(ylabel='Count')
#axs.legend_.set_title(None)
plt.title(db + " overlap", loc='center', fontsize=18)
plt.tight_layout()
axs.xaxis.set_ticks_position('none')
plt.savefig("figures/" + db + "_bar.png")
