import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy import genfromtxt
matplotlib.use("agg")

os.chdir(open("../data_dir").read().strip())
matplotlib.rcParams.update({'font.size': 14})
sns.set(style='ticks')
fig, axs = plt.subplots(1,1,figsize=(6,3))

db = "GWAS" # Clinvar
x_labels = ["Random", "Low scoring", "High scoring", "CAGE peaks"]
y = genfromtxt("figures_data/" + db + "_overlap.csv")
bp = sns.barplot(x=x_labels, y=y, saturation=0.5, linewidth=1, edgecolor="0.2", ax=axs)
axs.set(ylabel='Count')
plt.title(db + " overlap", loc='center', fontsize=18)
plt.tight_layout()
axs.xaxis.set_ticks_position('none')
plt.savefig("figures/" + db + "_bar.png")
plt.savefig("figures/" + db + "_bar.svg")