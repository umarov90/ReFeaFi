import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
matplotlib.use("agg")

os.chdir("/home/user/data/DeepRAG/")
matplotlib.rcParams.update({'font.size': 14})
sns.set(style='ticks')
fig, axs = plt.subplots(1,1,figsize=(12,3))

df = pd.read_csv("figures_data/reporter_results.tsv", sep="\t")
p = ["b" for i in range(21)]
p.append("r")
p.append("g")
bp = sns.barplot(x="Sample", y="Relative Light Unit Ratio (Cypridina/Red-firefly)", data=df,
    palette=p, saturation=0.5, linewidth=1,
            edgecolor="0.2", ax=axs)
for item in bp.get_xticklabels():
    item.set_rotation(90)
axs.set(ylabel='Relative Light Unit Ratio',
       title='Luciferase assay')
plt.tight_layout()
axs.xaxis.set_ticks_position('none')
plt.savefig("figures/reporter_bar.png")
