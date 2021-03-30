import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

e = np.genfromtxt("landscape_long.csv", delimiter=',')
e = e - e.min()
#e = e[::-1]
e = e[0:10000]
fig, ax = plt.subplots(figsize=(12, 6)) #frameon=False
ax = sns.lineplot(data=e)
#ax = sns.regplot(x=epk[:,0], y=epk[:,1], label='RNAstructure')
#ax.legend()
#ax.fill_between(range(len(e)),e, color="blue", alpha=0.1)
plt.axhline(y=0.5, color='r', linestyle='--')
#plt.axhline(y=0.0, color='black', linestyle='-')
#ax.axis('off')
plt.savefig("land_long.svg", transparent=True)