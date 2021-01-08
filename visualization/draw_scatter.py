import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

#e = np.genfromtxt("synth.csv", delimiter=',')
epk1 = np.genfromtxt("ppk_nnfold.tsv", delimiter='\t')
epk2 = np.genfromtxt("ppk_probknot.tsv", delimiter='\t')
epk3 = np.genfromtxt("ppk_spot.tsv", delimiter='\t')

fig, ax = plt.subplots()
#ax = sns.regplot(x=e[:,0], y=e[:,1])
ax = sns.regplot(x=epk1[:,0], y=epk1[:,1], label='NNfold')
ax = sns.regplot(x=epk2[:,0], y=epk2[:,1], label='RNAstructure')
ax = sns.regplot(x=epk3[:,0], y=epk3[:,1], label='SPOT-RNA')
ax.plot([0, 1], [0, 1], ls="--", color='gray', transform=ax.transAxes)
ax.set(xlabel='True PK number', ylabel='Predicted PK number')
ax.grid()
ax.legend()
plt.savefig("pk.pdf")