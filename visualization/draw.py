import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir("/home/user/data/DeepRAG/analysis")
organism = "human"
deeprag = np.genfromtxt("dtv_deeprag_"+organism+".csv", delimiter=',')
deeprag_noshift = np.genfromtxt("dtv_deeprag_noshift_"+organism+".csv", delimiter=',')
baseline = np.genfromtxt("dtv_baseline_"+organism+".csv", delimiter=',')
#dag = np.genfromtxt("dtv_deeprag_"+organism+".csv", delimiter=',')
#dag2 = np.genfromtxt("dtv_deepag2_"+organism+".csv", delimiter=',')
#pi = np.append(pi, [1, 1])
#pi = np.append(pi, [0, 0])


fig, ax = plt.subplots()
ax.plot(deeprag[:,1], deeprag[:,0], '-o', label='DeepRAG', markersize=0)
ax.plot(deeprag_noshift[:,1], deeprag_noshift[:,0], '-o', label='DeepRAG_noshift', markersize=0)
ax.plot(baseline[:,1], baseline[:,0], '-o', label='baseline', markersize=0)
ax.set(xlabel='FP per million BP', ylabel='Recall',
       title='Human CHR 1 performance')
ax.grid()
ax.legend()

plt.savefig("curve_"+organism+".svg")
plt.savefig("curve_"+organism+".png")