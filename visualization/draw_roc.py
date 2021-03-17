import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")
from sklearn.metrics import roc_curve, auc

os.chdir("/home/user/data/DeepRAG/")
matplotlib.rcParams.update({'font.size': 14})
ground_truth = np.genfromtxt('figures_data/ground_truth.csv', delimiter=',')
total_scores = np.genfromtxt('figures_data/total_scores.csv', delimiter=',')
fpr, tpr, thresholds = roc_curve(ground_truth, total_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
lw = 2
plt.plot(fpr, tpr,
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("VISTA enhancers performance")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/vista.png")