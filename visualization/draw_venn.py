import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import string
from matplotlib_venn import venn3, venn3_circles

data = []
data.append(np.genfromtxt("a.csv", delimiter=',') / 10)

set1 = set(['A', 'B', 'C', 'D'])
set2 = set(['B', 'C', 'D', 'E'])
set3 = set(['C', 'D',' E', 'F', 'G'])

venn3([set1, set2, set3], ('Set1', 'Set2', 'Set3'))

plt.savefig("venn3.svg", transparent=True)