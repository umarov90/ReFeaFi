import os
import numpy as np
import math
import common as cm

os.chdir(open("../data_dir").read().strip())


def get_refeafi(fpath):
    min_score = 100
    max_score = -100
    preds = {}
    with open(fpath) as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]
            score = float(vals[5])
            try:
                score = math.log(1 + score / (1.00001 - score))
                if score < min_score:
                    min_score = score
                if score > max_score:
                    max_score = score
                if (vals[2] == "promoter/enhancer"):
                    pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
                    preds.setdefault(chrn, []).append([pos, score])
            except:
                pass
    return preds


def get_results(organism, path, fasta_file):
    refeafi = get_refeafi("predictions/" + organism + ".gff")
    cage = get_cage(path)
    fasta = cm.parse_genome(fasta_file)
    print(organism)
    dtv = []
    for dti in range(100):
        dt = -2.2 + dti * 0.123
        res = compare(cage, refeafi, dt, fasta)
        fpr = 1000000 * (res[0] / res[5])
        dtv.append(str(res[1]) + "," + str(fpr))
        if round(res[1], 2) == 0.50:
            best_line = str(res[1]) + "\t" + str(res[2]) + "\t" + str(res[3]) + "\t" + str(res[4]) + "\t" + str(fpr)
            # print(best_line)
    print(res[5])
    print(res[6])

    with open("figures_data/dtv_refeafi_" + organism + ".csv", 'w+') as f:
        for l in dtv:
            f.write(str(l) + "\n")


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def compare(cage, preds, dt, fasta, margin=500):
    tp = 0.0
    fn = 0.0
    fp = 0.0
    flen = 0
    cnum = 0
    keys = cage.keys() & preds.keys()
    for key in keys:
        flen = flen + len(fasta[key])
        cg = [i[0] for i in cage[key]]
        cnum = cnum + len(cg)
        cg = np.asarray(cg)
        pr = [i[0] for i in preds[key] if i[1] >= dt]
        pr.sort()
        pr = np.asarray(pr)
        if len(pr) > 0:
            for gt in cg:
                v = find_nearest(pr, gt)
                if abs(v - gt) <= margin:
                    tp = tp + 1
                else:
                    fn = fn + 1

        for v in pr:
            gt = find_nearest(cg, v)
            if abs(v - gt) > margin:
                fp = fp + 1
    if tp > 0:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    else:
        recall = 0
        precision = 0
    if (recall + precision) != 0:
        f1 = 2.0 * ((recall * precision) / (recall + precision))
    else:
        f1 = 0

    if tp > 0:
        div = fp / tp
    else:
        div = 0
    return fp, recall, precision, f1, div, flen, cnum


def get_cage(path):
    reg_elements = {}
    with open(path) as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]
            chrp = int(vals[7]) - 1
            val = 1
            reg_elements.setdefault(chrn, []).append([chrp, val])
    for key in reg_elements.keys():
        reg_elements[key].sort(key=lambda x: x[0])
    return reg_elements


get_results("human", "data/hg19.cage_peak_phase1and2combined_coord.bed", "data/genomes/hg19.fa")
get_results("mouse", "data/cage/mm9.cage_peak_phase1and2combined_coord.bed", "data/genomes/mm9.fa")
get_results("chicken", "data/cage/galGal5.cage_peak_coord.bed", "data/genomes/galGal5.fa")
get_results("monkey", "data/cage/rheMac8.cage_peak_coord.bed", "data/genomes/rheMac8.fa")
get_results("rat", "data/cage/rn6.cage_peak_coord.bed", "data/genomes/rn6.fa")
get_results("dog", "data/cage/canFam3.cage_peak_coord.bed", "data/genomes/canFam3.fa")
