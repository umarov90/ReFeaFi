#!/usr/bin/env python
import numpy as np
import sys
import re
import math
import pickle


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def is_close(v, a_list, margin):
    nv = find_nearest(a_list, v)
    if (abs(nv - v) < margin):
        return True
    return False


def revcomp(enc_seq, strand):
    if (strand == "-"):
        enc_seq = enc_seq[::-1]
        rep = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        enc_seq = pattern.sub(lambda m: rep[re.escape(m.group(0))], enc_seq)
    return enc_seq


def clean_seq(s):
    s = s.upper()
    pattern = re.compile(r'\s+')
    s = re.sub(pattern, '', s)
    # ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return s


def parse_gct(file):
    pred_promoters = {}
    pred_enhancers = {}
    with open(file) as file:
        for line in file:
            vals = line.split("\t")
            pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
            strand = vals[6]
            chrn = vals[0]
            ck = chrn + strand
            if vals[2] == "promoter":
                pred_promoters.setdefault(ck,[]).append([pos, float(vals[5])])
            elif vals[2] == "enhancer":
                pred_enhancers.setdefault(chrn + '.',[]).append([pos, float(vals[5])])

    for key, value in pred_promoters.items():
        value.sort()

    for key, value in pred_enhancers.items():
        value.sort()

    return pred_promoters, pred_enhancers


def compare(cage, preds, dt, margin=500):
    tp = 0.0
    fn = 0.0
    fp = 0.0
    for key, value in preds.items():
        cg = [i[0] for i in cage[key]]
        cg = np.asarray(cg)
        pr = [i[0] for i in value if i[1] >= dt]
        pr = np.asarray(pr)
        if (len(pr) > 0):
            for gt in cg:
                v = find_nearest(pr, gt)
                if (abs(v - gt) <= margin):
                    tp = tp + 1
                else:
                    fn = fn + 1

        for v in pr:
            gt = find_nearest(cg, v)
            if (abs(v - gt) > margin):
                fp = fp + 1
    if (tp > 0):
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
    else:
        recall = 0
        precision = 0
    if ((recall + precision) != 0):
        f1 = 2.0 * ((recall * precision) / (recall + precision))
    else:
        f1 = 0
    return fp, recall, precision, f1


np.random.seed(2504)
half_size = 500
good_chr = ["chr1"]
#good_chr = ["chrX", "chrY"]
#for i in range(2, 23):
#    good_chr.append("chr" + str(i))
promoters = {}
enhancers = {}

input_file = sys.argv[1]
ground_truth = sys.argv[2]

print("Parsing bed at " + ground_truth)
with open(ground_truth) as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        val = 1
        chrp = int(vals[7]) - 1
        ck = chrn + strand
        if strand == ".":
            enhancers.setdefault(ck,[]).append([chrp, val])
        else:
            promoters.setdefault(ck,[]).append([chrp, val])

print("Done ")
fasta = pickle.load(open("fasta.p", "rb"))

if len(sys.argv) > 3:
    good_chr = sys.argv[3].split(",")
else:
    good_chr = fasta.keys()

margin = 500
dt = 0.5

pred_promoters, pred_enhancers = parse_gct(input_file)

print("Promoters")
res = compare(promoters, pred_promoters, dt)
print("FP: " + str(res[0]))
print("Recall: " + str(res[1]))
print("Precision: " + str(res[2]))
print("F1 Score: " + str(res[3]))

print("Enahncers")
res = compare(enhancers, pred_enhancers, dt)
print("FP: " + str(res[0]))
print("Recall: " + str(res[1]))
print("Precision: " + str(res[2]))
print("F1 Score: " + str(res[3]))
