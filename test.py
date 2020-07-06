#!/usr/bin/env python
import numpy as np
import sys
import re
import math
import pickle
import os


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
    preds = {}
    meta = {}
    with open(file) as file:
        for line in file:
            vals = line.split("\t")
            pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
            chrn = vals[0]
            preds.setdefault(chrn,[]).append([pos, float(vals[5])])
            if chrn + str(pos) in meta.keys():
                print("what")
            meta[chrn + ":" + str(pos)] = [vals[5], vals[6], line]

    for key, value in preds.items():
        value.sort()

    return preds, meta

def parse_gff(file):
    gff = {}
    with open(file) as file:
        for line in file:
            if line.startswith("#"):
                continue
            vals = line.split("\t")
            if vals[2] != "gene":
                continue
            chrn = re.search('.*0(.*)\.11', vals[0])
            start = int(vals[3])
            end = int(vals[4])
            pos = start + (end - start) / 2
            gff.setdefault(chrn,[]).append(pos)
    return gff


def compare(cage, preds, dt, margin, meta):
    tp = 0.0
    fn = 0.0
    fp = 0.0
    hard_list = []
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
                if float(meta[key + ":" + str(v)][0]) < dt:
                    print("what")
                hard_list.append(key + ", " + str(v) + ", " + str(abs(v - gt)) + ", " + str(meta[key + ":" + str(v)][0])
                                 + ", " + str(meta[key + ":" + str(v)][1]) + ", " + fasta[key][v-50:v + 51])
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
    return fp, recall, precision, f1, hard_list

def parse_vcf(file):
    vcf = {}
    with open(file) as file:
        for line in file:
            if line.startswith("#"):
                continue
            vals = line.split("\t")
            chrn = vals[0]
            pos = vals[0]
            vcf.setdefault(chrn,[]).append(pos)
    return vcf

data_folder = "/home/user/data/DeepRAG/"
os.chdir(data_folder)

np.random.seed(2504)
half_size = 500
reg_elements = {}

input_file = sys.argv[1]

with open("cage.bed") as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        val = 1
        chrp = int(vals[7]) - 1
        reg_elements.setdefault(chrn,[]).append([chrp, val])

with open("enhancers.bed") as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        val = 1
        chrp = int(vals[7]) - 1
        reg_elements.setdefault(chrn,[]).append([chrp, val])

print("Done ")
genes = parse_gff("GRCh37_latest_genomic.gff")
fasta = pickle.load(open("fasta.p", "rb"))
if len(sys.argv) > 3:
    good_chr = sys.argv[3].split(",")
else:
    good_chr = fasta.keys()

margin = 500
dt = 0.99

preds, meta = parse_gct(input_file)

res = compare(reg_elements, preds, dt, margin, meta)
print("FP: " + str(res[0]))
print("Recall: " + str(res[1]))
print("Precision: " + str(res[2]))
print("F1 Score: " + str(res[3]))
with open("hard.csv", 'w+') as f:
    f.write("Chr,Position,Closest peak,Score,Strand,Sequence\n")
    f.write('\n'.join(res[4]))