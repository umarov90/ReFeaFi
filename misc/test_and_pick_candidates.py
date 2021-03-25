#!/usr/bin/env python
import numpy as np
import sys
import re
import math
import os
import pyBigWig

from misc import prom_elem


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def find_distance_special(array, value):
    best = 100000000000
    for region in array:
        if region[0] < value < region[1]:
            return 0
        else:
            d = min(abs(region[0] - value), abs(region[1] - value))
            if d < best:
                best = d
    return best


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
    bw = pyBigWig.open("data/hg19.100way.phastCons.bw")
    with open(file) as file:
        for line in file:
            vals = line.split("\t")
            pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
            chrn = vals[0]
            score = float(vals[5])
            score = math.log(score / (1 - score))
            strand = vals[6]
            preds.setdefault(chrn + strand, []).append([pos, score])
            cons = bw.stats(chrn, pos - 100, pos + 101)
            cons = cons[0]
            if cons is None:
                cons = 0
            meta[chrn + ":" + str(pos)] = [score, cons, line]

    return preds, meta


def sort_positions(pos_dict):
    for key, value in pos_dict.items():
        value.sort()

data_folder = "/home/user/data/DeepRAG/"
os.chdir(data_folder)

np.random.seed(2504)
half_size = 500
reg_elements = {}
input_file = sys.argv[1]

with open("data/hg19.cage_peak_phase1and2combined_coord.bed") as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        pos = int(vals[7]) - 1
        strand = vals[5]
        reg_elements.setdefault(chrn + strand, []).append(pos)

with open("data/human_permissive_enhancers_phase_1_and_2.bed") as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        pos = int(vals[7]) - 1
        reg_elements.setdefault(chrn + ".", []).append(pos)

permissive_cage = {}
with open("data/DPIcluster_hg19_20120116.permissive_set.GencodeV10_annotated.osc") as file:
    for line in file:
        if line.startswith("#"):
            continue
        if line.startswith("chrom"):
            continue
        vals = line.split("\t")
        chrn = vals[0]
        pos = int(vals[7]) - 1
        strand = vals[5]
        permissive_cage.setdefault(chrn + strand, []).append(pos)

gencode = {}
with open("data/gencode.v34lift37.annotation.gff3") as file:
    for line in file:
        if line.startswith("#"):
            continue
        vals = line.split("\t")
        chrn = vals[0]
        start = int(vals[3]) - 1
        end = int(vals[4]) - 1
        strand = vals[6]
        gencode.setdefault(chrn + strand, []).append([start, end])

fasta = prom_elem.parse_genome("data/hg19.fa")
#fasta = pickle.load(open("fasta.p", "rb"))

preds, meta = parse_gct(input_file)

sort_positions(permissive_cage)
sort_positions(gencode)
sort_positions(reg_elements)
sort_positions(preds)

margin = 500
dt = 6

tp = 0.0
fn = 0.0
fp = 0.0
candidates_list = []


for key, value in preds.items():
    print(key)
    strand = key[-1]
    if strand == ".":
        continue
    chrn = key[:-1]
    cg = np.asarray([i for i in reg_elements[key]])
    # Means it is enhancer
    if key in permissive_cage.keys():
        pcg = np.asarray([i for i in permissive_cage[key]])
        rcg = cg
    else:
        pcg = [i for i in permissive_cage[chrn + "+"]]
        pcg.extend([i for i in permissive_cage[chrn + "-"]])
        pcg.sort()
        pcg = np.asarray(pcg)
        rcg = [i for i in reg_elements[chrn + "+"]]
        rcg.extend([i for i in reg_elements[chrn + "-"]])
        rcg.extend([i for i in reg_elements[chrn + "."]])
        rcg.sort()
        rcg = np.asarray(rcg)

    if key in gencode.keys():
        gcg = np.asarray([i for i in gencode[key]])
    else:
        gcg = [i for i in gencode[chrn + "+"]]
        gcg.extend([i for i in gencode[chrn + "-"]])
        gcg = np.asarray(gcg)
    pr = np.asarray([i[0] for i in value if 0.5 > i[1] > 0.1])# 0.5 > i[1] > 0.1
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
            mkey = chrn + ":" + str(v)
            seq = fasta[chrn][v - 100:v + 101]
            seq = revcomp(seq, strand)
            p_dist = -1
            if len(pcg) > 0:
                p_dist = abs(v - find_nearest(pcg, v))
            g_dist = -1
            if len(gcg) > 0:
                g_dist = find_distance_special(gcg, v)
            candidates_list.append(chrn + ", " + str(v) + ", " + str(abs(v - find_nearest(rcg, v)))
                                   + ", " + str(p_dist)
                                   + ", " + str(g_dist)
                                   + ", " + str(meta[mkey][0])
                                   + ", " + strand + ", " + str(meta[mkey][1])
                                   + ", " + prom_elem.check(seq) + ", " + str(prom_elem.gc_check(seq))
                                   + ", " + seq)
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
print("FP: " + str(fp))
print("Recall: " + str(recall))
print("Precision: " + str(precision))
print("F1 Score: " + str(f1))
with open("negative_candidates.csv", 'w+') as f:
    f.write("Chr,Position,Closest peak,Closest permissive,Closest gencode,Score,Strand,Conservation,TATA,Inr,GC%,"
            "Sequence\n")
    f.write('\n'.join(candidates_list))
