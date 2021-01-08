#!/usr/bin/env python
import math
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn3, venn2


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
    with open(file) as file:
        for line in file:
            vals = line.split("\t")
            pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
            chrn = vals[0]
            preds.setdefault(chrn, []).append([pos, float(vals[5])])

    for key, value in preds.items():
        value.sort()

    return preds


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
            gff.setdefault(chrn, []).append(pos)
    return gff


def compare(preds, vcf, reg_elements, dt):
    overlap = 0
    count = 0
    for key, value in preds.items():
        reg = [i[0] for i in reg_elements[key]]
        reg = np.asarray(reg)
        cg = vcf[key]
        cg = np.asarray(cg)
        if dt > 0:
            pr = [i[0] for i in value if i[1] >= dt and abs(i[0] - find_nearest(reg, i[0])) > margin][:5000]
        else:
            pr = [i[0] for i in value if i[1] < -dt and abs(i[0] - find_nearest(reg, i[0])) > margin][:5000]
        count = count + len(pr)
        pr = np.asarray(pr)
        if (len(pr) > 0):
            for gt in cg:
                v = find_nearest(pr, gt)
                if abs(v - gt) <= margin:
                    # pr = np.delete(pr, np.argwhere(pr == v))
                    overlap = overlap + 1

    return overlap, count


def compare_base(vcf, reg_elements):
    overlap = 0
    count = 0
    key = "chr1"
    reg = [i[0] for i in reg_elements[key]]
    count = count + len(reg)
    reg = np.asarray(reg)
    cg = vcf[key]
    cg = np.asarray(cg)
    for gt in cg:
        v = find_nearest(reg, gt)
        if abs(v - gt) <= margin:
            overlap = overlap + 1

    return overlap, count


def parse_vcf(file):
    vcf = {}
    count = 0
    with open(file) as file:
        for line in file:
            if line.startswith("#"):
                continue
            vals = line.split("\t")
            chrn = "chr" + vals[0]
            pos = int(vals[1])
            vcf.setdefault(chrn, []).append(pos)
            if chrn == "chr1":
                count = count + 1
    for key, value in vcf.items():
        value.sort()
    return vcf, count


def parse_tsv(file):
    df = pd.read_csv(file, sep="\t")
    snps = {}
    count = 0
    for index, row in df.iterrows():
        chrn = "chr" + str(row['CHR_ID'])
        spos = str(row['CHR_POS']).split(";")[0]
        if not spos.isnumeric():
            continue
        pos = int(spos)
        snps.setdefault(chrn, []).append(pos)
        if chrn == "chr1":
            count = count + 1
    for key, value in snps.items():
        value.sort()
    return snps, count


def parse_list(file):
    a = np.loadtxt(file, delimiter="\t")
    a = a.flatten()
    a = list(set(a.tolist()))
    snps = {}
    snps["chr1"] = a
    count = len(snps["chr1"])
    for key, value in snps.items():
        value.sort()
    return snps, count


data_folder = "/home/user/data/DeepRAG/"
os.chdir(data_folder)

np.random.seed(2504)
half_size = 500
reg_elements = {}

input_file = sys.argv[1]

with open("data/cage.bed") as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        val = 1
        chrp = int(vals[7]) - 1
        reg_elements.setdefault(chrn, []).append([chrp, val])

with open("data/enhancers.bed") as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        val = 1
        chrp = int(vals[7]) - 1
        reg_elements.setdefault(chrn, []).append([chrp, val])

with open("data/DPIcluster_hg19_20120116.permissive_set.GencodeV10_annotated.osc") as file:
    for line in file:
        if line.startswith("#"):
            continue
        if line.startswith("chrom"):
            continue
        vals = line.split("\t")
        chrn = vals[0]
        chrp = int(vals[7]) - 1
        reg_elements.setdefault(chrn, []).append([chrp, 1])

print("Done ")
# snps, count_snps = parse_vcf("clinvar.vcf")
# snps, count_snps = parse_tsv("gwas_catalog_v1.0-associations_e100_r2020-07-06.tsv")
snps, count_snps = parse_list("all_proxies.tsv")
fasta = pickle.load(open("../fasta.p", "rb"))
if len(sys.argv) > 3:
    good_chr = sys.argv[3].split(",")
else:
    good_chr = fasta.keys()

margin = 500
dt = 0.01

preds = parse_gct(input_file)

overlap1, count1 = compare(preds, snps, reg_elements, 0.99)
overlap2, count2 = compare(preds, snps, reg_elements, -0.99)
overlap3, count3 = compare_base(snps, reg_elements)
print(str(overlap1) + " - " + str(count1))
print(str(overlap2) + " - " + str(count2))

fig, axs = plt.subplots(1, figsize=(8,8))

venn3(subsets=(count_snps, count2, overlap2, count1, overlap1, 0, 0),
      set_labels=("GWAS", "Low Scoring", "High Scoring"), alpha=0.5, ax=axs)

#venn2(subsets=(count_snps, count3, overlap3), set_labels=("GWAS", "CAGE regulatory element"), alpha=0.5, ax=axs[0])

plt.savefig("venn_gwas.png")
plt.close(None)

