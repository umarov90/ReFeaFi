#!/usr/bin/env python
import math
import os
import pickle
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn3, venn2
from liftover import get_lifter
converter = get_lifter('hg38', 'hg19')

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
            gff.setdefault(chrn, []).append(converter[chrn][pos][0][1])
    return gff


def compare(preds, vcf, reg_elements, dt):
    overlap = 0
    count = 0
    for key, value in preds.items():
        reg = [i[0] for i in reg_elements[key]]
        reg.sort()
        value.sort(key=lambda x: x[1])
        reg = np.asarray(reg)
        cg = vcf[key]
        cg = np.asarray(cg)
        pr = [i[0] for i in value if abs(i[0] - find_nearest(reg, i[0])) > 500]
        num = len(reg)
        if dt > 0:
            pr = pr[-num:]
        else:
            pr = pr[:num]
        count = count + len(pr)
        pr.sort()
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
    for key, value in preds.items():
        reg = [i[0] for i in reg_elements[key]]
        reg.sort()
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
            count = count + 1
    for key, value in vcf.items():
        value.sort()
    return vcf, count


def parse_tsv(file):
    df = pd.read_csv(file, sep="\t")
    snps = {}
    count = 0
    for index, row in df.iterrows():
        chrn = str(row['CHR_ID']).split(";")
        vals = str(row['CHR_POS']).split(";")
        for i in range(len(vals)):
            if not vals[i].isnumeric():
                continue
            pos = int(vals[i])
            chr = "chr" + chrn[i]
            try:
                snps.setdefault(chr, []).append(converter[chrn[i]][pos][0][1])
                count = count + 1
            except:
                continue
    return snps, count


def parse_list(file):
    a = np.loadtxt(file, delimiter="\t")
    for i in range(len(a)):
        try:
            a[i] = converter["1"][a[i]][0][1]
        except:
            continue
    snps = {}
    snps["chr1"] = a
    count = len(snps["chr1"])
    for key, value in snps.items():
        value.sort()
    return snps, count


data_folder = "/home/user/data/DeepRAG/"
os.chdir(data_folder)

np.random.seed(2504)
half_size = 200
reg_elements = {}
reg_count = 0

input_file = "human.gff"

with open("data/hg19.cage_peak_phase1and2combined_coord.bed") as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        val = int(vals[4])
        # if val < 5000:
        #     continue
        chrp = int(vals[7]) - 1
        reg_elements.setdefault(chrn, []).append([chrp, val])
        reg_count = reg_count + 1

with open("data/human_permissive_enhancers_phase_1_and_2.bed") as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        val = int(vals[4])
        # if val < 5000:
        #     continue
        chrp = int(vals[7]) - 1
        reg_elements.setdefault(chrn, []).append([chrp, val])
        reg_count = reg_count + 1


print("Done ")
snps_clinvar, count_snps_clinvar = parse_vcf("data/clinvar.vcf")
snps_gwas, count_snps_gwas = parse_tsv("data/gwas_catalog_v1.0-associations_e100_r2020-07-06.tsv")
print("Clinvar: " + str(count_snps_clinvar))
print("GWAS: " + str(count_snps_gwas))
print("CAGE: " + str(reg_count))
# snps_gwas, count_snps_gwas = parse_list("all_proxies_snipa.csv")
# fasta = pickle.load(open("fasta.p", "rb"))

margin = 50
preds = parse_gct(input_file)

fig, axs = plt.subplots(1, figsize=(8,8))

overlap_clinvar_high, count_clinvar_high = compare(preds, snps_clinvar, reg_elements, 0.99)
overlap_clinvar_low, count_clinvar_low = compare(preds, snps_clinvar, reg_elements, -0.99)

# venn3(subsets=(count_snps_clinvar,  count_clinvar_high, overlap_clinvar_high, count_clinvar_low, overlap_clinvar_low, 0, 0),
#       set_labels=("Clinvar", "High Scoring", "Low Scoring"), alpha=0.5, ax=axs)
# plt.savefig("venn_clinvar.png")
# plt.close(fig)
# fig, axs = plt.subplots(1, figsize=(8,8))

overlap_gwas_high, count_gwas_high = compare(preds, snps_gwas, reg_elements, 0.99)
overlap_gwas_low, count_gwas_low = compare(preds, snps_gwas, reg_elements, -0.99)

# venn3(subsets=(count_snps_gwas, count_gwas_high, overlap_gwas_high, count_gwas_low, overlap_gwas_low, 0, 0),
#       set_labels=("GWAS", "High Scoring", "Low Scoring"), alpha=0.5, ax=axs)
# plt.savefig("venn_gwas.png")
# plt.close(fig)
# fig, axs = plt.subplots(1, figsize=(8,8))

overlap_clinvar_cage, count_clinvar_cage = compare_base(snps_clinvar, reg_elements)
overlap_gwas_cage, count_gwas_cage = compare_base(snps_gwas, reg_elements)


# venn3(subsets=(len(reg_elements["chr1"]), len(snps_clinvar["chr1"]), overlap3, len(snps_gwas["chr1"]), overlap4, 0, 0),
#       set_labels=("CAGE", "Clinvar", "GWAS"), alpha=0.5, ax=axs)
# plt.savefig("venn_cage.png")
# plt.close(fig)

print("-"*20)

pr = {}
for key, value in preds.items():
    pr[key] = []
    while len(pr[key]) < len(reg_elements[key]):
        rn = random.randint(0, reg_elements[key][-1][0])
        pr[key].append([rn, 1])
    pr[key].sort()

print("-"*20)

overlap_clinvar_random, count_clinvar_random = compare_base(snps_clinvar, pr)
overlap_gwas_random, count_gwas_random = compare_base(snps_gwas, pr)

clinvar_res = [overlap_clinvar_random, overlap_clinvar_low, overlap_clinvar_high, overlap_clinvar_cage]
gwas_res = [overlap_gwas_random, overlap_gwas_low, overlap_gwas_high, overlap_gwas_cage]

with open("figures_data/Clinvar_overlap.csv", 'w+') as f:
    for v in clinvar_res:
        f.write(str(v) + '\n')

with open("figures_data/GWAS_overlap.csv", 'w+') as f:
    for v in gwas_res:
        f.write(str(v) + '\n')

