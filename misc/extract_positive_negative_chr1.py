import gc
import math
import pickle
import os
import numpy as np
from Bio.Seq import Seq
import re
from random import randint, shuffle
from scipy import stats

def clean_seq(s):
    ns = s.upper()
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return ns


def extract(chrn, pos, fasta, seq_len, strand):
    half_size = int((seq_len - 1) / 2)
    if (pos - half_size < 0):
        enc_seq = "N" * (half_size - pos) + fasta[chrn][0: pos + half_size + 1]
    elif (pos + half_size + 1 > len(fasta[chrn])):
        enc_seq = fasta[chrn][pos - half_size: len(fasta[chrn])] + "N" * (half_size + 1 - (len(fasta[chrn]) - pos))
    else:
        enc_seq = fasta[chrn][pos - half_size: pos + half_size + 1]
    if strand == "-":
        enc_seq = str(Seq(enc_seq).reverse_complement())
    return enc_seq


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


os.chdir("/home/user/data/DeepRAG")
test_chr = "chr1"
fasta = pickle.load(open("fasta.p", "rb"))
seq = ""
# with open("data/genomes/hg19.fa") as f:
#     for line in f:
#         if line.startswith(">"):
#             if len(seq) != 0:
#                 seq = clean_seq(seq)
#                 fasta[chrn] = seq
#                 print(chrn + " - " + str(len(seq)))
#                 break
#             chrn = line.strip()[1:]
#             try:
#                 chrn = line.strip()[1:]
#             except Exception as e:
#                 pass
#             seq = ""
#         else:
#             seq += line
    # if len(seq) != 0:
    #     seq = clean_seq(seq)
    #     fasta[chrn] = seq
    #     print(chrn + " - " + str(len(seq)))
seq_len = 1001
promoters = []
enhancers = []
negatives = []
reg_elements = []
reg_elements_scores = {}

with open('data/hg19.cage_peak_phase1and2combined_coord.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]  # [3:len(vals[0])]
        strand = vals[5]
        if chrn != test_chr:
            continue
        if strand == "+":
            chrp = int(vals[7]) - 1
        elif strand == "-":
            chrp = int(vals[7]) - 1
            continue
        # if strand != "+":
        #     continue
        seq = extract(chrn, chrp, fasta, seq_len, strand)
        promoters.append(seq)
        reg_elements.append(chrp)
        reg_elements_scores[chrp] = int(vals[4])

overlap = 0
with open('data/human_permissive_enhancers_phase_1_and_2.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        if chrn != test_chr:
            continue
        chrp = int(vals[7]) - 1
        seq = extract(chrn, chrp, fasta, seq_len, strand)
        enhancers.append(seq)
        gt = find_nearest(reg_elements, chrp)
        if abs(chrp - gt) < 500:
            overlap += 1
        # reg_elements.append(chrp)
        reg_elements_scores[chrp] = int(vals[4])

print("Overlap:" + str(overlap))

reg_elements.sort()
while len(negatives) < 50000:
    try:
        rp = randint(1000000, len(fasta[test_chr]) - 1000000)
        gt = find_nearest(reg_elements, rp)
        if abs(rp - gt) > 500:
            seq = extract(test_chr, rp, fasta, seq_len, "+")
            negatives.append(seq)
    except:
        pass

# our_scores = []
# cage_scores = []
# ttss = []
# with open("human_chr1.gff") as file:
#     for line in file:
#         vals = line.split("\t")
#         score = float(vals[5])
#         strand = vals[6]
#         if strand != "+":
#             continue
#         if (vals[2] == "promoter/enhancer"):
#             pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
#         gt = find_nearest(reg_elements, pos)
#         if abs(pos - gt) < 500:
#             seq = extract(test_chr, pos, fasta, seq_len, "+")
#             ttss.append(seq)
#
#
# with open("tss.fa", 'w+') as f:
#     for p in ttss:
#         f.write(">Promoter\n")
#         f.write(p)
#         f.write("\n")
#
# corr = stats.pearsonr(np.asarray(our_scores), np.asarray(cage_scores))[0]

pk = 0
print("Promoters: " + str(len(promoters)))
with open("promoters.fa", 'w+') as f:
    for p in promoters:
        f.write(">Promoter" + str(pk) + "\n")
        pk += 1
        f.write(p)
        f.write("\n")


print("Enhancers: " + str(len(enhancers)))
with open("enhancers.fa", 'w+') as f:
    for p in enhancers:
        f.write(">Enhancer\n")
        f.write(p)
        f.write("\n")

print("Negatives: " + str(len(negatives)))
with open("negatives.fa", 'w+') as f:
    for p in negatives:
        f.write(">Negative\n")
        f.write(p)
        f.write("\n")
