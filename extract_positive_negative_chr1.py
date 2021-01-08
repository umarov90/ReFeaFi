import gc
import math
import pickle
import os
import numpy as np


def extract(chrn, pos, fasta, seq_len, strand):
    half_size = int((seq_len - 1) / 2)
    if (pos - half_size < 0):
        enc_seq = "N" * (half_size - pos) + fasta[chrn][0: pos + half_size + 1]
    elif (pos + half_size + 1 > len(fasta[chrn])):
        enc_seq = fasta[chrn][pos - half_size: len(fasta[chrn])] + "N" * (half_size + 1 - (len(fasta[chrn]) - pos))
    else:
        enc_seq = fasta[chrn][pos - half_size: pos + half_size + 1]
    return enc_seq


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

os.chdir("/home/user/data/DeepRAG")
fasta = pickle.load(open("fasta.p", "rb"))
seq_len = 1001
promoters = []
enhancers = []
negatives = []
reg_elements = []

with open('data/hg19.cage_peak_phase1and2combined_coord.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]  # [3:len(vals[0])]
        strand = vals[5]
        if chrn != "chr1":
            continue
        chrp = int(vals[7]) - 1
        seq = extract(chrn, chrp, fasta, seq_len, strand)
        promoters.append(seq)
        reg_elements.append(chrp)

with open('data/human_permissive_enhancers_phase_1_and_2.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        if chrn != "chr1":
            continue
        chrp = int(vals[7]) - 1
        seq = extract(chrn, chrp, fasta, seq_len, strand)
        enhancers.append(seq)
        reg_elements.append(chrp)

with open('data/human_permissive_enhancers_phase_1_and_2.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        if chrn != "chr1":
            continue
        chrp = int(vals[7]) - 1
        reg_elements.append(chrp)

reg_elements.sort()
with open("human_chr1_no_shift.gff") as file:
    for line in file:
        vals = line.split("\t")
        score = float(vals[5])
        strand = vals[6]
        if (vals[2] == "promoter/enhancer"):
            pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1

            gt = find_nearest(reg_elements, pos)
            if abs(pos - gt) > 500:
                seq = extract("chr1", pos, fasta, seq_len, strand)
                negatives.append(seq)

with open("promoters.fa", 'w+') as f:
    f.write('\n'.join(promoters))

with open("enhancers.fa", 'w+') as f:
    f.write('\n'.join(enhancers))

with open("negatives.fa", 'w+') as f:
    f.write('\n'.join(negatives))
