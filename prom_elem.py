import pickle
import re
import numpy as np

enc_mat = np.append(np.eye(4), [[0, 0, 0, 0]], axis=0)
mapping_pos = dict(zip("ATGCN", range(5)))


def encode(seq):
    seq2 = [mapping_pos[i] for i in seq]
    return enc_mat[seq2]


def tatascore(a, tss):
    tata = [[-1.02, -1.68, 0, -0.28], [-3.05, 0, -2.74, -2.06], [0, -2.28, -4.28, -5.22], [-4.61, 0, -4.61, -3.49],
            [0, -2.34, -3.77, -5.17], [0, -0.52, -4.73, -4.63], [0, -3.65, -2.65, -4.12], [0, -0.37, -1.5, -3.74],
            [-0.01, -1.4, 0, -1.13], [-0.94, -0.97, 0, -0.05], [-0.54, -1.4, -0.09, 0], [-0.48, -0.82, 0, -0.05],
            [-0.48, -0.66, 0, -0.11], [-0.74, -0.54, 0, -0.28], [-0.62, -0.61, 0, -0.4]]
    maxScore = -1000
    maxI = -1000
    for p in range(14):
        seq = a[tss - 39 + p:tss - 39 + 15 + p]
        score = 0
        for i in range(len(tata)):
            for j in range(4):
                score = score + tata[i][j] * seq[i][j]
        if (score > maxScore):
            maxScore = score
            maxI = 39 - p
    return maxScore, maxI


def ccaatscore(a, tss):
    ccat = [[-0.02, 0, -1.46, -0.01], [-0.49, -0.01, -0.24, 0], [-1.19, 0, -1.26, -0.57], [0, -3.16, -0.4, -3.46],
            [-0.61, -1.44, 0, -2.45], [-4.39, -3.99, -4.03, 0], [-4.4, -4, -4.4, 0], [0, -4.37, -4.37, -4.37],
            [0, -1.33, -1.69, -2.45], [-2.12, 0, -2.26, -4.27], [-1.32, -2.84, -0.47, 0], [0, -3.57, -0.81, -2.64]]
    maxScore = -1000
    maxI = -1000
    for p in range(142):
        seq = a[tss - 200 + p:tss - 200 + 12 + p]
        score = 0
        for i in range(len(ccat)):
            for j in range(4):
                score = score + ccat[i][j] * seq[i][j]
        if (score > maxScore):
            maxScore = score
            maxI = 200 - p
    return maxScore, maxI


def inrscore(a):
    inr = [[-1.14, 0, -0.75, -1.16], [-5.26, -5.26, -5.26, 0], [0, -2.74, -5.21, -5.21], [-1.51, -0.29, 0, -0.41],
           [-0.65, 0, -4.56, -0.45], [-0.55, -0.36, -0.86, 0], [-0.91, 0, -0.38, -0.29], [-0.82, 0, -0.65, -0.18]]
    score = 0
    for i in range(len(inr)):
        for j in range(4):
            score = score + inr[i][j] * a[i][j]
    return score


def tctscore(a):
    tct = [[0.08, 0.35, 0.30, 0.27], [0.08, 0.32, 0.17, 0.43], [0.00, 0.00, 0.00, 11.00], [0.07, 0.62, 0.08, 0.24],
           [0.09, 0.32, 0.16, 0.43], [0.11, 0.43, 0.15, 0.30], [0.09, 0.33, 0.22, 0.36], [0.10, 0.28, 0.24, 0.38]]
    score = 0
    for i in range(len(tct)):
        for j in range(4):
            score = score + tct[i][j] * a[i][j]
    return score


def check(seq):
    seq = encode(seq)
    tss = 100
    # cscore, cbp = ccaatscore(seq, tss)
    tscore, tbp = tatascore(seq, tss)
    val = ""
    # if cscore >= -4.54:
    #     val = val + "1,"
    # else:
    #     val = val + "0,"

    if tscore >= -8.16:
        val = val + "1,"
    else:
        val = val + "0,"
    if inrscore(seq[tss - 2:tss + 6]) >= -3.75:
        val = val + "1"
    # elif tctscore(seq[tss - 2:tss + 6]) >= 12.84:
    #     val = val + "0,1"
    else:
        val = val + "0"

    return val


def gc_check(seq):
    gc = (seq.count('G') + seq.count('C')) / (len(seq))
    gc = int(gc * 100)
    return gc


def clean_seq(s):
    s = s.upper()
    pattern = re.compile(r'\s+')
    s = re.sub(pattern, '', s)
    # ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return s


def parse_genome(file):
    print("Parsing fasta")
    fasta = {}
    seq = ""
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0:
                    seq = clean_seq(seq)
                    fasta[chrn] = seq
                    print(chrn + " - " + str(len(seq)))
                chrn = line.strip()[1:]
                seq = ""
                continue
            else:
                seq += line
        if len(seq) != 0:
            seq = clean_seq(seq)
            fasta[chrn] = seq
            print(chrn + " - " + str(len(seq)))
    print("Done")
    pickle.dump(fasta, open("fasta.p", "wb"))
    return fasta
