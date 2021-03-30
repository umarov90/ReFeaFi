import numpy as np
import math
import re


def brun(sess, x, y, a, keep_prob, in_training_mode):
    preds = []
    batch_size = 256
    number_of_full_batch = int(math.ceil(float(len(a)) / batch_size))
    for i in range(number_of_full_batch):
        preds += list(sess.run(y, feed_dict={x: np.asarray(a[i * batch_size:(i + 1) * batch_size]),
                                             keep_prob: 1.0, in_training_mode: False}))
    return preds


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

enc_mat = np.append(np.eye(4),
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                     [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], axis=0)
enc_mat = enc_mat.astype(np.bool)
mapping_pos = dict(zip("ACGTRYSWKMBDHVN", range(15)))


def encode(fa, pos, seq_len):
    half_size = int((seq_len - 1) / 2)
    if (pos - half_size < 0):
        enc_seq = "N" * (half_size - pos) + fa[0: pos + half_size + 1]
    elif (pos + half_size + 1 > len(fa)):
        enc_seq = fa[pos - half_size: len(fa)] + "N" * (half_size + 1 - (len(fa) - pos))
    else:
        enc_seq = fa[pos - half_size: pos + half_size + 1]

    return encode_seq(enc_seq)


def encode_seq(seq):
    try:
        seq2 = [mapping_pos[i] for i in seq]
        return enc_mat[seq2]
    except:
        print(seq)
        return None


def clean_seq(s):
    ns = s.upper()
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return ns


def parse_genome(g, chr1=False):
    fasta = {}
    seq = ""
    with open(g) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0:
                    seq = clean_seq(seq)
                    fasta[chrn] = seq
                    # print(chrn + " - " + str(len(seq)))
                    if chr1:
                        return fasta
                chrn = line.strip()[1:]
                try:
                    chrn = line.strip()[1:]
                except Exception as e:
                    pass
                seq = ""
            else:
                seq += line
        if len(seq) != 0:
            seq = clean_seq(seq)
            fasta[chrn] = seq
            # print(chrn + " - " + str(len(seq)))
    return fasta


def parse_bed(reg_elements, path):
    with open(path) as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]
            chrp = int(vals[7]) - 1
            reg_elements.setdefault(chrn, []).append(chrp)


def nuc_to_ind(nuc):
    nuc = nuc.upper()
    ind = -1
    if nuc == "A":
        ind = 0
    elif nuc == "C":
        ind = 1
    elif nuc == "G":
        ind = 2
    elif nuc == "T":
        ind = 3
    return ind
