import gc
import pickle
import os
import numpy as np

enc_mat = np.append(np.eye(4),
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                     [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], axis=0)
enc_mat = enc_mat.astype(np.bool)
mapping_pos = dict(zip("ACGTRYSWKMBDHVN", range(15)))


def encode(chrn, pos, fasta, seq_len):
    half_size = int((seq_len - 1) / 2)
    if (pos - half_size < 0):
        enc_seq = "N" * (half_size - pos) + fasta[chrn][0: pos + half_size + 1]
    elif (pos + half_size + 1 > len(fasta[chrn])):
        enc_seq = fasta[chrn][pos - half_size: len(fasta[chrn])] + "N" * (half_size + 1 - (len(fasta[chrn]) - pos))
    else:
        enc_seq = fasta[chrn][pos - half_size: pos + half_size + 1]

    try:
        seq2 = [mapping_pos[i] for i in enc_seq]
        return enc_mat[seq2]
    except:
        print(enc_seq)
        return None


data_folder = "/home/user/data/DeepRAG/"
os.chdir(data_folder)
good_chr = ["chrX", "chrY"]
for i in range(2, 23):
    good_chr.append('chr' + str(i))

fasta = pickle.load(open("fasta.p", "rb"))
ga = pickle.load(open("ga.p", "rb"))
# out_dir = sys.argv[1]
new_negs = []
preds = {}
pred_count = 0
encode_error = 0
seen_keys = []
seq_len = 1201
# All predictions in this file are considered to be negative
with open('human.gff') as file:
    for line in file:
        #try:
        vals = line.split("\t")
        chrn = vals[0]
        if chrn == "1":
            continue
        cl = vals[2]
        if cl == "promoter/enhancer":
            pred_count = pred_count + 1
        else:
            print("entry which is not promoter or enhancer is discovered")
            continue
        pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
        if chrn not in seen_keys:
            seen_keys.append(chrn)
            print(chrn + " " + str(len(ga[chrn])))
            print("Empty positions: " + str(np.count_nonzero(ga[chrn] == 0)))

        if sum(ga[chrn][pos - 300: pos + 300 + 1]) == 0:
            seq_mat = encode(chrn, pos, fasta, seq_len)
            if seq_mat is None:
                encode_error = encode_error + 1
            else:
                new_negs.append([seq_mat, [False, True]])
                ga[chrn][pos] = 100
        #except Exception as e:
        #    print(e)

print("Predictions: " + str(pred_count))
print("Encode error: " + str(encode_error))
print("Found " + str(len(new_negs)) + " negatives")
#exit()
x_train = pickle.load(open("x_train.p", "rb"))
y_train = pickle.load(open("y_train.p", "rb"))

for d in new_negs:
    x_train.append(d[0])
    y_train.append(d[1])

del fasta
gc.collect()
pickle.dump(x_train, open("x_train.p", "wb"))
pickle.dump(y_train, open("y_train.p", "wb"))
pickle.dump(ga, open("ga.p", "wb"))
print("Done")
