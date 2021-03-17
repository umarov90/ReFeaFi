#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
import common as cm
import tensorflow as tf
import math
from scipy import stats
import numpy as np

half_size = 500
batch_size = 128
seq_len = 1001
scan_step = 1
out = []

os.chdir("/home/user/data/DeepRAG/")
# fasta = cm.parse_genome("data/hg19.fa")
# pickle.dump(fasta, open("fasta.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
fasta = pickle.load(open("fasta.p", "rb"))
out2 = []

background = ["RPLP0_CE_bg", "ACTB_CE_bg", "C14orf166_CE_bg"] 
our_scores = []
real_scores = []
best_pos = []
new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_predict")
    saver = tf.train.Saver()
    saver.restore(sess, "model_predict/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr = tf.get_default_graph().get_tensor_by_name("kr:0")
    in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
    with open('data/Supplemental_Table_S7.tsv') as file:
        next(file)
        for line in file:
            vals = line.split("\t")    
            if(vals[3] in background):
                real_score = float(vals[23])
                if math.isnan(real_score):
                    continue
                real_scores.append(real_score)
                fa = vals[25].strip()
                fa = "N" * half_size + fa + "N" * half_size
                preds = []
                batch = []
                j = half_size
                while j <= len(fa) - half_size - 1:
                    fa_enc = cm.encode(fa, j, seq_len)
                    if len(fa_enc) == seq_len:
                        batch.append(fa_enc)
                    if len(batch) >= batch_size or j + scan_step >= len(fa) - half_size - 1:
                        predict = sess.run(y, feed_dict={input_x: batch, kr: 1.0, in_training_mode: False})
                        scores = [score[0] for score in predict]
                        preds.extend(scores)
                        batch = []
                    j = j + scan_step
                score = max(preds) # - 0.0001
                best_pos.append(np.argmax(preds))
                score = math.log(1 + score / (1.0001 - score))
                our_scores.append(score)
                out.append(line.strip() + "\t" + str(score))
                # fad = fa[half_size-5:half_size+6]
                out2.append(">1")
                out2.append("\n")
                out2.append(fa)
                out2.append("\n")
real_scores = np.asarray(real_scores)
real_scores[np.isnan(real_scores)] = 0
corr = stats.pearsonr(np.asarray(our_scores), np.asarray(real_scores))[0]
print(corr)
print(max(set(best_pos), key=best_pos.count))
# with open("designed_scores2.tsv", 'w+') as f:
#     f.write('\n'.join(out))
#
# with open("designed.fa", 'w+') as f:
#     f.write(''.join(out2))