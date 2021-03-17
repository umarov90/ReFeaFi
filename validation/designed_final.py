#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
import common as cm
import tensorflow as tf
import numpy as np
import math
import re
from scipy import stats
from Bio.Seq import Seq


half_size = 500
batch_size = 128
scan_step = 1
seq_len = 1001
out = []

os.chdir("/home/user/data/DeepRAG/")
# fasta = cm.parse_genome("data/hg19.fa")
fasta = pickle.load(open("fasta.p", "rb"))

background = {}
background["RPLP0_CE_bg"] = ["chr12", "-", 120638861 - 1, 120639013 - 1]
background["ACTB_CE_bg"] = ["chr7",   "-", 5570183 - 1, 5570335 - 1]
background["C14orf166_CE_bg"] = ["chr14", "+", 52456090 - 1,  52456242 - 1]
our_scores = []
real_scores = []
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
            fa = vals[25].strip()
            if(vals[3] in background.keys()):
                bg = background[vals[3]]
                strand = bg[1]
                real_score = float(vals[23])
                if math.isnan(real_score):
                    continue
                real_scores.append(real_score)

                if (strand == "+"):
                    fa_bg = fasta[bg[0]][bg[2] - (half_size - 114): bg[2] - (half_size - 114) + seq_len]
                    faf = fa_bg[:half_size - 49] + fa + fa_bg[half_size + 115:]
                else:
                    fa_bg = fasta[bg[0]][bg[2] - (half_size - 49): bg[2] - (half_size - 49) + seq_len]
                    fa_bg = str(Seq(fa_bg).reverse_complement())
                    faf = fa_bg[:half_size - 114] + fa + fa_bg[half_size + 50:]

                predict = sess.run(y,
                                   feed_dict={input_x: [cm.encode_seq(faf)], kr: 1.0, in_training_mode: False})
                score = predict[0][0] - predict[0][1]
                score = math.log(1 + score / (1.0001 - score))
                our_scores.append(score)
                out.append(str(real_score) + "," + str(score))

real_scores = np.asarray(real_scores)
corr = stats.pearsonr(np.asarray(our_scores), np.asarray(real_scores))[0]
print(corr)
with open("figures_data/synth.csv", 'w+') as f:
    f.write('\n'.join(out))