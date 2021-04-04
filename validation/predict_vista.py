#!/usr/bin/env python
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import re
import numpy as np
import tensorflow as tf
import common as cm
import random

seq_len = 1001
half_size = 500

def read_fasta(file):
    seq = ""
    fasta = []
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0:
                    seq = clean_seq(seq)
                    fasta.append(cm.encode_seq(seq))
                seq = ""
            else:
                seq += line
        if len(seq) != 0:
            seq = clean_seq(seq)
            fasta.append(cm.encode_seq(seq))
    return fasta


def read_vista(file):
    seq = ""
    fasta = []
    names = []
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0:
                    seq = clean_seq(seq)
                    if len(seq) < seq_len:
                        seq = rand_seq(half_size) + seq + rand_seq(half_size)
                    fasta.append(seq)
                names.append(line.strip()[1:])
                seq = ""
            else:
                seq += line
        if len(seq) != 0:
            seq = clean_seq(seq)
            if len(seq) < seq_len:
                seq = rand_seq(half_size) + seq + rand_seq(half_size)
            fasta.append(seq)
    return fasta, names

def rand_seq(n):
    rseq = ""
    nuclist = ["A", "C", "G", "T"]
    for i in range(n):
        ind = random.randint(0,3)
        rseq = rseq + nuclist[ind]
    return rseq


def clean_seq(s):
    ns = s.upper()
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return ns


os.chdir(open("../data_dir").read().strip())
models_folder = "models/"
batch_size = 128
min_dist = 100
fasta, names = read_vista("data/vista.fa")

scan_step = 1
print("")
print("---------------------------------------------------------")
print("---------------------------------------------------------")
print("")
vista_scores = []
new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], models_folder + "model_predict")
    saver = tf.train.Saver()
    saver.restore(sess, models_folder + "model_predict/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr = tf.get_default_graph().get_tensor_by_name("kr:0")
    in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
    for i in range(len(fasta)):
        if "chr1:" not in names[i]:
            continue
        fa = fasta[i]
        j = half_size
        m = 1
        preds = []
        batch = []
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
        vista_scores.append(max(preds))

    negative_set = read_fasta("data/negatives.fa")[:len(vista_scores) * 100]
    negative_pred = cm.brun(sess, input_x, y, negative_set, kr, in_training_mode)
    negative_pred = [p[0] for p in negative_pred]

total_scores = vista_scores.copy()
total_scores.extend(negative_pred)

ground_truth = [1] * len(vista_scores)
ground_truth.extend([0] * len(negative_set))
np.savetxt("figures_data/ground_truth.csv", np.asarray(ground_truth), delimiter=",")
np.savetxt("figures_data/total_scores.csv", np.asarray(total_scores), delimiter=",")
# cm.draw_roc(ground_truth, total_scores, "roc_vista.png", "VISTA enhancers performance")