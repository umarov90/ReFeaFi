import sys
import re
import tensorflow as tf
import os
import math
import numpy as np
import common as cm
from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def clean_seq(s):
    ns = s.upper()
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return ns


enc_mat = np.append(np.eye(4),
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
                     [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                     [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], axis=0)
enc_mat = enc_mat.astype(np.bool)
mapping_pos = dict(zip("ACGTRYSWKMBDHVN", range(15)))


def encode(enc_seq):
    try:
        seq2 = [mapping_pos[i] for i in enc_seq]
        return enc_mat[seq2]
    except:
        print(enc_seq)
        return None


def read_fasta(file, nn=0):
    seq = ""
    fasta = []
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0:
                    seq = clean_seq(seq)
                    seq = "N" * nn + seq + "N" * nn
                    fasta.append(encode(seq))
                seq = ""
            else:
                seq += line
        if len(seq) != 0:
            seq = clean_seq(seq)
            seq = "N" * nn + seq + "N" * nn
            fasta.append(encode(seq))
    return fasta


def brun(sess, x, y, a, keep_prob, in_training_mode):
    preds = []
    batch_size = 256
    number_of_full_batch = int(math.ceil(float(len(a)) / batch_size))
    for i in range(number_of_full_batch):
        preds += list(sess.run(y, feed_dict={x: np.asarray(a[i * batch_size:(i + 1) * batch_size]),
                                             keep_prob: 1.0, in_training_mode: False}))
    return preds


os.chdir("/home/user/data/DeepRAG")
positive_set = read_fasta(sys.argv[1])
negative_set = read_fasta(sys.argv[2])

models_folder = "models/"
new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], models_folder + "model_predict")
    saver = tf.train.Saver()
    saver.restore(sess, models_folder + "model_predict/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr = tf.get_default_graph().get_tensor_by_name("kr:0")
    in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
    positive_pred = brun(sess, input_x, y, positive_set, kr, in_training_mode)
    negative_pred = brun(sess, input_x, y, negative_set, kr, in_training_mode)

total_scores = positive_pred.copy()
total_scores.extend(negative_pred)
total_scores = [p[0] for p in total_scores]

ground_truth = [1] * len(positive_set)
ground_truth.extend([0] * len(negative_set))

cm.draw_roc(ground_truth, total_scores, "roc.png")