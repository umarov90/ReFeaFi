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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
sequences = read_fasta(sys.argv[1])
# sequences = sequences[:100]
region1 = sys.argv[2].split(":")
region1 = slice(int(region1[0]), int(region1[1]))
region2 = sys.argv[3].split(":")
region2 = slice(int(region2[0]), int(region2[1]))
# region1 = slice(495, 505)
# region2 = slice(160, 175)
total_score = 0
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
    for seq in sequences:
        orig_score = sess.run(y, feed_dict={input_x: np.asarray([seq]), kr: 1.0, in_training_mode: False})[0][0]
        seq1 = seq.copy()
        seq1[region1] = [[0,0,0,0] for _ in range(len(seq1[region1]))]
        rm1 = sess.run(y, feed_dict={input_x: np.asarray([seq1]), kr: 1.0, in_training_mode: False})[0][0]
        seq2 = seq.copy()
        seq2[region2] = [[0,0,0,0] for _ in range(len(seq2[region2]))]
        rm2 = sess.run(y, feed_dict={input_x: np.asarray([seq2]), kr: 1.0, in_training_mode: False})[0][0]
        seq12 = seq.copy()
        seq12[region1] = [[0,0,0,0] for _ in range(len(seq12[region1]))]
        seq12[region2] = [[0,0,0,0] for _ in range(len(seq12[region2]))]
        rm12 = sess.run(y, feed_dict={input_x: np.asarray([seq]), kr: 1.0, in_training_mode: False})[0][0]
        score = abs(orig_score - rm12) - abs(orig_score - rm1 + orig_score - rm2)
        total_score += score

print("Dependency score: " + str(total_score / len(sequences)))
