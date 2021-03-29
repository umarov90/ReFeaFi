import dependency_score
from random import randint
import numpy as np
from scipy.stats import ttest_ind
import os
import tensorflow as tf
import re
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
models_folder = "/home/user/data/DeepRAG/models/"
fimo_path = "/home/user/data/DeepRAG/data/fimo_test/"

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


def read_fasta(file, nn=0, sname=None):
    seq = ""
    fasta = []
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0 and (sname is None or sname == pname):
                    seq = clean_seq(seq)
                    seq = "N" * nn + seq + "N" * nn
                    fasta.append(encode(seq))
                    if sname is not None and sname == pname:
                        return fasta
                seq = ""
                pname = line[1:].strip()
            else:
                seq += line
        if len(seq) != 0 and (sname is None or sname == pname):
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


def calculate_score(sequences, region1, region2, sname=None):
    sequences = read_fasta(sequences, sname=sname)
    region1 = region1.split(":")
    region1 = slice(int(region1[0]), int(region1[1]))
    region2 = region2.split(":")
    region2 = slice(int(region2[0]), int(region2[1]))
    total_score = 0
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
    return total_score / len(sequences)


new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], models_folder + "model_predict")
    saver = tf.train.Saver()
    saver.restore(sess, models_folder + "model_predict/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr = tf.get_default_graph().get_tensor_by_name("kr:0")
    in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
    for filename in os.listdir(fimo_path):
        sub = os.path.join(fimo_path, filename) + "/"
        if not os.path.isdir(sub):
            continue
        try:
            tfs = [d for d in os.listdir(sub) if os.path.isdir(sub + d)]
            print("-"*100)
            print(tfs[0] + " " + tfs[1])
            with open(sub + "info.txt", 'r') as file:
                data = file.read().strip()
            print(data)
            print("-" * 10)
            tf1 = {}
            tf2 = {}
            with open(sub + tfs[0] + "/fimo.gff") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    vals = line.split("\t")
                    name = vals[0]
                    if name in tf1.keys():
                        continue
                    region = vals[3] + ":" + vals[4]
                    tf1[name] = region

            with open(sub + tfs[1] + "/fimo.gff") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    vals = line.split("\t")
                    name = vals[0]
                    if name in tf2.keys():
                        continue
                    region = vals[3] + ":" + vals[4]
                    tf2[name] = region

            scores1 = []
            scores2 = []
            common_keys = tf1.keys() & tf2.keys()
            for key in common_keys:
                score = calculate_score(fimo_path + "promoters.fa", tf1[key], tf2[key], key)
                scores1.append(score)
                size = int(tf2[key].split(":")[1]) - int(tf2[key].split(":")[0])
                start = randint(0, 1001-size)
                end = start + size
                score = calculate_score(fimo_path + "promoters.fa", tf1[key], str(start)+":"+str(end), key)
                scores2.append(score)

            scores1 = np.asarray(scores1)
            scores2 = np.asarray(scores2)
            print("scores 1 avg: " + str(np.mean(scores1)))
            print("scores 2 avg: " + str(np.mean(scores2)))
            t, p = ttest_ind(scores1, scores2)
            print("p value: " + str(p))
            with open(fimo_path + "pval.csv", "a+") as file:
                file.write("\n" + str(p) + "," + data + "\n")
            print("-" * 100)
        except:
            pass