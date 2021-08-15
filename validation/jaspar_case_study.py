import random
import numpy as np
from scipy.stats import ttest_ind
import os
import tensorflow as tf
import re
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

os.chdir("/media/user/30D4BACAD4BA9218/data_ubuntu/DeepRAG/")
models_folder = "models/"

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


def read_fasta(file):
    seq = ""
    fasta = {}
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                if len(seq) != 0:
                    seq = clean_seq(seq)
                    fasta[pname] = encode(seq)
                seq = ""
                pname = line[1:].strip()
            else:
                seq += line
        if len(seq) != 0:
            seq = clean_seq(seq)
            fasta[pname] = encode(seq)
    return fasta


def brun(sess, x, y, a, keep_prob, in_training_mode):
    preds = []
    batch_size = 256
    number_of_full_batch = int(math.ceil(float(len(a)) / batch_size))
    for i in range(number_of_full_batch):
        preds += list(sess.run(y, feed_dict={x: np.asarray(a[i * batch_size:(i + 1) * batch_size]),
                                             keep_prob: 1.0, in_training_mode: False}))
    return preds


def calculate_score(orig_score, sequence, regions):
    total_score = 0
    for region in regions:
        region1 = region.split(":")
        region1 = slice(int(region1[0]), int(region1[1]))      
        seq1 = sequence.copy()
        seq1[region1] = [[0,0,0,0] for _ in range(len(seq1[region1]))]
        rm1 = sess.run(y, feed_dict={input_x: np.asarray([seq1]), kr: 1.0, in_training_mode: False})[0][0]
        score = abs(orig_score - rm1)
        total_score += score
    return total_score / len(regions)

prom_seqs = read_fasta("data/promoters.fa")
enh_seqs = read_fasta("data/enhancers.fa")

results_final = ["TF,TF ID,Promoters Score,Enhancers score"]
new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], models_folder + "model_predict")
    saver = tf.train.Saver()
    saver.restore(sess, models_folder + "model_predict/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr = tf.get_default_graph().get_tensor_by_name("kr:0")
    in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")

    original_scores = {}
    prom_sum = 0
    for k in prom_seqs.keys():
        orig_score = sess.run(y, feed_dict={input_x: np.asarray([prom_seqs[k]]), kr: 1.0, in_training_mode: False})[0][0]
        prom_sum += orig_score
        original_scores[k] = orig_score
    print("Prom avg score " + str(prom_sum / len(prom_seqs.keys())))

    enh_sum = 0
    for k in enh_seqs.keys():
        orig_score = sess.run(y, feed_dict={input_x: np.asarray([enh_seqs[k]]), kr: 1.0, in_training_mode: False})[0][0]
        enh_sum += orig_score
        original_scores[k] = orig_score
    print("Enh avg score " + str(enh_sum / len(enh_seqs.keys())))

    fimo_dir = "/media/user/30D4BACAD4BA9218/data_ubuntu/DeepRAG/data/fimo_jaspar/"
    for filename in os.listdir(fimo_dir):
        # print(filename)
        tf_prom = {}
        tf_enh = {}
        with open(fimo_dir + filename + "/prom/fimo.gff") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                vals = line.split("\t")
                name = vals[0]
                if 460 < int(vals[3]) < 541:
                    continue
                region = vals[3] + ":" + vals[4]
                tf_prom.setdefault(name, []).append(region)

        with open(fimo_dir + filename + "/enh/fimo.gff") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                vals = line.split("\t")
                name = vals[0]
                if 460 < int(vals[3]) < 541:
                    continue
                region = vals[3] + ":" + vals[4]
                tf_enh.setdefault(name, []).append(region)

        final_score_prom = 0
        final_score_enh = 0
        if len(tf_prom.keys()) > 0:
            for key in tf_prom.keys():
                score = calculate_score(original_scores[key], prom_seqs[key], tf_prom[key])
                final_score_prom += score
            final_score_prom /= len(tf_prom.keys())

        if len(tf_enh.keys()) > 0:
            for key in tf_enh.keys():
                score = calculate_score(original_scores[key], enh_seqs[key], tf_enh[key])
                final_score_enh += score
            final_score_enh /= len(tf_enh.keys())

        # print(str(final_score_prom) + " " + str(final_score_enh))
        name = open(fimo_dir + filename + "/info.txt").read().strip()
        row = filename + "," + name + "," + str(final_score_prom) + "," + str(final_score_enh)
        print(row)
        results_final.append(row)

with open("jaspar_sub_results.csv", 'w+') as f:
    f.write('\n'.join(results_final))
