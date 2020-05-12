#!/usr/bin/env python
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import argparse
import logging
import math
import pickle
import re
import time
import bisect
import numpy as np
import tensorflow as tf
from pathlib import Path


# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_options():
    parser = argparse.ArgumentParser(description='Version: 1.0')
    parser.add_argument('-I', metavar='input', required=True,
                        help='path to the input genome')
    parser.add_argument('-O', metavar='output', required=True,
                        help='path to the output file')
    parser.add_argument('-M', metavar='mode', default=1,
                        type=int, choices=range(0, 2),
                        help='')
    parser.add_argument('-T', metavar='threshold', default=0.5,
                        type=float,
                        help='decision threshold for the prediction model'
                             ', defaults to 0.5')
    parser.add_argument('-C', metavar='chromosomes', default="",
                        type=str, help='comma separated list of chromosomes to use for promoter prediction '
                                       ', defaults to all chromosomes')
    parser.add_argument('-CE', metavar='chromosomes_except', default="",
                        type=str, help='comma separated list of chromosomes excluded for promoter prediction ')

    args = parser.parse_args()

    return args


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


def close(s, a):
    fmd = float('inf')
    for v in a:
        if (abs(s - v) < fmd):
            fmd = abs(s - v)
    return fmd


def clean_seq(s):
    ns = s.upper()
    pattern = re.compile(r'\s+')
    ns = re.sub(pattern, '', ns)
    ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return ns


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def pick(scores, dt, minDist):
    scores.sort(key=lambda x: x[1], reverse=True)
    all_chosen = []
    rows = []
    for s in range(len(scores)):
        position = scores[s][0]
        score = scores[s][1]
        strand = scores[s][2]
        cl = scores[s][3]
        scaling = 1.0
        if len(all_chosen) > 0:
            fmd = abs(position - find_nearest(all_chosen, position))
            if fmd < minDist:
                scaling = fmd / minDist
        if score * scaling >= dt:
            bisect.insort(all_chosen, position)
            rows.append([position, score, strand, cl])
    return rows


def main():
    args = get_options()

    if None in [args.I, args.O]:
        logging.error('Usage: deepag [-h] [-I [input]] [-O [output]] -D distance -T threshold '
                      ' -C chromosomes')
        exit()

    print("DeepRAG v1.02")
    sLen = 1001
    half_size = 500
    batch_size = 128
    prediction_names = ["promoter", "promoter", "enhancer", "negative"]

    dt1 = 0.9
    dt2 = args.T
    minDist = 500

    test_mode = args.M == 1
    if test_mode:
        print("Testing all locations")
    else:
        print("Predicting for new negatives")

    ga = pickle.load(open("ga.p", "rb"))
    print("Scan threshold: " + str(dt1))
    print("Prediction threshold: " + str(dt2))
    print("Parsing fasta at " + str(args.I))
    fasta = {}
    seq = ""

    if Path("fasta.p").is_file():
        fasta = pickle.load(open("fasta.p", "rb"))
    else:
        with open(args.I) as f:
            for line in f:
                if line.startswith(">"):
                    if len(seq) != 0:
                        seq = clean_seq(seq)
                        fasta[chrn] = seq
                        print(chrn + " - " + str(len(seq)))
                    chrn = line.strip()[1:]
                    try:
                        if args.I.endswith('.fna'):
                            chrn = re.search('chromosome (.*), GRCh38', chrn)
                            chrn = chrn.group(1).strip()
                        else:
                            chrn = line.strip()[1:]
                    except Exception as e:
                        pass
                    seq = ""
                    continue
                else:
                    seq += line
            if len(seq) != 0:
                seq = clean_seq(seq)
                fasta[chrn] = seq
                print(chrn + " - " + str(len(seq)))

    good_chr = fasta.keys()
    if args.C != "":
        good_chr = args.C.split(",")
    elif args.CE != "":
        exclude = args.CE.split(",")
        good_chr = [x for x in good_chr if x not in exclude]

    for key in list(fasta.keys()):
        if key not in good_chr:
            del fasta[key]
    putative = {}
    scan_step = 100
    print("")
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print("")
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "models/model_scan")
        saver = tf.train.Saver()
        saver.restore(sess, "models/model_scan/variables/variables")
        input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
        y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
        kr = tf.get_default_graph().get_tensor_by_name("kr:0")
        in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
        for key in fasta.keys():
            print("Scanning " + key)
            putative[key] = []
            j = half_size
            m = 1
            batch = []
            inds = []
            while j < len(fasta[key]) - half_size - 1:
                if test_mode or sum(ga[key][j - half_size: j + half_size + 1]) == 0:
                    fa = encode(key, j, fasta, sLen)
                    if len(fa) == sLen:
                        batch.append(fa)
                        inds.append(j)
                    if len(batch) >= batch_size or j + scan_step >= len(fasta[key]) - half_size - 1:
                        predict = sess.run(y, feed_dict={input_x: batch, kr: 1.0, in_training_mode: False})
                        chosen = [inds[i] for i in range(len(batch)) if max(predict[i][:-1]) > dt1]
                        putative[key].extend(chosen)
                        batch = []
                        inds = []
                j = j + scan_step
                if j > m * 10000000:
                    print(str(j))
                    m = m + 1

            print("Scanned chromosome " + key + ". Found " + str(
                len(putative[key])) + " putative regulatory regions." + " [" + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                             time.gmtime()) + "] ")

    out = []
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "models/model_predict")
        saver = tf.train.Saver()
        saver.restore(sess, "models/model_predict/variables/variables")
        input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
        y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
        kr = tf.get_default_graph().get_tensor_by_name("kr:0")
        in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
        for key in fasta.keys():
            print("Predicting " + key)
            rows = []
            scores = []
            reset = 0
            m = 1
            prev_pred = -1
            for p in putative[key]:
                batch = []
                inds = []
                for j in range(p - int(scan_step / 2), p + int(scan_step / 2) + 1):
                    fa = encode(key, j, fasta, sLen)
                    if len(fa) == sLen:
                        batch.append(fa)
                        inds.append(j)
                predict = sess.run(y, feed_dict={input_x: batch, kr: 1.0, in_training_mode: False})
                predict = np.delete(predict, -1, 1)
                mr = np.argmax(np.max(predict, axis=1))
                mc = np.argmax(np.max(predict, axis=0))
                if predict[mr][mc] > dt2:
                    strand = "."
                    if mc == 0:
                        strand = '+'
                    elif mc == 1:
                        strand = '-'
                    if prev_pred != -1 and abs(inds[mr] - prev_pred) >= minDist:
                        new_scores = pick(scores, dt2, minDist)
                        rows.extend(new_scores)
                        scores = []
                        reset = reset + 1
                        # print("Predicted " + str(len(new_scores)) + " regions")
                    scores.append([inds[mr], predict[mr][mc], strand, mc])
                    # rows.append([inds[mr], predict[mr][mc], strand, mc])
                    prev_pred = inds[mr]
                if p > m * 10000000:
                    print(str(p))
                    m = m + 1

            if len(scores) > 0:
                scores.sort(key=lambda x: x[1], reverse=True)
                new_scores = pick(scores, dt2, minDist)
                rows.extend(new_scores)
            print("Reset: " + str(reset))
            print("Prediction complete for " + key + " chromosome. Predicted " + str(
                len(rows)) + " regulatory regions." + " [" + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                           time.gmtime()) + "] ")

            for row in rows:
                out.append(key + "\t" + "DeepRAG" + "\t" + prediction_names[row[3]] + "\t" + str(
                    row[0] - 100 + 1) + "\t" + str(
                    row[0] + 100 + 2) + "\t" +
                           str(row[1]) + "\t" + row[2] + "\t" + "." + "\t" +
                           key + ":" + str(row[0] - half_size + 1) + ":" + str(row[0] + half_size + 2) + ":" + row[2])

    with open(args.O, 'w+') as f:
        f.write('\n'.join(out))


if __name__ == '__main__':
    main()
