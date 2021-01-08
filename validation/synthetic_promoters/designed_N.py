#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from math import sqrt
import numpy as np
from numpy import zeros
import sys
import re
import math
import os
from random import randint
from tensorflow.python.saved_model import builder as saved_model_builder
import pickle
import time

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

enc_mat = np.append(np.eye(4), [[0,0,0,0]], axis=0)
enc_mat = enc_mat.astype(np.bool)
mapping_pos = dict(zip("ACGTN", range(5)))
mapping_neg = dict(zip("TGCAN", range(5)))
def encode(seq, strand):
    if(strand == "+"):
        seq2 = [mapping_pos[i] for i in seq]
    else:
        seq = seq[::-1]
        seq2 = [mapping_neg[i] for i in seq]
    return enc_mat[seq2]

half_size = 500
out = []
out2 = []

background = ["RPLP0_CE_bg", "ACTB_CE_bg", "C14orf166_CE_bg"] 

new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_pos_no_bn_improved_small")
    saver = tf.train.Saver()
    saver.restore(sess, "model_pos_no_bn_improved_small/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr = tf.get_default_graph().get_tensor_by_name("kr:0")
    in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
    with open('Supplemental_Table_S7.tsv') as file:
        next(file)
        for line in file:
            vals = line.split("\t")    
            if(vals[3] in background):        
                fa = vals[25].strip()
                fa = "N"*(half_size - 103 - 11) + fa + "N"*( half_size + 1 - 50)
                predict = sess.run(y, feed_dict={input_x: [encode(fa, "+")], kr: 1.0, in_training_mode: False})
                score = predict[0][0]
                out.append(line.strip() + "\t" + str(score))
                fad = fa[half_size-5:half_size+6]
                out2.append(">1")
                out2.append("\n")
                out2.append(fad)
                out2.append("\n")

with open("designed_scores2.tsv", 'w+') as f:
    f.write('\n'.join(out))

with open("designed.fa", 'w+') as f:
    f.write(''.join(out2))