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

def encode(ns, strand):
    if(strand == "+"):
        rep = {"A": "1,0,0,0,", "T": "0,1,0,0,", "G": "0,0,1,0,", "C": "0,0,0,1,", "N": "0,0,0,0,"} 
    else:
        ns = ns[::-1]
        rep = {"A": "0,1,0,0,", "T": "1,0,0,0,", "G":"0,0,0,1," , "C": "0,0,1,0,", "N": "0,0,0,0,"} 
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    ns = pattern.sub(lambda m: rep[re.escape(m.group(0))], ns)
    return np.fromstring(ns[:-1], dtype=int, sep=",").reshape(-1, 4) 

half_size = 1000
out = []

fasta = pickle.load( open( "fasta.p", "rb" ) )

background = {}
background["RPLP0_CE_bg"] = ["chr12", "-", 120638861 - 1, 120639013 - 1]
background["ACTB_CE_bg"] = ["chr7",   "-", 5570183 - 1, 5570335 - 1]
background["C14orf166_CE_bg"] = ["chr14", "+", 52456090 - 1,  52456242 - 1] 

new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_pos")
    saver = tf.train.Saver()
    saver.restore(sess, "model_pos/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr = tf.get_default_graph().get_tensor_by_name("kr:0")
    in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
    with open('Supplemental_Table_S7.tsv') as file:
        next(file)
        for line in file:
            vals = line.split("\t")            
            fa = vals[25]
            if(vals[3] in background.keys()):
                bg = background[vals[3]]
                strand = bg[1]
                if(strand == "+"):
                    fa_bg = encode(fasta[bg[0]][bg[2] - (1000 - 114) : bg[2] - (1000 - 114) + 2001], strand)
                else:
                    fa_bg = encode(fasta[bg[0]][bg[2] - (1000 - 49) : bg[2] - (1000 - 49) + 2001], strand)

                fa_bg[1000 - 114 : 1000 + 50] = encode(fa, "+")
                predict = sess.run(y, feed_dict={input_x: [fa_bg], kr: 1.0, in_training_mode: False})
                score = predict[0][0]
                out.append(line.strip() + "\t" + str(score))

with open("designed_scores2.tsv", 'w+') as f:
    f.write('\n'.join(out))