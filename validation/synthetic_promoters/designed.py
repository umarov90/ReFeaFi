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

fasta = pickle.load( open( "fasta.p", "rb" ) )
half_size = 1000
out = []
out2 = []

cage = {}
with open('hg19.cage_peak_phase1and2combined_coord.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]
        strand = vals[5]
        val = 1
        if(strand == "+"):
            chrp = int(vals[7]) - 1
        elif(strand == "-"):
            chrp = int(vals[7]) - 1
        ck = chrn + strand
        if(ck in cage.keys()):
            cage[ck].append(chrp)
        else:
            cage[ck] = [chrp] 

new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_pos")
    saver = tf.train.Saver()
    saver.restore(sess, "model_pos/variables/variables")
    input_x = tf.get_default_graph().get_tensor_by_name("input_prom:0")
    y = tf.get_default_graph().get_tensor_by_name("output_prom:0")
    kr = tf.get_default_graph().get_tensor_by_name("kr:0")
    in_training_mode = tf.get_default_graph().get_tensor_by_name("in_training_mode:0")
    with open('des.tsv') as file:
        next(file)
        for line in file:
            vals = line.split("\t")
            chrn = vals[2]
            start = int(vals[3])
            end = int(vals[4])
            if(start < end):
                lside = 103 + 11
                strand = "+"
                j = start - (half_size - lside)
                #gt = find_nearest(cage["chr" + chrn + "+"], j)
                #print(str(j) + " - " + str(gt), flush=True)
                fa = fasta["chr" + chrn][j - half_size: j + half_size + 1]
                fad = fa[0:1001-lside] + vals[8].strip() + fa[1000 - lside + len(vals[8]):2001]
                out2.append(">1")
                out2.append("\n")
                out2.append(fad)
                out2.append("\n")
                predict = sess.run(y, feed_dict={input_x: [encode(fad, strand)], kr: 1.0, in_training_mode: False})
                score = predict[0][0]
                out.append(line.strip() + "\t" + str(score))

with open("designed_scores.tsv", 'w+') as f:
    f.write('\n'.join(out))

with open("designed.fa", 'w+') as f:
    f.write(''.join(out2))