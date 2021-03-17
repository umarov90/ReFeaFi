import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
from math import sqrt
from numpy import zeros
import sys
import re
import math

import pickle
from pathlib import Path
import time
from random import randint
from random import uniform
from random import shuffle
from tensorflow.contrib import rnn
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
import gc

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 2e-5


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def rand_seq(rshift):
    z = zeros((rshift, 4), dtype=np.bool)
    for i in range(rshift):
        r = np.random.randint(0, 4)
        z[i][r] = True
    return z

def test_loss(predict, y_test):
    return np.square(np.subtract(np.asarray(predict), np.asarray(y_test))).mean()


def fully_connected(input, name, output_size):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_' + name,
                            shape=[input_size, output_size])
        b = tf.get_variable(name='bias_' + name,
                            shape=[output_size])
        input = tf.reshape(input, [-1, input_size])
        out = tf.add(tf.matmul(input, W), b)
        return out, W


def identity_building_block(input_tensor,
                            kernel_size,
                            filters,
                            training=None):
    x = layers.Conv1D(filters, kernel_size,
                      padding='same', use_bias=True)(input_tensor)
    x = layers.BatchNormalization()(x, training=training)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(filters, kernel_size,
                      padding='same', use_bias=True)(x)
    x = layers.BatchNormalization()(x, training=training)

    x = layers.add([x, input_tensor])
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def conv_building_block(input_tensor,
                        kernel_size,
                        filters,
                        strides=1,
                        training=None):
    x = layers.Conv1D(filters, kernel_size, strides=strides,
                      padding='same', use_bias=True)(input_tensor)
    x = layers.BatchNormalization()(x, training=training)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=True)(x)
    x = layers.BatchNormalization()(x, training=training)

    shortcut = layers.Conv1D(filters, 1, strides=strides, use_bias=True)(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut, training=training)

    x = layers.add([x, shortcut])
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def resnet_block(input_tensor,
                 size,
                 kernel_size,
                 filters,
                 conv_strides,
                 training):
    x = conv_building_block(input_tensor, kernel_size, filters,
                            strides=conv_strides,
                            training=training)
    for i in range(size - 1):
        x = identity_building_block(x, kernel_size, filters, training=training)
    return x


def model(x, y, keep_ratio, in_training_mode):
    num_blocks = 3
    filter_num = 128
    lc1 = layers.Conv1D(filter_num, 3, padding='same', use_bias=True)(x)
    lc1 = layers.LeakyReLU(alpha=0.2)(lc1)
    resnet = resnet_block(lc1, size=num_blocks, kernel_size=3, filters=filter_num,
                          conv_strides=1, training=in_training_mode)

    resnet = resnet_block(resnet, size=num_blocks, kernel_size=3, filters=filter_num,
                          conv_strides=2, training=in_training_mode)

    resnet = resnet_block(resnet, size=num_blocks, kernel_size=3, filters=filter_num,
                          conv_strides=2, training=in_training_mode)

    print(num_blocks)
    print(resnet)
    dp = tf.nn.dropout(resnet, keep_prob=keep_ratio)
    out, W = fully_connected(dp, "output_p", num_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)
    out = tf.nn.softmax(out)
    # loss = tf.add(loss, 0.00001 * tf.nn.l2_loss(W))
    return out, loss


def clean_seq(s):
    s = s.upper()
    pattern = re.compile(r'\s+')
    s = re.sub(pattern, '', s)
    # ns = re.sub(r'[^a-zA-Z]{1}', 'N', ns)
    return s


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


def brun(sess, x, y, a, keep_prob, in_training_mode):
    preds = []
    batch_size = 256
    number_of_full_batch = int(math.ceil(float(len(a)) / batch_size))
    for i in range(number_of_full_batch):
        preds += list(sess.run(y, feed_dict={x: np.asarray(a[i * batch_size:(i + 1) * batch_size]),
                                             keep_prob: 1.0, in_training_mode: False}))
    return preds


def un(a, m, h, seq):
    rv = -1
    while (True):
        rv = randint(h, m - h)
        bad = False
        for v in a:
            if (abs(rv - v[0]) < 800):
                bad = True
                break
        nseq = seq[rv - h: rv + h + 1]
        if (not bad and nseq.count("N") < 400):
            break
    return rv


def rand_unmapped(chrn, ga, fasta, seq_len):
    half_size = int((seq_len - 1) / 2)
    while (True):
        try:
            rp = randint(0, len(ga))
            if (sum(ga[rp - 500: rp + 500 + 1]) == 0):
                neg_mat = encode(chrn, rp, fasta, seq_len)
                if (neg_mat is None):
                    pass
                else:
                    return rp, neg_mat
        except:
            pass


def prep(ar):
    flat = np.asarray(ar)
    if len(flat) == 100:
        flat = np.append(flat, [[0, 0]], axis=0)
    return flat


data_folder = "/home/user/data/DeepRAG/"
os.chdir(data_folder)

good_chr = ["chrX", "chrY"]
for i in range(1, 23):
    good_chr.append("chr" + str(i))

max_features = 4
seq_len = 1001
out_dir = sys.argv[1]
data = []
elens = []
overlap = 0
enhancer_count = 0
promoter_count = 0
fasta = pickle.load(open("fasta.p", "rb"))

print("Parsing CAGE bed")
with open('data/hg19.cage_peak_phase1and2combined_coord.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]  # [3:len(vals[0])]
        strand = vals[5]
        if (chrn == "chr1"):
            continue
        if (chrn not in good_chr):
            continue
        chrp = int(vals[7]) - 1
        label = [False, False, True]
        if strand == "+":
            label = [True, False, False]
        elif strand == "-":
            label = [False, True, False]
        else:
            continue

        promoter_count = promoter_count + 1
        seq_mat = encode(chrn, chrp, fasta, seq_len)
        if (seq_mat is None):
            pass
        else:
            data.append([seq_mat, label])
with open('data/human_permissive_enhancers_phase_1_and_2.bed') as file:
    for line in file:
        vals = line.split("\t")
        chrn = vals[0]  # [3:len(vals[0])]
        strand = vals[5]
        if (chrn == "chr1"):
            continue
        if (chrn not in good_chr):
            continue
        chrp = int(vals[7]) - 1
        if strand == ".":
            seq_mat = encode(chrn, chrp, fasta, seq_len)
            data.append([seq_mat, [False, False, True]])
print("done")
print("Parsed data " + str(len(data)))

shuffle(data)
training_size = 0.9
tr_data = data[0:int(training_size * len(data))]
ts_data = data[int(training_size * len(data)): len(data)]

x_train = []
y_train = []
x_test = []
y_test = []

half_size = 500

for d in tr_data:
    x_train.append(d[0])
    y_train.append(d[1])
    cl = np.where(d[1])[0][0]
for d in ts_data:
    x_test.append(d[0])
    y_test.append(d[1])
    cl = np.where(d[1])[0][0]

print("Generated training set")

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

num_classes = 3
best_error = 10000000
patience = 30
wait = 0
batch_size = 128
nb_epoch = 300
# initialize inputs
x = tf.placeholder(tf.float32, shape=[None, seq_len, max_features], name="input_prom")
y = tf.placeholder(tf.float32, shape=[None, num_classes])
in_training_mode = tf.placeholder(tf.bool, name="in_training_mode")
keep_prob = tf.placeholder(tf.float32, name="kr")
# build the model
out, loss = model(x, y, keep_prob, in_training_mode)
out = tf.identity(out, name="output_prom")
# initialize optimizer
extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
# run the training loop
kp = 0.5
prev_to_delete = []
print(out_dir)
if not os.path.exists("./store/" + out_dir):
    os.makedirs("./store/" + out_dir, exist_ok=True)
open("./store/" + out_dir + "/check", 'a').close()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    for epoch in range(nb_epoch):
        gc.collect()
        my_file = Path("./store/" + out_dir + "/check")
        if not my_file.is_file():
            print("train file not found")
            break
        rng_state = np.random.get_state()
        np.random.shuffle(x_train)
        np.random.set_state(rng_state)
        np.random.shuffle(y_train)
        np.random.set_state(rng_state)
        total = int(math.ceil(float(len(x_train)) / batch_size))
        sp = 0
        for i in range(total):
            feed = {x: x_train[i * batch_size:(i + 1) * batch_size],
                    y: y_train[i * batch_size:(i + 1) * batch_size], keep_prob: kp, in_training_mode: True}
            train_step.run(feed_dict=feed)
        pred_tr = []
        if epoch % 1 == 0:
            pred_tr = brun(sess, x, out, x_train, keep_prob, in_training_mode)
            trloss = test_loss(pred_tr, y_train)
            pred = brun(sess, x, out, x_test, keep_prob, in_training_mode)
            tsloss = test_loss(pred, y_test)
            prom_correct = 0
            prom_error = 0
            enhancer_correct = 0
            enhancer_error = 0
            for p in range(len(pred)):
                if y_test[p][2] == 0:
                    if max(pred[p][0], pred[p][1]) > pred[p][2]:
                        prom_correct = prom_correct + 1
                    else:
                        prom_error = prom_error + 1
                else:
                    if max(pred[p][0], pred[p][1]) < pred[p][2]:
                        enhancer_correct = enhancer_correct + 1
                    else:
                        enhancer_error = enhancer_error + 1

            print("Training set loss: " + str(trloss))
            print("Test set loss: " + str(tsloss))
            print(len(pred))
            print(prom_correct)
            print(prom_error)
            print(enhancer_correct)
            print(enhancer_error)
            errors = enhancer_error + prom_error
            saver.save(sess, "./store/" + out_dir + "/" + str(epoch) + ".ckpt")
            if errors < best_error:
                best = epoch
                print("------------------------------------best so far")
                best_error = errors

        gc.collect()

    # exit()
    saver.restore(sess, "./store/" + out_dir + "/" + str(best) + ".ckpt")
    if (os.path.exists(out_dir)):
        out_dir = out_dir + str(time.time())
    builder = tf.saved_model.builder.SavedModelBuilder(out_dir)
    predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(x)
    predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(y)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(inputs={"input_prom": predict_tensor_inputs_info},
                                                               outputs={"output_prom": predict_tensor_scores_info},
                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={"model": prediction_signature})
    builder.save(True)
    print("model saved: " + out_dir)
