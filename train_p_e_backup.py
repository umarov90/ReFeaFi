import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
L2_WEIGHT_DECAY = 1e-5


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


def test_mcc(predict, y_test):
    cl = 0
    a = zeros(len(y_test))
    for i in range(len(predict)):
        if predict[i][cl] > 0.5:
            a[i] = 1
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0

    for i in range(len(y_test)):
        if (y_test[i][cl] == 1):
            if (a[i] == 1):
                tp += 1
            else:
                fn += 1
        if (y_test[i][cl] == 0):
            if (a[i] == 1):
                fp += 1
            else:
                tn += 1
    sn = 0.0
    sp = 0.0
    mcc = 0.0
    try:
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except Exception:
        pass
    return tp, tn, fp, fn, sn, sp, mcc


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
    bn_axis = 1
    x = layers.Conv1D(filters, kernel_size,
                      padding='same', use_bias=True)(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x, training=training)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size,
                      padding='same', use_bias=True)(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x, training=training)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_building_block(input_tensor,
                        kernel_size,
                        filters,
                        strides=1,
                        training=None):
    bn_axis = 1
    x = layers.Conv1D(filters, kernel_size, strides=strides,
                      padding='same', use_bias=True)(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x, training=training)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=True)(x)
    x = layers.BatchNormalization(
        axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(x, training=training)

    shortcut = layers.Conv1D(filters, 1, strides=strides, use_bias=True)(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(shortcut, training=training)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
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
    filter_num = 256
    lc1 = layers.Conv1D(filter_num, num_classes, padding='same', use_bias=True)(x)
    lc1 = layers.Activation('relu')(lc1)
    resnet = resnet_block(lc1, size=num_blocks, kernel_size=3, filters=filter_num,
                          conv_strides=1, training=in_training_mode)

    resnet = resnet_block(resnet, size=num_blocks, kernel_size=3, filters=filter_num,
                          conv_strides=2, training=in_training_mode)

    resnet = resnet_block(resnet, size=num_blocks, kernel_size=3, filters=filter_num,
                          conv_strides=2, training=in_training_mode)

    resnet = resnet_block(resnet, size=num_blocks, kernel_size=3, filters=filter_num,
                          conv_strides=2, training=in_training_mode)

    resnet = resnet_block(resnet, size=num_blocks, kernel_size=3, filters=filter_num,
                          conv_strides=2, training=in_training_mode)

    print(num_blocks)
    print(resnet)
    dp = tf.nn.dropout(resnet, keep_ratio)
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


def rand_unmapped(chrn, ga, fasta, seq_len):
    half_size = int((seq_len - 1) / 2)
    while True:
        try:
            rp = randint(0, len(ga))
            if sum(ga[rp - half_size: rp + half_size + 1]) == 0:
                neg_mat = encode(chrn, rp, fasta, seq_len)
                if neg_mat is None:
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
seq_len = 1201
out_dir = sys.argv[1]
scan_model = sys.argv[2] == '1'
if scan_model:
    print("Training scan model")
else:
    print("Training prediction model")

x_train = []
y_train = []
x_test = []
y_test = []

half_size = 500
tr_stat = [0, 0]
ts_stat = [0, 0]
my_file = Path("x_train.p")
if my_file.is_file():
    x_train = pickle.load(open("x_train.p", "rb"))
    x_test = pickle.load(open("x_test.p", "rb"))
    y_train = pickle.load(open("y_train.p", "rb"))
    y_test = pickle.load(open("y_test.p", "rb"))
    print("====================================================================")
    print("Loaded existing training set")
    print("Training set size: " + str(len(y_train)))
    print("====================================================================")
else:
    tr_dir = "fasta"
    fasta = {}
    data = []
    elens = []
    ga = {}
    overlap = 0
    enhancer_count = 0
    promoter_count = 0
    print("Parsing fasta")
    seq = ""
    with open("hg19.fa") as f:  # .masked
        for line in f:
            if (line.startswith(">")):
                if (len(seq) != 0):
                    if (chrn in good_chr):
                        ga[chrn] = zeros((len(seq)), dtype=np.uint8)
                        seq = clean_seq(seq)
                        fasta[chrn] = seq
                        print(chrn + " - " + str(len(seq)))
                chrn = line.strip()[1:]
                seq = ""
                continue
            else:
                seq += line
        if (len(seq) != 0):
            if (chrn in good_chr):
                ga[chrn] = zeros((len(seq)), dtype=np.uint8)
                seq = clean_seq(seq)
                fasta[chrn] = seq
                print(chrn + " - " + str(len(seq)))
    print("Done")
    print("Parsing CAGE bed")
    with open('cage_permissive.bed') as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]  # [3:len(vals[0])]
            strand = vals[5]
            if (chrn == "chr1"):
                continue
            if (chrn not in good_chr):
                continue
            chrp = int(vals[7]) - 1
            ga[chrn][chrp] = 4
    with open('cage.bed') as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]  # [3:len(vals[0])]
            strand = vals[5]
            if (chrn == "chr1"):
                continue
            if (chrn not in good_chr):
                continue
            chrp = int(vals[7]) - 1
            if strand == "+":
                ga[chrn][chrp] = 1
            elif strand == "-":
                ga[chrn][chrp] = 2
            else:
                continue
            promoter_count = promoter_count + 1
            seq_mat = encode(chrn, chrp, fasta, seq_len)
            if (seq_mat is None):
                pass
            else:
                # Positive
                data.append([seq_mat, [True, False]])
                # Negative
                rp, neg_mat = rand_unmapped(chrn, ga[chrn], fasta, seq_len)
                ga[chrn][rp] = 100
                data.append([neg_mat, [False, True]])
    with open('enhancers.bed') as file:
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
                data.append([seq_mat, [True, False]])
                ga[chrn][chrp] = 3
                enhancer_count = enhancer_count + 1
                # Negative
                rp, neg_mat = rand_unmapped(chrn, ga[chrn], fasta, seq_len)
                ga[chrn][rp] = 100
                data.append([neg_mat, [False, True]])

    print("done")
    print("Promoters: " + str(promoter_count))
    print("Enhancers: " + str(enhancer_count))
    print("Parsed data " + str(len(data)))

    shuffle(data)
    training_size = 0.9
    tr_data = data[0:int(training_size * len(data))]
    ts_data = data[int(training_size * len(data)): len(data)]
    for d in tr_data:
        x_train.append(d[0])
        y_train.append(d[1])
        cl = np.where(d[1])[0][0]
        tr_stat[cl] = tr_stat[cl] + 1
    for d in ts_data:
        x_test.append(d[0])
        y_test.append(d[1])
        cl = np.where(d[1])[0][0]
        ts_stat[cl] = ts_stat[cl] + 1

    print("Generated training set")
    print("====================================================================")
    print("Training set size: " + str(len(y_train)))
    print("Training set positive: " + str(tr_stat[0]))
    print("Training set negative: " + str(tr_stat[1]))
    print("Test set positive: " + str(ts_stat[0]))
    print("Test set negative: " + str(ts_stat[1]))
    print("====================================================================")
    pickle.dump(ga, open("ga.p", "wb"))
    del data
    del ga
    del tr_data
    del ts_data
    del fasta
    gc.collect()
    pickle.dump(x_train, open("x_train.p", "wb"))
    pickle.dump(y_train, open("y_train.p", "wb"))
    pickle.dump(x_test, open("x_test.p", "wb"))
    pickle.dump(y_test, open("y_test.p", "wb"))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

num_classes = 2
seq_len = 1001

best_mcc = -1
patience = 30
wait = 0
batch_size = 128
nb_epoch = 300
x = tf.placeholder(tf.float32, shape=[None, seq_len, max_features], name="input_prom")
y = tf.placeholder(tf.float32, shape=[None, num_classes])
in_training_mode = tf.placeholder(tf.bool, name="in_training_mode")
keep_prob = tf.placeholder(tf.float32, name="kr")
out, loss = model(x, y, keep_prob, in_training_mode)
out = tf.identity(out, name="output_prom")
extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
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
        start = time.time()
        x_train_shift = []
        gc.collect()
        print("Adding shift")
        for i in range(len(y_train)):
            if y_train[i][num_classes - 1] == 1 or scan_model:
                rshift = randint(0, 200)
                x_train_shift.append(x_train[i][rshift: rshift + seq_len])
            else:
                x_train_shift.append(x_train[i][100: 100 + seq_len])
        x_test_shift = []
        for i in range(len(y_test)):
            x_test_shift.append(x_test[i][100: 100 + seq_len])
        end = time.time()
        print("Done, elapsed time " + str(end - start))
        total = int(math.ceil(float(len(x_train_shift)) / batch_size))
        sp = 0
        for i in range(total):
            feed = {x: x_train_shift[i * batch_size:(i + 1) * batch_size],
                    y: y_train[i * batch_size:(i + 1) * batch_size], keep_prob: kp, in_training_mode: True}
            train_step.run(feed_dict=feed)
        pred_tr = []
        if epoch % 1 == 0:
            pred_tr = brun(sess, x, out, x_train_shift, keep_prob, in_training_mode)
            trloss = test_loss(pred_tr, y_train)
            pred = brun(sess, x, out, x_test_shift, keep_prob, in_training_mode)
            tsloss = test_loss(pred, y_test)

            ts = time.gmtime()
            print(str(epoch) + " [" + time.strftime("%Y-%m-%d %H:%M:%S", ts) + "] ", end="")
            print("Training sn, sp, mcc: ", end="")
            tp, tn, fp, fn, sn, sp, mcc = test_mcc(pred_tr, y_train)
            print(format(sn, '.2f') + " " + format(sp, '.2f') + " " + format(mcc, '.2f') + " ", end="")
            print("Validation: ", end="")
            tp, tn, fp, fn, sn, sp, mcc = test_mcc(pred, y_test)
            print(format(sn, '.2f') + " " + format(sp, '.2f') + " " + format(mcc, '.2f') + " ")
            saver.save(sess, "./store/" + out_dir + "/" + str(epoch) + ".ckpt")
            if mcc > best_mcc:  # and sp >= 0.99
                best = epoch
                print("------------------------------------best so far")
                best_mcc = mcc

            print("Training set loss: " + str(trloss))
            print("Test set loss: " + str(tsloss))

        del (x_train_shift)
        del (x_test_shift)
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
