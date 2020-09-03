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
import random
from random import randint, shuffle
from tensorflow.python.keras import layers
import gc


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


def model(x, y, keep_ratio, in_training_mode):
    lc1 = layers.Conv1D(64, 3, padding='same', use_bias=True)(x)
    lc1 = layers.LeakyReLU(alpha=0.2)(lc1)
    lc2 = layers.Conv1D(128, 3, padding='same', use_bias=True)(lc1)
    lc2 = layers.LeakyReLU(alpha=0.2)(lc2)
    dp = tf.nn.dropout(lc2, keep_ratio)
    out, W = fully_connected(dp, "output_p", num_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)
    out = tf.nn.softmax(out)
    return out, loss


def brun(sess, x, y, a, keep_prob, in_training_mode):
    preds = []
    batch_size = 256
    number_of_full_batch = int(math.ceil(float(len(a)) / batch_size))
    for i in range(number_of_full_batch):
        preds += list(sess.run(y, feed_dict={x: np.asarray(a[i * batch_size:(i + 1) * batch_size]),
                                             keep_prob: 1.0, in_training_mode: False}))
    return preds


#data_folder = "/home/user/data/DeepRAG/"
#os.chdir(data_folder)

x_train = pickle.load(open("x_train.p", "rb"))
x_test = pickle.load(open("x_test.p", "rb"))
y_train = pickle.load(open("y_train.p", "rb"))
y_test = pickle.load(open("y_test.p", "rb"))
print("====================================================================")
print("Loaded existing training set")
print("Training set size: " + str(len(y_train)))
print("====================================================================")


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

num_classes = 2
seq_len = 1001
shift = 50
out_dir = "model_predict_simple"

best_mcc = -1
patience = 30
wait = 0
batch_size = 128
nb_epoch = 300
x = tf.placeholder(tf.float32, shape=[None, seq_len, 4], name="input_prom")
y = tf.placeholder(tf.float32, shape=[None, num_classes])
in_training_mode = tf.placeholder(tf.bool, name="in_training_mode")
keep_prob = tf.placeholder(tf.float32, name="kr")
out, loss = model(x, y, keep_prob, in_training_mode)
out = tf.identity(out, name="output_prom")
extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
kp = 0.5
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
        # Shuffle training set
        rng_state = np.random.get_state()
        np.random.shuffle(x_train)
        np.random.set_state(rng_state)
        np.random.shuffle(y_train)
        np.random.set_state(rng_state)
        x_train_shift = []
        gc.collect()
        # Adding shift
        y_train_shift = []
        for i in range(len(y_train)):
            # Original negatives
            y_train_shift.append(y_train[i])
            x_train_shift.append(x_train[i][shift: shift + seq_len])
            # Shifted negatives
            if y_train[i][num_classes - 1] == 1:
                rshift = randint(0, 2 * shift)
                x_train_shift.append(x_train[i][rshift: rshift + seq_len])
                y_train_shift.append(y_train[i])
        # No shifting for the validation
        x_test_shift = []
        for i in range(len(y_test)):
            x_test_shift.append(x_test[i][shift: shift + seq_len])
        total = int(math.ceil(float(len(x_train_shift)) / batch_size))
        sp = 0
        for i in range(total):
            feed = {x: x_train_shift[i * batch_size:(i + 1) * batch_size],
                    y: y_train_shift[i * batch_size:(i + 1) * batch_size], keep_prob: kp, in_training_mode: True}
            train_step.run(feed_dict=feed)
        pred_tr = []
        # Evaluate the performance
        if epoch % 1 == 0:
            pred_tr = brun(sess, x, out, x_train_shift, keep_prob, in_training_mode)
            trloss = test_loss(pred_tr, y_train_shift)
            pred = brun(sess, x, out, x_test_shift, keep_prob, in_training_mode)
            tsloss = test_loss(pred, y_test)

            ts = time.gmtime()
            print(str(epoch) + " [" + time.strftime("%Y-%m-%d %H:%M:%S", ts) + "] ", end="")
            print("Training sn, sp, mcc: ", end="")
            tp, tn, fp, fn, sn, sp, mcc = test_mcc(pred_tr, y_train_shift)
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

        del x_train_shift
        del x_test_shift
        gc.collect()

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
