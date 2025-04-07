import numpy as np
import pandas as pd
import tensorflow as tf

def calculate_ivrmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def calculate_r_oos(y_true, y_pred, all_points=False):
    if not all_points:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        mean_IV = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True) # should be shape of 114 long
        # print(mean_IV[0], mean_IV[1], mean_IV[2])
        # print(y_true[0], y_true[1], y_true[2])
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV))
        # print((y_true - mean_IV)[0])
        r2 = 1 - ss_res/ss_tot
    else:
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2])
        mean_IV = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_IV), axis=[1, 2])
        r2 = 1 - ss_res/ss_tot
    return r2.numpy()