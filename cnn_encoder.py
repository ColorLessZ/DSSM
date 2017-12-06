import os
import sys
import logging
import random
import time

import tensorflow as tf
import numpy as np
import _pickle as cPickle
import scipy.stats as ss

from collections import OrderedDict
from numpy.random import RandomState

def inference(x, y, n_chars, embedding_size, filters, feature_size):
    max_len = x.shape[1].value
    with tf.name_scope('embedding_layer'):
        w_emb = tf.get_variable("c_embed", [n_chars , embedding_size])    
        x_emb = tf.expand_dims(tf.nn.embedding_lookup(w_emb, tf.cast(x, dtype='int32')), -1)
        y_emb = tf.expand_dims(tf.nn.embedding_lookup(w_emb, tf.cast(y, dtype='int32')), -1)
    with tf.name_scope('conv_layers'):
        W_conv = OrderedDict()
        b_conv = OrderedDict()

        cov_output_x = OrderedDict()
        cov_output_y = OrderedDict()

        for i in range(len(filters)):
            W_conv[i] = weight_variable([filters[i], embedding_size, 1, feature_size], "conv_" + str(i) + "_weight")
            b_conv[i] = bias_variable([feature_size], "conv_" + str(i) + "_bias")
            cov_output_x[i] = tf.nn.relu(conv2d(x_emb, W_conv[i]) + b_conv[i])
            cov_output_y[i] = tf.nn.relu(conv2d(y_emb, W_conv[i]) + b_conv[i])
    
    with tf.name_scope('maxpooling_layer'):
        pooling_output_x = []
        pooling_output_y = []
        for i in range(len(filters)):
            pooling_output_x.append(max_pool(cov_output_x[i]))
            pooling_output_y.append(max_pool(cov_output_y[i]))
        feature_x = tf.reshape(tf.concat(pooling_output_x, 3),[-1,len(filters)*feature_size])
        feature_y = tf.reshape(tf.concat(pooling_output_y, 3),[-1,len(filters)*feature_size])
        """if needed drop out here"""
    return feature_x, feature_y

def calloss(feature_x, feature_y, negtive_count, gamma, batch_size):
    #TODO: passing cy in instead of get them in mini batch
    with tf.name_scope('FD_rotate'):
        # Rotate FD+ to produce 50 FD-
        temp_x = tf.tile(feature_x, [1, 1])
        temp_y = tf.tile(feature_y, [1, 1])
        x_feature = tf.slice(feature_x,  [0, 0], [0, -1])
        y_feature = tf.slice(feature_y, [0, 0], [0, -1])
        for i in range(batch_size):
            rotation = tf.concat([tf.slice(temp_y, [i, 0], [batch_size - i, -1]), tf.slice(temp_y, [0, 0], [i, -1])], 0)
            cy_rows = tf.gather(rotation, random.sample(range(1,batch_size), negtive_count))
            y_feature = tf.concat([y_feature, tf.slice(rotation,[0,0],[1,-1]), cy_rows], 0)
            x_feature = tf.concat([x_feature, tf.tile(tf.slice(temp_x,[i, 0], [1,-1]), [negtive_count + 1, 1])] , 0)   
    with tf.name_scope('Cosine_Similarity'):
        # Cosine similarity
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x_feature), 1, True))
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y_feature), 1, True))

        prod = tf.reduce_sum(tf.multiply(x_feature, y_feature), 1, True)
        norm_prod = tf.multiply(x_norm, y_norm) + 1e-20

        cos_sim_raw = tf.truediv(prod, norm_prod)
        cos_sim = tf.reshape(cos_sim_raw, [batch_size, negtive_count + 1]) * gamma
    with tf.name_scope('Loss'):
        # Train Loss
        # investigation on loss calculation
        prob = tf.nn.softmax((cos_sim)) 
        hit_prob = tf.slice(prob, [0, 0], [-1, 1]) #P_Q_D+
        loss = -tf.reduce_sum(tf.log(hit_prob)) / batch_size
    return cos_sim, loss

def training(loss, lrate):
    with tf.name_scope('Training'):
        tf.summary.scalar('loss', loss)
        # global_step = tf.train.get_or_create_global_step()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Optimizer
        train_op = tf.train.AdamOptimizer(lrate).minimize(loss, global_step = global_step)
        return train_op
    
def evaluation(cos_sim):
    with tf.name_scope('Evaluation'):
        npts = cos_sim.shape[0].value
        r = []
        v1 = []
        v2 = []
        v3 = []
        for index in range(npts):
            cos_sim_reshape = tf.reshape(cos_sim[index],[1,cos_sim.shape[1].value])
            top_value, top_indices = tf.nn.top_k(cos_sim_reshape, k=cos_sim.shape[1].value)
            where = tf.where(tf.equal(top_indices, 0))
            result = tf.segment_min(where[:, 1], where[:, 0])
            r.append(result)
            v1.append(top_value[0][0])
            v2.append(top_value[0][1])
            v3.append(top_value[0][2])
                    
        # Compute metrics
        ranks = tf.reshape(r,[1,len(r)])
        
        r1_mask = tf.less(ranks, 1)
        r3_mask = tf.less(ranks, 3)
        r10_mask = tf.less(ranks, 10)
        r1 = 100 * tf.count_nonzero(r1_mask) / ranks.shape[1].value
        r3 = 100 * tf.count_nonzero(r3_mask) / ranks.shape[1].value
        r10 = 100 * tf.count_nonzero(r10_mask) / ranks.shape[1].value
        medr = tf.nn.top_k(ranks, ranks.shape[1].value//2).values[0][-1] + 1
        meanr = tf.reduce_mean(ranks) + 1
        h_meanr = 1/tf.reduce_mean(1/(ranks+1))
        tf.summary.scalar('r1', r1)
        tf.summary.scalar('r3', r3)
        tf.summary.scalar('r10', r10)
        
        top_1_value = tf.reshape(v1,[1,len(v1)])
        top_2_value = tf.reshape(v2,[1,len(v2)])
        top_3_value = tf.reshape(v3,[1,len(v3)])
        v1 = tf.reduce_mean(top_1_value)
        v2 = tf.reduce_mean(top_2_value)
        v3 = tf.reduce_mean(top_3_value)

        tf.summary.scalar('v1', v1)
        tf.summary.scalar('v2', v2)
        tf.summary.scalar('v3', v3)
    
    return r1, r3, r10, medr, meanr,h_meanr, v1, v2, v3

def prediction(feature_x, feature_y, gamma):
    x_norm = tf.sqrt(tf.reduce_sum(tf.square(feature_x), 1, True))
    y_norm = tf.sqrt(tf.reduce_sum(tf.square(feature_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(feature_x, feature_y), 1, True)
    norm_prod = tf.multiply(x_norm, y_norm)

    cos_sim_raw = tf.truediv(prod, norm_prod + 1e-10) * gamma
    return cos_sim_raw

def conv2d(x, W, format='NHWC'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, x.shape[2].value, 1], padding='SAME', data_format=format)

def max_pool(x,format='NHWC'):
      return tf.nn.max_pool(x, ksize=[1, x.shape[1].value, 1, 1],strides=[1, 1, 1, 1], padding='VALID', data_format=format)

def weight_variable(shape, name, stddev=0.01):
  """weight_variable generates a weight variable of a given shape."""           
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
      """bias_variable generates a bias variable of a given shape."""
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name=name)

def uniform_weight(nin,nout=None, scale=1):
    if nout == None:
        nout = nin
    initial = tf.random_uniform(shape=(nin, nout), minval=-scale, maxval=scale)
    return tf.Variable(initial)