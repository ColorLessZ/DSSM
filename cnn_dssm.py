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
from sklearn.model_selection import KFold
from dataprepare import load_data, build_vocab, get_idx_from_data

SAVEDRI="./model"
MAX_EPOCHS = 20
SUMMARY_FREQ = 100
EVAL_FREQ = 100
BS = 100
FEATURE = 100
EMBEDDING = 128
FILTERS = [2,3,4]
NCON = 50
GAMMA = 10
LRATE = 0.0002
SEED = 1234
VALIDATION_RATIO = 0.1
FOLD_COUNT = 10

def inference(x, y, n_chars):
    batch_size = x.shape[0].value
    max_len = x.shape[1].value
    with tf.name_scope('embedding'):
        w_emb = tf.get_variable("char_embed", [n_chars , EMBEDDING])
        
        x_flatten_emb = tf.transpose(tf.nn.embedding_lookup(w_emb,tf.cast(tf.reshape(x,[1,(x.shape[0]*x.shape[1]).value]), dtype='int32')))
        y_flatten_emb = tf.transpose(tf.nn.embedding_lookup(w_emb,tf.cast(tf.reshape(y,[1,(y.shape[0]*y.shape[1]).value]), dtype='int32')))

        x_emb = tf.reshape(x_flatten_emb,[x.shape[0].value, 1, EMBEDDING, x.shape[1].value])
        y_emb = tf.reshape(y_flatten_emb,[y.shape[0].value, 1, EMBEDDING, y.shape[1].value])
    
    with tf.name_scope('convLayers'):
        W_conv = OrderedDict()
        b_conv = OrderedDict()

        cov_output_x = OrderedDict()
        cov_output_y = OrderedDict()

        for i in range(len(FILTERS)):
            W_conv[i] = weight_variable([EMBEDDING, FILTERS[i], 1, FEATURE])
            b_conv[i] = bias_variable([max_len])
            cov_output_x[i] = tf.nn.relu(conv2d(x_emb, W_conv[i]) + b_conv[i])
            cov_output_y[i] = tf.nn.relu(conv2d(y_emb, W_conv[i]) + b_conv[i])
    
    with tf.name_scope('maxpooling'):
        pooling_output_x = []
        pooling_output_y = []
        for i in range(len(FILTERS)):
            pooling_output_x.append(max_pool(cov_output_x[i]))
            pooling_output_y.append(max_pool(cov_output_y[i]))
        feature_x = tf.reshape(tf.concat(pooling_output_x, 1),[batch_size,len(FILTERS)*FEATURE])
        feature_y = tf.reshape(tf.concat(pooling_output_y, 1),[batch_size,len(FILTERS)*FEATURE])
        """if needed drop out here"""

    return feature_x, feature_y

def calloss(feature_x, feature_y):
    with tf.name_scope('FD_rotate'):
        # Rotate FD+ to produce 50 FD-
        temp = tf.tile(feature_y, [1, 1])

        for i in range(NCON):
            rand = int((random.random() + i) * BS / NCON)
            feature_y = tf.concat([feature_y,
                            tf.slice(temp, [rand, 0], [BS - rand, -1]),
                            tf.slice(temp, [0, 0], [rand, -1])], 0)

    with tf.name_scope('Cosine_Similarity'):
        # Cosine similarity
        x_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(feature_x), 1, True)), [NCON + 1, 1])
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(feature_y), 1, True))

        prod = tf.reduce_sum(tf.multiply(tf.tile(feature_x, [NCON + 1, 1]), feature_y), 1, True)
        norm_prod = tf.multiply(x_norm, y_norm)

        cos_sim_raw = tf.truediv(prod, norm_prod)
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NCON + 1, BS])) * GAMMA

    with tf.name_scope('Loss'):
        # Train Loss
        # investigation on loss calculation
        prob = tf.nn.softmax((cos_sim)) 
        hit_prob = tf.slice(prob, [0, 0], [-1, 1]) #P_Q_D+
        loss = -tf.reduce_sum(tf.log(hit_prob)) / BS    
    return cos_sim, loss

def training(loss, lrate):
    with tf.name_scope('Training'):
        tf.summary.scalar('loss', loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Optimizer
        train_op = tf.train.AdamOptimizer(lrate).minimize(loss, global_step = global_step)
        return train_op

def evaluation(cos_sim):
    with tf.name_scope('Evaluation'):
        npts = cos_sim.shape[0].value
    
    index_list = []

    ranks = np.zeros(npts)
    for index in range(npts):
        orders = ss.rankdata(tf.reshape(cos_sim[index],[1,cos_sim.shape[1].value]))
        ranks[index] =  orders[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = np.mean(ranks) + 1
    h_meanr = 1./np.mean(1./(ranks+1))
    return r1, r3, r10, medr, meanr,h_meanr
    
def input_placeholder(batch_size, max_len):
    x_placeholder = tf.placeholder(tf.float32, shape=(batch_size, max_len))
    y_placeholder = tf.placeholder(tf.float32, shape=(batch_size, max_len))
    return x_placeholder, y_placeholder

def fill_feed_dict(x_minibatch, y_minibatch, x_placeholder, y_placeholder):
    feed_dict = {
        x_placeholder: x_minibatch,
        y_placeholder: y_minibatch,
    }
    return feed_dict

def do_eval(sess,
            r1,r3,r10,
            x_placeholder,
            y_placeholder,
            x,
            y):
  # Runs one evaluation against the full evaluation set
  r_at_10 = 0
  prng = RandomState(SEED-eidx-1)
  num_samples = len(x)
  inds = np.arange(num_samples)
  prng.shuffle(inds)
  batch_count = len(inds)/BS

  for mini_batch in range(batch_count):
      feed_dict = fill_feed_dict(x_minibatch, y_minibatch, x_placeholder, y_placeholder)
      r1, r3, r10, medr, meanr,h_meanr = sess.run(evaluation, feed_dict=feed_dict)
      print('Valid Rank: r1: %.f, r3: %.f, r10: %.f, medr: %d, meanr: %.f, h_meanr: %.f' % (r1, r3, r10, medr, meanr,h_meanr))
  return r1+r3+r10

def run_training(train, validation, test, n_char, max_len):
    x_train = np.array(train[0])
    y_train = np.array(train[1])
    x_validation = np.array(validation[0])
    y_validation = np.array(validation[1])
    x_test = np.array(test[0])
    y_test = np.array(test[1])

    with tf.Graph().as_default():
        #Generate placeholders for text pairs x and y.

        x_placeholder, y_placeholder = input_placeholder(BS, max_len)

        #Add a Graph to compute the feature vectors.
        feature_x, feature_y = inference(x_placeholder, y_placeholder, n_char)

        #Add Ops for loss calculation to the Graph.
        cos_sim, loss = calloss(feature_x, feature_y)

        #Add Ops that calculate and apply gradients.
        train_op = training(loss, LRATE)

        #Ops of evaluation: r1+r3+r10
        r1, r3, r10, medr, meanr,h_meanr = evaluation(cos_sim)

        #Build the summary tensor based ont he tf collection of summaries.
        summary = tf.summary.merge_all()

        #Add the variable initializer Op.
        init = tf.global_variables_initializer()

        #Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        #Create a session for running Ops on the Graph.
        sess = tf.Session()

        #Instantiate a summery Writer to output summaries and the Graph
        summary_writer = tf.summary.FileWriter(SAVEDRI, sess.graph)

        #Run the Op to initialize the variables
        sess.run(init)

        #Start training loop.
        uidx = 0
        curr_score = 0
        for eidx in range(MAX_EPOCHS):
            prng = RandomState(SEED-eidx-1)
            num_samples = len(x_train)
            inds = np.arange(num_samples)
            prng.shuffle(inds)
            batch_count = len(inds)//BS

            for mini_batch in range(batch_count):
                start_time = time.time()
                uidx += 1
                x_minibatch = [x_train[seq] for seq in inds[mini_batch::batch_count]]
                y_minibatch = [y_train[seq] for seq in inds[mini_batch::batch_count]]
                feed_dict = fill_feed_dict(x_minibatch, y_minibatch, x_placeholder, y_placeholder)

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time

                #Write the summaries and print an overview
                if np.mod(uidx, SUMMARY_FREQ):
                    print('Epoch %d: loss = %.2f (%.3f sec)' % (edix, loss_value, duration))
                    summary_str = sess.run(summary, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_str, edix)
                    summary_writer.flush()

                #Save checkpoint and evaluate the model periodically.
                if np.mod(uidx, EVAL_FREQ):                    
                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    _ = do_eval(sess,
                            r1, r3, r10,
                            x_placeholder,
                            y_placeholder,
                            x_train,
                            y_train)
                    # Evaluate against the validation set.                    
                    print('Validation Data Eval:')
                    r_at_10 = do_eval(sess,
                                r1, r3, r10,
                                x_placeholder,
                                y_placeholder,
                                x_validation,
                                y_validation)
                    if r_at_10 > curr_score:
                        #TODO: lr decay
                        print('Save model.')
                        checkpoint_file = os.path.join(SAVEDRI, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=uidx)

                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    do_eval(sess,
                            r1, r3, r10,
                            x_placeholder,
                            y_placeholder,
                            x_test,
                            y_test)
        return r1, r3, r10, medr, meanr,h_meanr 

def main(_):
    x, y, n_chars, max_len = load_data()
    kf = KFold(n_splits=FOLD_COUNT, shuffle=True, random_state=1234)
    results = []
    i = 0
    for train_index, test_index in kf.split(x):
        train_index = train_index.tolist()
        test_index = test_index.tolist()
    
        x_train = [x[ix] for ix in train_index]
        y_train = [y[ix] for ix in train_index]
    
        x_test = [x[ix] for ix in test_index]
        y_test = [y[ix] for ix in test_index]
            
        train = (x_train, y_train)
        test = (x_test, y_test)

        train, valid = create_valid(train, valid_portion=VALIDATION_RATIO) 
        i += 1       
        r1, r3, r10, medr, meanr,h_meanr = run_training(train, valid, test, n_chars, max_len)
        print('cv %d: r1: %.f, r3: %.f, r10: %.f, medr: %d, meanr: %.f, h_meanr: %.f' % (i, r1, r3, r10, medr, meanr,h_meanr))

def create_valid(train_set,valid_portion=VALIDATION_RATIO):
    
    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)

    return train, valid

def conv2d(x, W, format='NCHW'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[x.shape[0].value, 1, x.shape[2].value, 1], padding='SAME', data_format=format)

def max_pool(x,format='NCHW'):
      return tf.nn.max_pool(x, ksize=[1, 1, 1, x.shape[3].value],strides=[1, 1, 1, x.shape[3].value], padding='SAME', data_format=format)

def weight_variable(shape, stddev=0.01):
  """weight_variable generates a weight variable of a given shape."""           
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial)

def bias_variable(shape):
      """bias_variable generates a bias variable of a given shape."""
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

def uniform_weight(nin,nout=None, scale=1):
    if nout == None:
        nout = nin
    initial = tf.random_uniform(shape=(nin, nout), minval=-scale, maxval=scale)
    return tf.Variable(initial)

if __name__=="__main__":  
    os.chdir(sys.path[0])
    tf.app.run(main=main)