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
LRATE = 0.001
SEED = 1234
VALIDATION_RATIO = 0.1
FOLD_COUNT = 10

def inference(x, y, n_chars):
    batch_size = x.shape[0].value
    max_len = x.shape[1].value
    with tf.name_scope('embedding'):
        w_emb = tf.get_variable("c_embed", [n_chars , EMBEDDING], trainable=False)
        
        x_flatten_emb = tf.transpose(tf.nn.embedding_lookup(w_emb,tf.cast(tf.reshape(x,[1,(x.shape[0]*x.shape[1]).value]), dtype='int32')))
        y_flatten_emb = tf.transpose(tf.nn.embedding_lookup(w_emb,tf.cast(tf.reshape(y,[1,(y.shape[0]*y.shape[1]).value]), dtype='int32')))

        x_emb = tf.transpose(tf.reshape(x_flatten_emb,[x.shape[0].value, 1, EMBEDDING, x.shape[1].value]), [0, 2, 3, 1])
        y_emb = tf.transpose(tf.reshape(y_flatten_emb,[y.shape[0].value, 1, EMBEDDING, y.shape[1].value]), [0, 2, 3, 1])
    
    with tf.name_scope('convLayers'):
        W_conv = OrderedDict()
        b_conv = OrderedDict()

        cov_output_x = OrderedDict()
        cov_output_y = OrderedDict()

        for i in range(len(FILTERS)):
            W_conv[i] = weight_variable([EMBEDDING, FILTERS[i], 1, FEATURE])
            b_conv[i] = bias_variable([FEATURE])
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
        x_feature = tf.tile(feature_x, [NCON + 1, 1])
        y_feature = tf.slice(feature_y, [0, 0], [0, -1])
        for i in range(BS):
            rotation = tf.concat([tf.slice(temp, [i, 0], [BS - i, -1]), tf.slice(temp, [0, 0], [i, -1])], 0)
            cy_rows = tf.gather(rotation, random.sample(range(1,BS), NCON))
            y_feature = tf.concat([y_feature, tf.slice(rotation,[0,0],[1,-1]), cy_rows], 0)        
    with tf.name_scope('Cosine_Similarity'):
        # Cosine similarity
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x_feature), 1, True))
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y_feature), 1, True))

        prod = tf.reduce_sum(tf.multiply(x_feature, y_feature), 1, True)
        norm_prod = tf.multiply(x_norm, y_norm)

        cos_sim_raw = tf.truediv(prod, norm_prod)
        cos_sim = tf.reshape(cos_sim_raw, [BS, NCON + 1]) * GAMMA

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
    r = []
    for index in range(npts):
        top_value, top_indices = tf.nn.top_k(tf.reshape(cos_sim[index],[1,cos_sim.shape[1].value]), k=cos_sim.shape[1].value)
        where = tf.where(tf.equal(top_indices, 0))
        result = tf.segment_min(where[:, 1], where[:, 0])
        r.append(result)
                
    # Compute metrics
    ranks = tf.reshape(r,[1,len(r)])
    r1 = 100 * tf.count_nonzero(tf.less(ranks, 1)) / ranks.shape[1].value
    r3 = 100 * tf.count_nonzero(tf.less(ranks, 3)) / ranks.shape[1].value
    r10 = 100 * tf.count_nonzero(tf.less(ranks, 10)) / ranks.shape[1].value
    medr = tf.nn.top_k(ranks, ranks.shape[1].value//2).values[0][-1] + 1
    meanr = tf.reduce_mean(ranks) + 1
    h_meanr = 1/tf.reduce_mean(1/(ranks+1))
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
            r1, r3, r10, medr, meanr,h_meanr,
            x_placeholder,
            y_placeholder,
            eidx,
            x,
            y):
  # Runs one evaluation against the full evaluation set
  r_at_10 = 0
  prng = RandomState(SEED-eidx-1)
  num_samples = len(x)
  inds = np.arange(num_samples)
  prng.shuffle(inds)
  batch_count = len(inds)//BS
  rank1 =.0
  rank3 = .0
  rank10 = .0
  med_rank = .0
  mean_rank = .0
  h_mean_rank=.0
  for mini_batch in range(batch_count):
      x_minibatch = [x[seq] for seq in inds[mini_batch::batch_count]]
      y_minibatch = [y[seq] for seq in inds[mini_batch::batch_count]]
      feed_dict = fill_feed_dict(x_minibatch, y_minibatch, x_placeholder, y_placeholder)
      r_1, r_3, r_10, med_r, mean_r,h_mean_r = sess.run([r1, r3, r10, medr, meanr,h_meanr], feed_dict=feed_dict)
      rank1+=r_1
      rank3+=r_3
      rank10+=r_10
      med_rank+=med_r
      mean_rank+=mean_r
      h_mean_rank+=h_mean_r
  print('Valid Rank: r1: %.f, r3: %.f, r10: %.f, medr: %d, meanr: %.f, h_meanr: %.f' % (rank1/batch_count, rank3/batch_count, rank10/batch_count, med_rank/batch_count, mean_rank/batch_count,h_mean_rank/batch_count))
  return (rank1+rank3+rank10)/batch_count

def run_training(train, validation, test, n_char, max_len, iteration):

    directory = SAVEDRI + '/' + str(iteration)
    if not os.path.exists(directory):
        os.makedirs(directory)

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
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

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

                _, loss_value, cos = sess.run([train_op, loss, cos_sim], feed_dict=feed_dict)
                
                duration = time.time() - start_time

                #Write the summaries and print an overview
                if np.mod(uidx+1, SUMMARY_FREQ):
                    print('Iteration: %d: Epoch %d: loss = %.2f (%.3f sec)' % (iteration, eidx, loss_value, duration))
                    summary_str = sess.run(summary, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_str, uidx)
                    summary_writer.flush()
                    # Evaluate against the training set.
                    print('Training Data Eval on iteration %d' % (iteration))
                    _ = do_eval(sess,
                            r1, r3, r10, medr, meanr,h_meanr,
                            x_placeholder,
                            y_placeholder,
                            eidx,
                            x_minibatch,
                            y_minibatch)

                #Save checkpoint and evaluate the model periodically.
                if np.mod(uidx+1, EVAL_FREQ):                                        
                    # Evaluate against the validation set.                    
                    print('Validation Data Eval on iteration %d' % (iteration))
                    r_at_10 = do_eval(sess,
                                r1, r3, r10, medr, meanr,h_meanr,
                                x_placeholder,
                                y_placeholder,
                                eidx,
                                x_validation,
                                y_validation)
                    if r_at_10 > curr_score:
                        curr_score = r_at_10
                        print('Save model.')
                        checkpoint_file = os.path.join(directory, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=uidx)

                    # Evaluate against the test set.
                    print('Test Data Eval on iteration %d' % (iteration))
                    do_eval(sess,
                            r1, r3, r10, medr, meanr,h_meanr,
                            x_placeholder,
                            y_placeholder,
                            eidx,
                            x_test,
                            y_test)
        return r1, r3, r10, medr, meanr, h_meanr

def main(_):
    x, y, n_chars, max_len, char2ix, ix2char = load_data()
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
        r1, r3, r10, medr, meanr,h_meanr = run_training(train, valid, test, n_chars, max_len, i)

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

def conv2d(x, W, format='NHWC'):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, x.shape[1].value, 1, 1], padding='SAME', data_format=format)

def max_pool(x,format='NHWC'):
      return tf.nn.max_pool(x, ksize=[1, 1, x.shape[2].value, 1],strides=[1, 1, x.shape[2].value, 1], padding='SAME', data_format=format)

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