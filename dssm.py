import os
import os.path
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
from data_prepare import load_data, build_vocab, get_idx_from_data,prepare_test_data
from cnn_encoder import inference, calloss, training, evaluation,prediction
from utilities import save_obj, load_obj,save_data_mapping

SAVEDRI="./model"
MAX_EPOCHS = 10
SUMMARY_FREQ = 10
EVAL_FREQ = 10
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
IS_TRAIN = False
    
def input_placeholder(max_len):
    x_placeholder = tf.placeholder(tf.float32, shape=(None, max_len))
    y_placeholder = tf.placeholder(tf.float32, shape=(None, max_len))
    return x_placeholder, y_placeholder

def fill_feed_dict(x_minibatch, y_minibatch,x_placeholder,y_placeholder):
    feed_dict = {
        x_placeholder: x_minibatch,
        y_placeholder: y_minibatch,
    }
    return feed_dict

def do_eval(sess,
            r1, r3, r10, medr, meanr,h_meanr,v1,v2,v3,
            x_placeholder,
            y_placeholder,
            eidx,
            x,
            y):
            # Runs one evaluation against the full evaluation set
            r_at_10 = 0
            # Use np.random.seed and np.random.shuffle instead.
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
            value1 = .0
            value2 = .0
            value3 = .0
            for mini_batch in range(batch_count):
                x_minibatch = [x[seq] for seq in inds[mini_batch::batch_count]]
                y_minibatch = [y[seq] for seq in inds[mini_batch::batch_count]]
                feed_dict = fill_feed_dict(x_minibatch, y_minibatch, x_placeholder, y_placeholder)
                r_1, r_3, r_10, med_r, mean_r,h_mean_r,v_1,v_2,v_3 = sess.run([r1, r3, r10, medr, meanr,h_meanr,v1,v2,v3], feed_dict=feed_dict)
                rank1+=r_1
                rank3+=r_3
                rank10+=r_10
                med_rank+=med_r
                mean_rank+=mean_r
                h_mean_rank+=h_mean_r
                value1 += v_1
                value2 += v_2
                value3 += v_3
            print('Valid Rank: r1: %.2f, r3: %.2f, r10: %.2f, medr: %.2f, meanr: %.2f, h_meanr: %.2f, v1: %.5f, v2: %.5f, v3: %.5f' % (rank1/batch_count, rank3/batch_count, rank10/batch_count, med_rank/batch_count, mean_rank/batch_count,h_mean_rank/batch_count, value1/batch_count, value2/batch_count, value3/batch_count))
            return (rank1+rank3+rank10)/batch_count

def training_prediction(train, validation, test, n_char, max_len, char2ix, ix2char, iteration, l_rate):

    directory = SAVEDRI + '/' + str(iteration)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    with tf.Graph().as_default():
        #Generate placeholders for text pairs x and y.

        x_placeholder, y_placeholder = input_placeholder(max_len)

        #Add a Graph to compute the feature vectors.
        feature_x, feature_y = inference(x_placeholder, y_placeholder, n_char, EMBEDDING, FILTERS, FEATURE)

        #Add Ops for loss calculation to the Graph.
        cos_sim, loss = calloss(feature_x, feature_y, NCON, GAMMA, BS)

        #Add Ops that calculate and apply gradients.
        train_op = training(loss, l_rate)

        #Ops of evaluation: r1+r3+r10
        r1, r3, r10, medr, meanr,h_meanr, v1, v2, v3 = evaluation(cos_sim)

        #Ops of Prediction
        predict = prediction(feature_x, feature_y, GAMMA)

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
        summary_writer = tf.summary.FileWriter(directory, sess.graph)

        #Run the Op to initialize the variables
        sess.run(init)
        #Start training loop.
        uidx = 0
        curr_score = 0
        if (IS_TRAIN):
            x_train = np.array(train[0])
            y_train = np.array(train[1])
            x_validation = np.array(validation[0])
            y_validation = np.array(validation[1])
            x_test = np.array(test[0])
            y_test = np.array(test[1])
            
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
                    if np.mod(uidx+1, SUMMARY_FREQ) == 0:
                        print('(%d/%d Iteration: %d: Epoch %d) loss = %.2f (%.3f sec)' % (mini_batch, batch_count, iteration, eidx, loss_value, duration))
                        summary_str = sess.run(summary, feed_dict = feed_dict)
                        summary_writer.add_summary(summary_str, uidx)
                        summary_writer.flush()
                    
                    #Save checkpoint and evaluate the model periodically.
                    if np.mod(uidx+1, EVAL_FREQ) == 0:       
                        # Evaluate against the training set.
                        print('(%d/%d Iteration: %d: Epoch %d) Training Data Eval:' % (mini_batch, batch_count, iteration, eidx))
                        _ = do_eval(sess,
                                r1, r3, r10, medr, meanr,h_meanr,v1,v2,v3,
                                x_placeholder,
                                y_placeholder,
                                eidx,
                                x_minibatch,
                                y_minibatch)

                        # Evaluate against the validation set.                    
                        print('(%d/%d Iteration: %d: Epoch %d) Validation Data Eval:' % (mini_batch, batch_count, iteration, eidx))
                        r_at_10 = do_eval(sess,
                                    r1, r3, r10, medr, meanr,h_meanr,v1,v2,v3,
                                    x_placeholder,
                                    y_placeholder,
                                    eidx,
                                    x_validation,
                                    y_validation)
                        curr_score = r_at_10
                        '''print('Save model.')
                        checkpoint_file = os.path.join(directory, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step=uidx)
                        if r_at_10 > curr_score:
                            curr_score = r_at_10
                            print('Save model.')
                            checkpoint_file = os.path.join(directory, 'model.ckpt')
                            saver.save(sess, checkpoint_file, global_step=uidx)
                        else:
                            l_rate = 0.9*l_rate'''
                        # Evaluate against the test set.
                        print('(%d/%d Iteration: %d: Epoch %d) Test Data Eval:' % (mini_batch, batch_count, iteration, eidx))
                        do_eval(sess,
                                r1, r3, r10, medr, meanr,h_meanr,v1,v2,v3,
                                x_placeholder,
                                y_placeholder,
                                eidx,
                                x_test,
                                y_test)
                print('Save model.')
                checkpoint_file = os.path.join(directory, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=uidx)
        else:
            test_x, test_y = prepare_test_data(char2ix, ix2char, max_len)
            saver.restore(sess, tf.train.latest_checkpoint('D:/GitHub/DSSM/model/1/'))
            print("restore model.")
            x = np.array(test_x)
            y = np.array(test_y)        
            feed_dict = fill_feed_dict(x, y, x_placeholder, y_placeholder)
            x_f, y_f, predict_value = sess.run([feature_x, feature_y, predict], feed_dict = feed_dict)
            print(predict_value)
        return

def main(_):
    if IS_TRAIN:
        run_training()
    else:
        run_prediction()

def run_training():
    x, y, n_chars, max_len, char2ix, ix2char = load_data()
    save_data_mapping(n_chars, max_len, char2ix, ix2char, os.path.join(SAVEDRI, data_mapping_file))
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
        training_prediction(train, valid, test, n_chars, max_len,char2ix, ix2char, i, LRATE)

def run_prediction():
    data_mapping = load_obj(os.path.join(SAVEDRI, 'data_mapping.pkl'))
    training_prediction([], [], [], data_mapping['n_chars'], data_mapping['max_len'],data_mapping['char2ix'], data_mapping['ix2char'], 0, LRATE)

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

if __name__=="__main__":  
    os.chdir(sys.path[0])
    tf.app.run(main=main)