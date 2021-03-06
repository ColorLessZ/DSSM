import os
import sys
import numpy as np
from collections import defaultdict
import _pickle as cPickle

def load_test_data(char2ix, ix2char, max_len, path = './data/dataset_test.txt'): 
    textA, textB = [], []
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            text = line.lower().strip().split('\t')
            textA.append(text[0])
            textB.append(text[1])
    text = get_idx_from_data([textA,textB],char2ix,ix2char)
    
    x = []
    y = []
    
    for t in text[0]:
        x.append(pad_or_truncate(t, max_len, 0))
    for t in text[1]:
        y.append(pad_or_truncate(t, max_len, 0))
    
    return x, y

def load_data(path = './data/dataset_150K.txt'): 
    textA, textB = [], []
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            text = line.lower().strip().split('\t')
            textA.append(text[0])
            textB.append(text[1])
    char2ix, ix2char = build_vocab(textA, textB)
    text = get_idx_from_data([textA,textB],char2ix,ix2char)
    
    n_chars = len(ix2char)
    ix2char[0] = '<pad_zero>'
    char2ix['<pad_zero>'] = 0

    text0 = text[0] + text[1]
    length = []
    for sent in text0:
        length.append(len(sent))        
    max_len = np.max(length) + 1
    
    x = []
    y = []
    
    for t in text[0]:
        x.append(pad_or_truncate(t, max_len, 0))
    for t in text[1]:
        y.append(pad_or_truncate(t, max_len, 0))
    
    return x, y, n_chars + 1, max_len,char2ix,ix2char
    
def build_vocab(textA, textB):    
    vocab = {}
    text = textA + textB
    for seq in text:
        chars = set(list(seq))
        for char in chars:
            if char in vocab:
                vocab[char] += 1
            else:
                vocab[char] = 1
    vocab = sorted(vocab.items(), reverse=False)
        
    ix2char = defaultdict()
    char2ix = defaultdict()
    
    count = 1
    for c in vocab:
        char2ix[c[0]] = count
        ix2char[count] = c[0]
        count += 1

    return char2ix, ix2char
    
def get_idx_from_data(train,char2ix,ix2char):
    
    trainA, trainB = [], []
    
    for string in train[0]:
        seq = []
        chars = list(string)
        for c in chars:
            if c in char2ix:
                seq.append(char2ix[c])
        trainA.append(seq)
    
    for string in train[1]:
        seq = []
        chars = list(string)
        for c in chars:
            if c in char2ix:
                seq.append(char2ix[c])
        trainB.append(seq)
        
    train = [trainA,trainB]
    
    return train

def prepare_test_data(char2ix, ix2char, max_len):
    x_test, y_test = load_test_data(char2ix, ix2char, max_len)
    return x_test, y_test

def pad_or_truncate(l, target_len, pad):
    return l[:target_len] + [pad]*(target_len - len(l))
