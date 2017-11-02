import os
import sys
import numpy as np
from collections import defaultdict
import _pickle as cPickle

def load_data(path = './data/dataset.txt'): 
    textA, textB = [], []
    with open(path, 'r') as f:
        for line in f:
            text = line.strip().split('\t')
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
    
    return x, y, n_chars, max_len
    
def build_vocab(textA, textB):
    
    vocab = defaultdict(float)
    text = textA + textB
    for seq in text:
        chars = set(list(seq))
        for char in chars:
            vocab[char] +=1
            
    ix2char = defaultdict()
    char2ix = defaultdict()
    
    count = 1
    for c in vocab.keys():
        char2ix[c] = count
        ix2char[count] = c
        count += 1

    return char2ix, ix2char
    
def get_idx_from_data(train,char2ix,ix2char):
    
    trainA, trainB = [], []
    
    for string in train[0]:
        seq = []
        chars = list(string)
        for c in chars:
            seq.append(char2ix[c])
        trainA.append(seq)
    
    for string in train[1]:
        seq = []
        chars = list(string)
        for c in chars:
            seq.append(char2ix[c])
        trainB.append(seq)
        
    train = [trainA,trainB]
    
    return train

def pad_or_truncate(l, target_len, pad):
    return l[:target_len] + [pad]*(target_len - len(l))
