import _pickle as pickle
from collections import defaultdict

def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_data_mapping(n_chars, max_len, char2ix, ix2char, path):
    data_prepare = defaultdict()
    data_prepare["n_chars"] = n_chars
    data_prepare["max_len"] = max_len
    data_prepare["char2ix"] = char2ix
    data_prepare["ix2char"] = ix2char
    save_obj(data_prepare, path)