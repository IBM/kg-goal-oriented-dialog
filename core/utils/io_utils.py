'''
Created on July, 2018

@author: hugo

'''

import numpy as np


def load_ndarray(path_to_file):
    try:
        with open(path_to_file, 'rb') as f:
            data = np.load(f)
    except Exception as e:
        raise e
    return data