'''
Created on July, 2018

@author: hugo

Copyright 2018 IBM Corp.

'''
import argparse
import os

from core.dataset.dataset import Dataset
from core.dataset.vocabulary import Vocabulary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_path', '--train_path', required=True, type=str, help='path to the data dir')
    parser.add_argument('-dev_path', '--dev_path', required=True, type=str, help='path to the data dir')
    parser.add_argument('-test_path', '--test_path', type=str, help='path to the data dir')
    parser.add_argument('-db_path', '--db_path', required=True, type=str, help='path to the data dir')
    parser.add_argument('-out_dir', '--out_dir', required=True, type=str, help='path to the output dir')
    args = parser.parse_args()
    opt = vars(parser.parse_args())


    # Prepare dataset
    train_data = Dataset.from_file(opt['train_path'], opt['db_path'], max_vocab=100000)
    train_data.save(os.path.join(opt['out_dir'], 'train_data.pkl'), vocab_file_name=os.path.join(opt['out_dir'], 'vocab.txt'))
    del train_data.data

    dev_data = Dataset.from_file(opt['dev_path'], opt['db_path'], vocab=train_data.vocab)
    dev_data.save(os.path.join(opt['out_dir'], 'dev_data.pkl'))
    del dev_data.data
    
    if opt['test_path']:
        test_data = Dataset.from_file(opt['test_path'], opt['db_path'], vocab=train_data.vocab)
        test_data.save(os.path.join(opt['out_dir'], 'test_data.pkl'))

    print('Saved data to {}'.format(opt['out_dir']))
    import pdb;pdb.set_trace()
