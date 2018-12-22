'''
Created on July, 2018

@author: hugo

'''
import argparse
import os

from core.dataset.utils import dump_embeddings
from core.dataset.vocabulary import Vocabulary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', '--embed_path', required=True, type=str, help='path to the pretrained word embeddings')
    parser.add_argument('-vocab_path', '--vocab_path', required=True, type=str, help='path to the data dir')
    parser.add_argument('-out', '--out_path', required=True, type=str, help='path to the output path')
    parser.add_argument('-emb_size', '--emb_size', required=True, type=int, help='embedding size')
    parser.add_argument('--binary', action='store_true', help='flag: binary file')
    args = parser.parse_args()
    opt = vars(parser.parse_args())


    vocab = Vocabulary.load(opt['vocab_path'])
    dump_embeddings(vocab, opt['embed_path'], opt['out_path'], emb_size=opt['emb_size'], binary=True if opt['binary'] else False)