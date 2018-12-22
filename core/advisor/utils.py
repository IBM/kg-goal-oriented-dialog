'''
Created on July, 2018

@author: hugo

'''
import torch
from torch.autograd import Variable
import numpy as np



def to_var(x, use_cuda=True, inference_mode=False, requires_grad=False):
    if torch.cuda.is_available() and use_cuda:
        x = x.cuda()
    return Variable(x, volatile=inference_mode, requires_grad=requires_grad)

# # One pass over the dataset
# def next_batch(memories, mem_dep_mat, queries, query_words, query_markups, query_lengths, gold_ans_inds, batch_size):
#     for i in range(0, len(memories), batch_size):
#         yield ((memories[i: i + batch_size], mem_dep_mat[i: i + batch_size]), queries[i: i + batch_size], query_words[i: i + batch_size], query_markups[i: i + batch_size], query_lengths[i: i + batch_size]), gold_ans_inds[i: i + batch_size]

# def to_onehot(x, depth, zero_padding=False):
#     code = torch.sparse.torch.eye(depth)
#     if zero_padding:
#         code = torch.cat([torch.zeros(1, depth), code], 0)
#     vec = code.index_select(0, x.view(int(np.prod(x.size())))).view(list(x.size()) + [depth])
#     return vec