'''
Created on July, 2018

@author: hugo

'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
import math

from .modules import EncoderCNN, EncoderRNN, Attention, GRUStep, create_mask
from .utils import to_var


class HRED(nn.Module):
    def __init__(self, vocab_size, word_emb_size, hidden_size, \
        init_w2v=None, enc_type='cnn', rnn_type='lstm', \
        bidirectional=True, num_rnn_layers=1, \
        utter_enc_dropout=None, knowledge_enc_dropout=None, \
        atten_type='add', score_type='clf', use_cuda=True):
        super(HRED, self).__init__()
        print('[ Using the HRED model ]')
        self.use_cuda = use_cuda
        self.score_type = score_type
        self.hidden_size = hidden_size
        self.utter_enc_dropout = utter_enc_dropout

        self.enc_type = enc_type
        if self.enc_type == 'rnn':
            self.utter_enc = EncoderRNN(vocab_size, word_emb_size, hidden_size, \
                    num_rnn_layers=num_rnn_layers, dropout=utter_enc_dropout, \
                    bidirectional=bidirectional, init_word_emb=init_w2v, \
                    rnn_type=rnn_type, use_cuda=self.use_cuda)
        elif self.enc_type == 'cnn':
            self.utter_enc = EncoderCNN(vocab_size, word_emb_size, hidden_size, \
                    kernel_size=[3], dropout=utter_enc_dropout, \
                    init_word_emb=init_w2v)
        else:
            raise RuntimeError('Unknown enc_type: {}'.format(self.enc_type))

        utter_emb_size = hidden_size
        mem_emb_size = hidden_size
        self.gru_step = GRUStep(hidden_size, 2 * hidden_size)
        

        if self.score_type == 'clf':
            self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False)
            self.score_func = self.clf_score
        elif self.score_type == 'ranking':
            self.score_func = self.ranking_score
        else:
            raise RuntimeError('Unknown score_type: {}'.format(self.score_type))
        print('[ score_type is {} ]'.format(score_type))

    def forward(self, ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num, temperature=0.1):
        # Data format
        # ctx: [[x_1, y_1, ..., x_{t-1}, y_{t-1}, ..., x_t], ...], B x T x L
        # ctx_utter_len: [[len_x_1, len_y_1, ..., len_x_t], ...], B x T 
        # ctx_turns_num: [T_1, ...], B x 1
        # cands: [[r_1, ..., r_m], ...], B x m x L
        # cands_utter_len: [[len_r_1, ..., len_r_m], ...], B x m
        B = ctx.size(0)
        if self.enc_type == 'rnn':
            ctx_vec = self.utter_enc(ctx.view(-1, ctx.size(-1)), ctx_utter_len.view(-1))[1].view(ctx.size(0), ctx.size(1), -1) # B x T x d
            cands_vec = self.utter_enc(cands.view(-1, cands.size(-1)), cands_utter_len.view(-1))[1].view(cands.size(0), cands.size(1), -1) # B x m x d
        else:
            ctx_vec = self.utter_enc(ctx.view(-1, ctx.size(-1))).view(ctx.size(0), ctx.size(1), -1) # B x T x d
            cands_vec = self.utter_enc(cands.view(-1, cands.size(-1))).view(cands.size(0), cands.size(1), -1) # B x m x d
        # if self.utter_enc_dropout:
        #     ctx_vec = F.dropout(ctx_vec, p=self.utter_enc_dropout, training=self.training)
        #     cands_vec = F.dropout(cands_vec, p=self.utter_enc_dropout, training=self.training)
        
        # ctx_turns_mask = create_mask(ctx_turns_num / 2 - 1, ctx.size(1) // 2 - 1, self.use_cuda)
        ctx_last_turns_ids = (ctx_turns_num / 2 - 1).view(ctx_turns_num.size(0), 1, 1).repeat(1, 1, ctx_vec.size(-1))

        # Init
        dialog_state = to_var(torch.zeros(B, self.hidden_size), use_cuda=self.use_cuda) # B x d
        all_o = []
        # all_sys_utter = []
        for i in range(0, ctx.size(1), 2):
            pre_sys_idx, user_idx = i, i + 1
            pre_sys_utter = ctx_vec[:, pre_sys_idx]
            user_utter = ctx_vec[:, user_idx]
            dialog_state = self.update_dialog_state(dialog_state, user_utter, pre_sys_utter)
            all_o.append(dialog_state.unsqueeze(1))
            # if user_idx + 1 < ctx.size(1):
            #     sys_utter = ctx_vec[:, user_idx + 1]
            #     all_sys_utter.append(sys_utter.unsqueeze(1))

        # all_sys_utter = torch.cat(all_sys_utter, 1)
        all_o = torch.cat(all_o, 1)
        assert all_o.size(1) == ctx.size(1) // 2
        last_o = all_o.gather(1, ctx_last_turns_ids).squeeze(1)
        score = self.score_func(last_o, cands_vec) # B x m

        # turn-wise score
        # turn_loss = -F.logsigmoid(torch.matmul(all_sys_utter.unsqueeze(2), self.linear_h(all_o[:, :-1, :]).unsqueeze(-1)).squeeze(-1).squeeze(-1)) # B x _
        # turn_loss = torch.sum(turn_loss * ctx_turns_mask) / B
        return score, None, None, None

    def update_dialog_state(self, dialog_state, x, y):
        dialog_state = self.gru_step(dialog_state, torch.cat([x, y], -1))
        return dialog_state

    def ranking_score(self, utter, cands):
        return torch.bmm(cands, utter.unsqueeze(-1)).squeeze(-1)

    def clf_score(self, utter, cands):
        return torch.bmm(cands, self.linear_h(utter).unsqueeze(-1)).squeeze(-1)