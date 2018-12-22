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

from .modules import EncoderCNN, EncoderRNN, Attention, GRUStep, AttentionRowsRows, EncoderKnowledge, create_mask
from .utils import to_var

INF = 1e20
NUM_MODES = 4

class HRED_DB(nn.Module):
    def __init__(self, vocab_size, word_emb_size, hidden_size, \
        init_w2v=None, enc_type='cnn', rnn_type='lstm', \
        bidirectional=True, num_rnn_layers=1, \
        utter_enc_dropout=None, knowledge_enc_dropout=None, \
        atten_type='add', score_type='clf', use_cuda=True):  
        ''' This is a simplified version of the Advisor model which does not contain the decision-making module.
        '''
        super(HRED_DB, self).__init__()
        print('[ Using the HRED_DB model ]')
        self.use_cuda = use_cuda
        self.score_type = score_type
        self.hidden_size = hidden_size
        self.utter_enc_dropout = utter_enc_dropout
        self.knowledge_enc_dropout = knowledge_enc_dropout

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

        self.knowledge_enc = EncoderKnowledge(hidden_size, \
                                            vocab_size=vocab_size, \
                                            word_emb_size=word_emb_size, \
                                            shared_word_emb=self.utter_enc.word_emb, \
                                            dropout=knowledge_enc_dropout)

        utter_emb_size = hidden_size
        mem_emb_size = hidden_size
        self.arr_step = AttentionRowsRows(hidden_size, utter_emb_size, mem_emb_size, atten_type=atten_type)
        self.acc_step = AttentionColsCols(hidden_size, mem_emb_size, atten_type=atten_type, use_cuda=self.use_cuda)
        self.gru_step = GRUStep(hidden_size, 2 * hidden_size)

        self.P_h = torch.Tensor(hidden_size, hidden_size)
        self.P_h = nn.Parameter(nn.init.xavier_uniform(self.P_h))
        self.P_m = torch.Tensor(mem_emb_size, hidden_size)
        self.P_m = nn.Parameter(nn.init.xavier_uniform(self.P_m))

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
        # db_fields: [[f_1, ..., f_k], ...], B x K x Lf
        # db_cells: [[[c_{11}, ..., c_{1K}], ..., [c_{N1}, ..., c_{NK}]], ...], B x N x K x Lc
        # db_rows_num: [N_1, ...], B x 1
        B, N, K, _ = db_cells.size()
        if self.enc_type == 'rnn':
            ctx_vec = self.utter_enc(ctx.view(-1, ctx.size(-1)), ctx_utter_len.view(-1))[1].view(ctx.size(0), ctx.size(1), -1) # B x T x d
            cands_vec = self.utter_enc(cands.view(-1, cands.size(-1)), cands_utter_len.view(-1))[1].view(cands.size(0), cands.size(1), -1) # B x m x d
        else:
            ctx_vec = self.utter_enc(ctx.view(-1, ctx.size(-1))).view(ctx.size(0), ctx.size(1), -1) # B x T x d
            cands_vec = self.utter_enc(cands.view(-1, cands.size(-1))).view(cands.size(0), cands.size(1), -1) # B x m x d
        # if self.utter_enc_dropout:
            # ctx_vec = F.dropout(ctx_vec, p=self.utter_enc_dropout, training=self.training)
            # cands_vec = F.dropout(cands_vec, p=self.utter_enc_dropout, training=self.training)

        db_fields_vec, db_cells_vec = self.knowledge_enc(db_fields, db_cells)
        if self.knowledge_enc_dropout:
            db_fields_vec = F.dropout(db_fields_vec, p=self.knowledge_enc_dropout, training=self.training)
            db_cells_vec = F.dropout(db_cells_vec, p=self.knowledge_enc_dropout, training=self.training)


        db_row_mask = create_mask(db_rows_num, db_cells.size(1), self.use_cuda) # B x N
        ctx_last_turns_ids = (ctx_turns_num / 2 - 1).view(ctx_turns_num.size(0), 1, 1).repeat(1, 1, ctx_vec.size(-1))

        # Init
        dialog_state = to_var(torch.zeros(B, self.hidden_size), use_cuda=self.use_cuda) # B x d
        carr = to_var(torch.ones(B, N) / N, use_cuda=self.use_cuda) # distribution, B x N

        all_o = []
        all_acc = []
        for i in range(0, ctx.size(1), 2):
            pre_sys_idx, user_idx = i, i + 1
            pre_sys_utter = ctx_vec[:, pre_sys_idx]
            user_utter = ctx_vec[:, user_idx]
            o, dialog_state, carr, acc = self.hop(dialog_state, carr, user_utter, pre_sys_utter, db_fields_vec, db_cells_vec, db_row_mask, temperature)
            all_o.append(o.unsqueeze(1))
            all_acc.append(acc.unsqueeze(1))

        all_o = torch.cat(all_o, 1)
        all_acc = torch.cat(all_acc, 1)
        assert all_o.size(1) == ctx.size(1) // 2
        last_o = all_o.gather(1, ctx_last_turns_ids).squeeze(1)
        score = self.score_func(last_o, cands_vec) # B x m
        return score, None, all_acc, None

    def hop(self, dialog_state, carr, user_utter, pre_sys_utter, db_fields_vec, db_cells_vec, db_row_mask, temperature=0.1):
        # Update dialog state
        dialog_state = self.update_dialog_state(dialog_state, user_utter, pre_sys_utter)

        # Update CARR
        carr, acr = self.arr_step(user_utter, db_fields_vec, db_cells_vec, carr, db_row_mask)

        # Update ACC
        acc = self.acc_step(user_utter, db_fields_vec, acr) # field idx 4: course

        # Compute output state
        o = self.compute_output_state(dialog_state, db_cells_vec, carr, acc)
        return o, dialog_state, carr, acc

    def update_dialog_state(self, dialog_state, x, y):
        dialog_state = self.gru_step(dialog_state, torch.cat([x, y], -1))
        return dialog_state

    def compute_output_state(self, dialog_state, db_cells_vec, carr, acc):
        B, N, K, _ = db_cells_vec.size()
        m = torch.bmm(carr.unsqueeze(1), db_cells_vec.view(B, N, -1)).view(B, K, -1)
        m = torch.bmm(acc.unsqueeze(1), m).squeeze(1)
        o = torch.mm(dialog_state, self.P_h) + torch.mm(m, self.P_m)
        return o

    def ranking_score(self, utter, cands):
        return torch.bmm(cands, utter.unsqueeze(-1)).squeeze(-1)

    def clf_score(self, utter, cands):
        return torch.bmm(cands, self.linear_h(utter).unsqueeze(-1)).squeeze(-1)

class AttentionColsCols(nn.Module):
    def __init__(self, hidden_size, memory_embed_size, atten_type='add', use_cuda=True):
        super(AttentionColsCols, self).__init__()
        self.use_cuda = use_cuda
        self.atten_acc = Attention(hidden_size, hidden_size, memory_embed_size, atten_type=atten_type)

    def forward(self, x, db_fields_vec, acr):
        # cacc_q [B x K]
        B, K, d = db_fields_vec.size()
        fields_vec_i = (1 - acr).unsqueeze(-1).repeat(1, 1, d) * db_fields_vec
        acc = F.softmax(self.atten_acc(x, fields_vec_i), -1)
        return acc