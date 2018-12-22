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

from .utils import to_var

INF = 1e20
NUM_MODES = 4
# modes: recommend, query, inform, social

class Advisor(nn.Module):
    def __init__(self, vocab_size, word_emb_size, hidden_size, \
        init_w2v=None, enc_type='cnn', rnn_type='lstm', \
        bidirectional=True, num_rnn_layers=1, \
        utter_enc_dropout=None, knowledge_enc_dropout=None, \
        atten_type='add', score_type='clf', use_cuda=True):  
        super(Advisor, self).__init__()
        print('[ Using the Advisor model ]')
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
        self.decision_step = Decision(hidden_size, utter_emb_size, use_cuda=self.use_cuda)
        self.acc_step = AttentionColsCols(hidden_size, mem_emb_size, atten_type=atten_type, use_cuda=self.use_cuda)
        self.gru_step = GRUStep(hidden_size, 2 * hidden_size)

        self.P_h = torch.Tensor(hidden_size, hidden_size)
        self.P_h = nn.Parameter(nn.init.xavier_uniform(self.P_h))
        self.P_m = torch.Tensor(mem_emb_size, hidden_size)
        self.P_m = nn.Parameter(nn.init.xavier_uniform(self.P_m))
        # self.P_f = torch.Tensor(mem_emb_size, hidden_size)
        # self.P_f = nn.Parameter(nn.init.xavier_uniform(self.P_f))

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

        # Encode context
        if self.enc_type == 'rnn':
            ctx_vec = self.utter_enc(ctx.view(-1, ctx.size(-1)), ctx_utter_len.view(-1))[1].view(ctx.size(0), ctx.size(1), -1) # B x T x d
            cands_vec = self.utter_enc(cands.view(-1, cands.size(-1)), cands_utter_len.view(-1))[1].view(cands.size(0), cands.size(1), -1) # B x m x d
        else:
            ctx_vec = self.utter_enc(ctx.view(-1, ctx.size(-1))).view(ctx.size(0), ctx.size(1), -1) # B x T x d
            cands_vec = self.utter_enc(cands.view(-1, cands.size(-1))).view(cands.size(0), cands.size(1), -1) # B x m x d
        # if self.utter_enc_dropout:
            # ctx_vec = F.dropout(ctx_vec, p=self.utter_enc_dropout, training=self.training)
            # cands_vec = F.dropout(cands_vec, p=self.utter_enc_dropout, training=self.training)

        # Encode external knowledge
        db_fields_vec, db_cells_vec = self.knowledge_enc(db_fields, db_cells)
        if self.knowledge_enc_dropout:
            db_fields_vec = F.dropout(db_fields_vec, p=self.knowledge_enc_dropout, training=self.training)
            db_cells_vec = F.dropout(db_cells_vec, p=self.knowledge_enc_dropout, training=self.training)


        db_row_mask = create_mask(db_rows_num, db_cells.size(1), self.use_cuda) # B x N
        # ctx_turns_mask = create_mask(ctx_turns_num / 2 - 1, ctx.size(1) // 2 - 1, self.use_cuda)
        ctx_last_turns_ids = (ctx_turns_num / 2 - 1).view(ctx_turns_num.size(0), 1, 1).repeat(1, 1, ctx_vec.size(-1))

        # Init
        dialog_state = to_var(torch.zeros(B, self.hidden_size), use_cuda=self.use_cuda) # B x d
        carr = to_var(torch.ones(B, N) / N, use_cuda=self.use_cuda) # distribution, B x N
        cacc_q = to_var(torch.zeros(B, K), use_cuda=self.use_cuda) # [0, 1]^K, B x K

        all_o = []
        all_theta = []
        all_acc = []
        # all_sys_utter = []
        for i in range(0, ctx.size(1), 2):
            # Update internal states at each stage
            pre_sys_idx, user_idx = i, i + 1
            pre_sys_utter = ctx_vec[:, pre_sys_idx]
            user_utter = ctx_vec[:, user_idx]
            o, dialog_state, carr, cacc_q, theta, acc = self.hop(dialog_state, carr, cacc_q, user_utter, pre_sys_utter, db_fields_vec, db_cells_vec, db_row_mask, temperature)
            all_o.append(o.unsqueeze(1))
            all_theta.append(theta.unsqueeze(1))
            all_acc.append(acc.unsqueeze(1))
            # if user_idx + 1 < ctx.size(1):
            #     sys_utter = ctx_vec[:, user_idx + 1]
            #     all_sys_utter.append(sys_utter.unsqueeze(1))

        all_o = torch.cat(all_o, 1)
        all_theta = torch.cat(all_theta, 1)
        all_acc = torch.cat(all_acc, 1)
        # all_sys_utter = torch.cat(all_sys_utter, 1)
        assert all_o.size(1) == ctx.size(1) // 2
        last_o = all_o.gather(1, ctx_last_turns_ids).squeeze(1) # Get last output state for each example
        score = self.score_func(last_o, cands_vec) # B x m

        # # turn-wise score
        # turn_loss = -F.logsigmoid(torch.matmul(all_sys_utter.unsqueeze(2), self.linear_h(all_o[:, :-1, :]).unsqueeze(-1)).squeeze(-1).squeeze(-1)) # B x _
        # turn_loss = torch.sum(turn_loss * ctx_turns_mask) / B
        return score, all_theta, all_acc, None


    # def forward(self, ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num, temperature=0.1):
    #     # Randomly sample negative examples for intermediate turns
    #     # Data format
    #     # ctx: [[x_1, y_1, ..., x_{t-1}, y_{t-1}, ..., x_t], ...], B x T x L
    #     # ctx_utter_len: [[len_x_1, len_y_1, ..., len_x_t], ...], B x T 
    #     # ctx_turns_num: [T_1, ...], B x 1
    #     # cands: [[r_1, ..., r_m], ...], B x m x L
    #     # cands_utter_len: [[len_r_1, ..., len_r_m], ...], B x m
    #     # db_fields: [[f_1, ..., f_k], ...], B x K x Lf
    #     # db_cells: [[[c_{11}, ..., c_{1K}], ..., [c_{N1}, ..., c_{NK}]], ...], B x N x K x Lc
    #     # db_rows_num: [N_1, ...], B x 1
    #     B, N, K, _ = db_cells.size()
    #     if self.enc_type == 'rnn':
    #         ctx_vec = self.utter_enc(ctx.view(-1, ctx.size(-1)), ctx_utter_len.view(-1))[1].view(ctx.size(0), ctx.size(1), -1) # B x T x d
    #         cands_vec = self.utter_enc(cands.view(-1, cands.size(-1)), cands_utter_len.view(-1))[1].view(cands.size(0), cands.size(1), -1) # B x m x d
    #     else:
    #         ctx_vec = self.utter_enc(ctx.view(-1, ctx.size(-1))).view(ctx.size(0), ctx.size(1), -1) # B x T x d
    #         cands_vec = self.utter_enc(cands.view(-1, cands.size(-1))).view(cands.size(0), cands.size(1), -1) # B x m x d
    #     # if self.utter_enc_dropout:
    #         # ctx_vec = F.dropout(ctx_vec, p=self.utter_enc_dropout, training=self.training)
    #         # cands_vec = F.dropout(cands_vec, p=self.utter_enc_dropout, training=self.training)

    #     db_fields_vec, db_cells_vec = self.knowledge_enc(db_fields, db_cells)
    #     if self.knowledge_enc_dropout:
    #         db_fields_vec = F.dropout(db_fields_vec, p=self.knowledge_enc_dropout, training=self.training)
    #         db_cells_vec = F.dropout(db_cells_vec, p=self.knowledge_enc_dropout, training=self.training)


    #     db_row_mask = create_mask(db_rows_num, db_cells.size(1), self.use_cuda) # B x N
    #     ctx_turns_mask = create_mask(ctx_turns_num / 2 - 1, ctx.size(1) // 2 - 1, self.use_cuda)
    #     ctx_last_turns_ids = (ctx_turns_num / 2 - 1).view(ctx_turns_num.size(0), 1, 1).repeat(1, 1, ctx_vec.size(-1))

    #     # Init
    #     dialog_state = to_var(torch.zeros(B, self.hidden_size), use_cuda=self.use_cuda) # B x d
    #     carr = to_var(torch.ones(B, N) / N, use_cuda=self.use_cuda) # distribution, B x N
    #     cacc_q = to_var(torch.zeros(B, K), use_cuda=self.use_cuda) # [0, 1]^K, B x K

    #     all_o = []
    #     all_theta = []
    #     all_acc = []
    #     all_sys_utter = []
    #     for i in range(0, ctx.size(1), 2):
    #         pre_sys_idx, user_idx = i, i + 1
    #         pre_sys_utter = ctx_vec[:, pre_sys_idx]
    #         user_utter = ctx_vec[:, user_idx]
    #         o, dialog_state, carr, cacc_q, theta, acc = self.hop(dialog_state, carr, cacc_q, user_utter, pre_sys_utter, db_fields_vec, db_cells_vec, db_row_mask, temperature)
    #         all_o.append(o.unsqueeze(1))
    #         all_theta.append(theta.unsqueeze(1))
    #         all_acc.append(acc.unsqueeze(1))
    #         if user_idx + 1 < ctx.size(1):
    #             sys_utter = ctx_vec[:, user_idx + 1]
    #             all_sys_utter.append(sys_utter.unsqueeze(1))

    #     all_o = torch.cat(all_o, 1)
    #     all_theta = torch.cat(all_theta, 1)
    #     all_acc = torch.cat(all_acc, 1)
    #     all_sys_utter = torch.cat(all_sys_utter, 1)
    #     assert all_o.size(1) == ctx.size(1) // 2
    #     last_o = all_o.gather(1, ctx_last_turns_ids).squeeze(1)
    #     score = self.score_func(last_o, cands_vec) # B x m

    #     n = 5
    #     # turn-wise score
    #     # neg_cands = []
    #     # for _ in range(all_sys_utter.size(1)):
    #     #     ids = to_var(torch.LongTensor([np.random.choice(cands_vec.size(1), n - 1, replace=False).tolist() for i in range(cands_vec.size(0))]), use_cuda=self.use_cuda)
    #     #     neg_cands.append(cands_vec.gather(1, ids.unsqueeze(-1).repeat(1, 1, cands_vec.size(-1))).unsqueeze(1))
    #     # neg_cands = torch.cat(neg_cands, 1)

    #     neg_cands = cands_vec[:, :n - 1, :].unsqueeze(1).repeat(1, all_sys_utter.size(1), 1, 1)
    #     all_sys_utter = torch.cat([all_sys_utter.unsqueeze(2), neg_cands], 2) # B x _ x n x d
    #     turn_score = self.score_func(all_o[:, :-1, :], all_sys_utter)
    #     turn_loss = -F.log_softmax(turn_score, dim=-1)
    #     turn_loss = torch.sum(turn_loss[:, :, 0] * ctx_turns_mask) / B
    #     return score, all_theta, all_acc, turn_loss


    def hop(self, dialog_state, carr, cacc_q, user_utter, pre_sys_utter, db_fields_vec, db_cells_vec, db_row_mask, temperature=0.1):
        # Update dialog state
        dialog_state = self.update_dialog_state(dialog_state, user_utter, pre_sys_utter)

        # Update CARR
        carr, acr = self.arr_step(user_utter, db_fields_vec, db_cells_vec, carr, db_row_mask)

        # Update decision prob.
        theta = self.decision_step(dialog_state, user_utter, carr, temperature)

        # Update ACC
        acc, cacc_q = self.acc_step(user_utter, dialog_state, db_fields_vec, db_cells_vec, cacc_q, acr, carr, theta, r_idx=[4]) # field idx 4: course

        # Compute output state
        o = self.compute_output_state(dialog_state, db_fields_vec, db_cells_vec, carr, acc)
        return o, dialog_state, carr, cacc_q, theta, acc

    def update_dialog_state(self, dialog_state, x, y):
        dialog_state = self.gru_step(dialog_state, torch.cat([x, y], -1))
        return dialog_state

    def compute_output_state(self, dialog_state, db_fields_vec, db_cells_vec, carr, acc):
        B, N, K, _ = db_cells_vec.size()
        m = torch.bmm(carr.unsqueeze(1), db_cells_vec.view(B, N, -1)).view(B, K, -1)
        m = torch.bmm(acc.unsqueeze(1), m).squeeze(1)
        # f = torch.bmm(acc.unsqueeze(1), db_fields_vec).squeeze(1)

        o = torch.mm(dialog_state, self.P_h) + torch.mm(m, self.P_m)# + torch.mm(f, self.P_f)
        return o

    def ranking_score(self, utter, cands):
        return torch.matmul(cands, utter.unsqueeze(-1)).squeeze(-1)

    def clf_score(self, utter, cands):
        return torch.matmul(cands, self.linear_h(utter).unsqueeze(-1)).squeeze(-1)

class Decision(nn.Module):
    def __init__(self, hidden_size, input_size, use_cuda=True):
        super(Decision, self).__init__()
        '''Decision module:
        Compute a prob. distribution over modes {recommend, query, inform, social}'''
        self.use_cuda = use_cuda
        self.Wx = torch.Tensor(input_size, NUM_MODES)
        self.Wx = nn.Parameter(nn.init.xavier_uniform(self.Wx))
        self.Wh = torch.Tensor(hidden_size, NUM_MODES)
        self.Wh = nn.Parameter(nn.init.xavier_uniform(self.Wh))
        self.Ws = torch.Tensor(1, NUM_MODES)
        self.Ws = nn.Parameter(nn.init.xavier_uniform(self.Ws))

    def forward(self, h_state, x, carr, temperature=0.1):
        sharpness = self.sharpness_squared_sum(carr)
        logits = torch.mm(h_state, self.Wh) + torch.mm(x, self.Wx) + torch.mm(sharpness, self.Ws)
        theta = F.softmax(logits, dim=-1)
        return theta

    # def forward(self, h_state, x, carr, temperature=0.1):
    #     sharpness = self.sharpness_squared_sum(carr)
    #     logits = torch.mm(h_state, self.Wh) + torch.mm(x, self.Wx) + torch.mm(sharpness, self.Ws)
    #     # theta = F.softmax(logits, dim=-1)
    #     if not self.training:
    #         theta = hard_sampling(logits, use_cuda=self.use_cuda)
    #     else:
    #         theta = gumbel_softmax_sample(logits, temperature=temperature, use_cuda=self.use_cuda)
    #     return theta

    def sharpness_squared_sum(self, p):
        # Measure the sharpness of a prob. distribution
        return torch.pow(p, 2).sum(-1, keepdim=True)

    # def sharpness_cross_entropy(self, p):
    #     return (p * torch.log(p)).sum(-1, keepdim=True)
        

class AttentionRowsRows(nn.Module):
    def __init__(self, hidden_size, input_size, memory_embed_size, atten_type='add'):
        super(AttentionRowsRows, self).__init__()
        ''' ARR module:
        Compute the cumulative ARR (attention over rows for rows),
        which is a prob. distribution over rows.
        '''
        self.atten_arr = Attention(hidden_size, input_size, memory_embed_size, atten_type=atten_type)
        self.atten_acr = Attention(hidden_size, input_size, memory_embed_size, atten_type=atten_type)
    
    def forward(self, x, db_fields_vec, db_cells_vec, carr, db_row_mask=None):
        # db_fields_vec [B x K x d]
        # db_cells_vec [B x N x K x d]
        # carr [B x N]
        N = db_cells_vec.size(1)
        d = db_cells_vec.size(-1)
        acr = self.atten_acr(x, db_fields_vec) # B x K, attention over columns for rows
        acr = F.softmax(acr, dim=-1)
        rows_cells_vec = torch.matmul(acr.unsqueeze(1).unsqueeze(1).repeat(1, N, 1, 1), db_cells_vec).squeeze(2) # B x N x d
        arr = self.atten_arr(x, carr.unsqueeze(-1).repeat(1, 1, d) * rows_cells_vec, atten_mask=db_row_mask) # B x N
        arr = F.softmax(arr, dim=-1)
        carr = F.normalize(carr + arr, p=1, dim=-1)
        return carr, acr

class AttentionColsCols(nn.Module):
    def __init__(self, hidden_size, memory_embed_size, atten_type='add', use_cuda=True):
        super(AttentionColsCols, self).__init__()
        '''ACC module: 
        Compute ACC (attention over cols for cols),
        which is a prob. distribution over cols.
        Values stored in those high attention cols will be retrieved.
        ACCs are computed differently in different modes.
        The final ACC is the weighted sum of all mode-specific ACCs.
        '''
        self.use_cuda = use_cuda
        self.atten_acc_q = Attention(hidden_size, hidden_size, memory_embed_size, atten_type=atten_type)
        self.atten_acc_i = Attention(hidden_size, hidden_size, memory_embed_size, atten_type=atten_type)
        # self.linear_acc_q = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, h_state, db_fields_vec, db_cells_vec, cacc_q, acr, carr, theta, r_idx=[0]):
        # cacc_q [B x K]
        B, K, d = db_fields_vec.size()
        # 1) acc in 'recommend' mode
        acc_r = to_var(torch.zeros(1, K), use_cuda=self.use_cuda)
        for i in r_idx:
            acc_r[0, i] = 1 / len(r_idx)
        acc_r = acc_r.repeat(B, 1)


        # 2) acc in 'query' mode
        fields_vec_q = (1 - cacc_q).unsqueeze(-1).repeat(1, 1, d) * db_fields_vec
        acc_q = F.softmax(self.atten_acc_q(h_state, fields_vec_q), -1)

        # db_cell_q = carr.unsqueeze(-1).unsqueeze(-1) * db_cells_vec
        # acc_q = torch.bmm(carr.unsqueeze(1), self.linear_acc_q(torch.pow(db_cell_q - torch.mean(db_cell_q, 1, keepdim=True), 2)).squeeze(-1)).squeeze(1)
        # acc_q = F.softmax((1 - cacc_q) * acc_q, dim=-1)

        # B, N, K, d = db_cells_vec.size()
        # mean_vec = torch.bmm(carr.unsqueeze(1), db_cells_vec.view(B, N, -1)).view(B, 1, K, -1)
        # acc_q = torch.bmm(carr.unsqueeze(1), self.linear_acc_q(torch.pow(db_cells_vec - mean_vec, 2)).squeeze(-1)).squeeze(1)
        # acc_q = F.softmax((1 - cacc_q) * acc_q, dim=-1)

        cacc_q = torch.clamp(cacc_q + theta[:, 1].unsqueeze(-1) * acc_q, 0, 1)


        # 3) acc in 'inform' mode
        fields_vec_i = (1 - acr).unsqueeze(-1).repeat(1, 1, d) * db_fields_vec
        acc_i = F.softmax(self.atten_acc_i(x, fields_vec_i), -1)


        # 4) acc in 'social' mode
        acc_s = to_var(torch.zeros(B, K), use_cuda=self.use_cuda)

        acc = torch.cat([acc_r.unsqueeze(-1), acc_q.unsqueeze(-1), acc_i.unsqueeze(-1), acc_s.unsqueeze(-1)], -1)
        acc = torch.bmm(acc, theta.unsqueeze(-1)).squeeze(-1)
        return acc, cacc_q

class EncoderKnowledge(nn.Module):
    def __init__(self, hidden_size, vocab_size=None, word_emb_size=None, shared_word_emb=None, dropout=None):
        super(EncoderKnowledge, self).__init__()
        '''Encode the external knowledge.
        db_fields_vec: vector representations of all the col names.
        db_cells_vec: vector representations of all the cells.
        Use average word embeddings.
        '''
        self.dropout = dropout
        self.word_emb = shared_word_emb if shared_word_emb is not None else nn.Embedding(vocab_size, word_emb_size, padding_idx=0)
        self.word_emb_size = self.word_emb.weight.data.size(1)
        self.linear_fields = nn.Linear(self.word_emb_size, hidden_size, bias=False)
        self.linear_cells = nn.Linear(self.word_emb_size, hidden_size, bias=False)

    def forward(self, fields, cells):
        # fields [B x K x Lf]
        # cells [B x N x K x Lc]
        db_fields_vec, db_cells_vec = self.encode(fields, cells)
        db_cells_vec = self.linear_cells(db_cells_vec)
        db_fields_vec = self.linear_fields(db_fields_vec)
        return db_fields_vec, db_cells_vec

    # def forward(self, fields, cells):
    #     # fields [B x K x Lf]
    #     # cells [B x N x K x Lc]
    #     db_fields_vec, db_cells_vec = self.encode(fields, cells)
    #     # db_cells_vec = F.tanh(self.linear_cells(torch.cat([db_fields_vec.unsqueeze(1).repeat(1, cells.size(1), 1, 1), db_cells_vec], -1)))
    #     # db_fields_vec = self.linear_fields(db_fields_vec)
    #     return db_fields_vec, db_cells_vec


    def encode(self, fields, cells):
        B, K, Lf = fields.size()
        _, N, _, Lc = cells.size()

        db_fields_vec = torch.mean(self.word_emb(fields.view(-1, Lf)), 1).view(B, K, -1) # B x K x d
        db_cells_vec = torch.mean(self.word_emb(cells.view(-1, Lc)), 1).view(B, N, K, -1) # B x N x K x d
        if self.dropout:
            db_fields_vec = F.dropout(db_fields_vec, p=self.dropout, training=self.training)
            db_cells_vec = F.dropout(db_cells_vec, p=self.dropout, training=self.training)
        return db_fields_vec, db_cells_vec



class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, word_emb_size, hidden_size, num_rnn_layers=1, dropout=None, \
        bidirectional=False, shared_word_emb=None, init_word_emb=None, rnn_type='lstm', use_cuda=True):
        super(EncoderRNN, self).__init__()
        ''''RNN encoder: LSTM or GRU.
        '''
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))
        if bidirectional:
            print('[ Using bidirectional {} encoder ]'.format(rnn_type))
        else:
            print('[ Using {} encoder ]'.format(rnn_type))
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.num_directions = 2 if bidirectional else 1
        self.word_emb = shared_word_emb if shared_word_emb is not None else nn.Embedding(vocab_size, word_emb_size, padding_idx=0)
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(word_emb_size, self.hidden_size, num_rnn_layers, batch_first=True, \
            dropout=dropout if dropout else 0, bidirectional=bidirectional)
        if shared_word_emb is None:
            self.init_weights(init_word_emb)

    def init_weights(self, init_word_emb):
        if init_word_emb is not None:
            print('[ Using pretrained word embeddings ]')
            self.word_emb.weight.data.copy_(torch.from_numpy(init_word_emb))
        else:
            self.word_emb.weight.data.uniform_(-0.08, 0.08)

    def forward(self, x, x_len):
        """x: [batch_size * max_length]
           x_len: [batch_size]
        """
        x = self.word_emb(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)

        h0 = to_var(torch.zeros(self.num_rnn_layers * self.num_directions, x_len.size(0), self.hidden_size), use_cuda=self.use_cuda)
        if self.rnn_type == 'lstm':
            c0 = to_var(torch.zeros(self.num_rnn_layers * self.num_directions, x_len.size(0), self.hidden_size), use_cuda=self.use_cuda)
            packed_h, (packed_h_t, _) = self.model(x, (h0, c0))
            if self.num_directions == 2:
                packed_h_t = torch.cat([packed_h_t[i] for i in range(packed_h_t.size(0))], -1)
        else:
            packed_h, packed_h_t = self.model(x, h0)
            if self.num_directions == 2:
                packed_h_t = packed_h_t.transpose(0, 1).contiguous().view(x_len.size(0), -1)

        hh, _ = pad_packed_sequence(packed_h, batch_first=True)

        # restore the sorting
        _, inverse_indx = torch.sort(indx, 0)
        restore_hh = hh[inverse_indx]
        restore_packed_h_t = packed_h_t[inverse_indx]
        return restore_hh, restore_packed_h_t


class EncoderCNN(nn.Module):
    def __init__(self, vocab_size, word_emb_size, hidden_size, kernel_size=[2, 3], \
            dropout=None, shared_word_emb=None, init_word_emb=None):
        super(EncoderCNN, self).__init__()
        '''CNN encoder
        '''
        print('[ Using CNN encoder ]')
        self.dropout = dropout
        self.word_emb = shared_word_emb if shared_word_emb is not None else nn.Embedding(vocab_size, word_emb_size, padding_idx=0)

        self.cnns = nn.ModuleList([nn.Sequential(
            nn.Conv1d(word_emb_size, hidden_size, kernel_size=k, padding=k-1)
            ) for k in kernel_size])

        if len(kernel_size) > 1:
            self.fc = nn.Linear(len(kernel_size) * hidden_size, hidden_size)
        if shared_word_emb is None:
            self.init_weights(init_word_emb)

    def init_weights(self, init_word_emb):
        if init_word_emb is not None:
            print('[ Using pretrained word embeddings ]')
            self.word_emb.weight.data.copy_(torch.from_numpy(init_word_emb))
        else:
            self.word_emb.weight.data.uniform_(-0.08, 0.08)

    def forward(self, x):
        """x: [batch_size * max_length]
           x_len: reserved
           x_features: reserved
        """
        x = self.word_emb(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Trun(batch_size, seq_len, embed_size) to (batch_size, embed_size, seq_len) for cnn1d
        x = x.transpose(1, 2)
        z = [conv(x) for conv in self.cnns]
        output = [F.max_pool1d(i, kernel_size=i.size(-1)).squeeze(-1) for i in z]

        if len(output) > 1:
            output = self.fc(torch.cat(output, -1))
        else:
            output = output[0]
        return output


class GRUStep(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        '''GRU module'''
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, h_state, input):        
        z = F.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        r = F.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        t = F.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state

class Attention(nn.Module):
    def __init__(self, hidden_size, h_state_embed_size, in_memory_embed_size, atten_type='simple'):
        super(Attention, self).__init__()
        '''Attention module: 
        Three variants: additive, multiplicative or simple.
        '''
        self.atten_type = atten_type
        if atten_type in ('mul', 'add'):
            self.W = torch.Tensor(h_state_embed_size, hidden_size)
            self.W = nn.Parameter(nn.init.xavier_uniform(self.W))
            if atten_type == 'add':
                self.W2 = torch.Tensor(in_memory_embed_size, hidden_size)
                self.W2 = nn.Parameter(nn.init.xavier_uniform(self.W2))
                self.W3 = torch.Tensor(hidden_size, 1)
                self.W3 = nn.Parameter(nn.init.xavier_uniform(self.W3))
        elif atten_type == 'simple':
            pass
        else:
            raise RuntimeError('Unknown atten_type: {}'.format(self.atten_type))

    def forward(self, query_embed, in_memory_embed, atten_mask=None):
        if self.atten_type == 'simple': # simple attention
            attention = torch.bmm(in_memory_embed, query_embed.unsqueeze(2)).squeeze(2)# / math.sqrt(query_embed.size(-1))
        elif self.atten_type == 'mul': # multiplicative attention
            attention = torch.bmm(in_memory_embed, torch.mm(query_embed, self.W).unsqueeze(2)).squeeze(2)# / math.sqrt(query_embed.size(-1))
        elif self.atten_type == 'add': # additive attention
            attention = F.tanh(torch.mm(in_memory_embed.view(-1, in_memory_embed.size(-1)), self.W2)\
                .view(in_memory_embed.size(0), -1, self.W2.size(-1)) \
                + torch.mm(query_embed, self.W).unsqueeze(1))
            attention = torch.mm(attention.view(-1, attention.size(-1)), self.W3).view(attention.size(0), -1)
        else:
            raise RuntimeError('Unknown atten_type: {}'.format(self.atten_type))

        if atten_mask is not None:
            attention = atten_mask * attention - (1 - atten_mask) * INF
        return attention

def create_mask(x, N, use_cuda=True):
        x = x.data
        mask = np.zeros((x.size(0), N))
        for i in range(x.size(0)):
            mask[i, :x[i]] = 1
        return to_var(torch.Tensor(mask), use_cuda=use_cuda)

# def hard_sampling(x, use_cuda=True):
#     max_ids = torch.max(x, -1)[1].unsqueeze(-1)
#     if use_cuda:
#         max_ids = max_ids.cpu()
#     x_onehot = torch.zeros(x.size())
#     x_onehot.scatter_(1, max_ids.data, 1)
#     return to_var(x_onehot, use_cuda=use_cuda)

# def sample_gumbel(shape, eps=1e-20):
#     U = torch.rand(shape)
#     return -torch.log(-torch.log(U + eps) + eps)

# def gumbel_softmax_sample(logits, temperature=0.5, use_cuda=True):
#     sample = to_var(sample_gumbel(logits.size()), use_cuda=use_cuda)
#     y = logits + sample
#     return F.softmax(y / temperature, dim=-1)

# def gumbel_softmax(logits, temperature=0.5, use_cuda=True):
#     """
#     input: [*, n_class]
#     return: [*, n_class] an one-hot vector
#     """
#     y = gumbel_softmax_sample(logits, temperature, use_cuda)
#     shape = y.size()
#     _, ind = y.max(dim=-1)
#     y_hard = torch.zeros_like(y).view(-1, shape[-1])
#     y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     return (y_hard - y).detach() + y