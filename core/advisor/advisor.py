'''
Created on July, 2018

@author: hugo

'''
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MultiLabelMarginLoss, CrossEntropyLoss
import torch.backends.cudnn as cudnn

import os
import numpy as np

# from .modules_mask import Advisor, create_mask
from .modules import Advisor, create_mask
from .hred_db import HRED_DB
from .hred_db0 import HRED_DB0
from .hred import HRED
from .bilstm import BiLSTM
from .utils import to_var
from ..utils.io_utils import load_ndarray
from ..evaluator.evaluator import Evaluator
from core.config import *


class AdvisorAgent(object):
    """docstring for AdvisorAgent"""
    def __init__(self, opt):
        super(AdvisorAgent, self).__init__()
        self.model_name = opt['model_name']
        self.evaluate_every = opt['evaluate_every_steps']
        if self.model_name == 'advisor':
            Module = Advisor
        elif self.model_name == 'hred_db':
            Module = HRED_DB
        elif self.model_name == 'hred_db0':
            Module = HRED_DB0
        elif self.model_name == 'hred':
            Module = HRED
        else:
            Module = BiLSTM

        opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
        if opt['cuda']:
            print('[ Using CUDA ]')
            torch.cuda.device(opt['gpu'])
            # torch.cuda.device([0, 1])
            
            # It enables benchmark mode in cudnn, which
            # leads to faster runtime when the input sizes do not vary.
            cudnn.benchmark = True

        self.opt = opt
        if opt['pre_word2vec']:
            pre_w2v = load_ndarray(opt['pre_word2vec'])
        else:
            pre_w2v = None

        self.evaluator = Evaluator(CrossEntropyLoss(), batch_size=opt['batch_size'], use_cuda=opt['cuda'], model_name=self.model_name)
        self.score_type = 'clf'

        self.model = Module(opt['vocab_size'], \
                opt['word_emb_size'], \
                opt['hidden_size'], \
                init_w2v=pre_w2v, \
                enc_type=opt['enc_type'], \
                rnn_type=opt['rnn_type'], \
                bidirectional=not opt['no_bidirectional'], \
                utter_enc_dropout=opt['utter_enc_dropout'], \
                knowledge_enc_dropout=opt['knowledge_enc_dropout'], \
                atten_type=opt['atten_type'], \
                score_type=self.score_type, \
                use_cuda=opt['cuda']#, \
                # phase=opt['phase']
                )

        if opt['cuda']:
            self.model.cuda()
            # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])


        if self.score_type == 'ranking':
            # MultiLabelMarginLoss
            # For each sample in the mini-batch:
            # loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x.size(0)
            self.loss_fn = MultiLabelMarginLoss()
        else:
            self.loss_fn = CrossEntropyLoss()

        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        lr = opt['learning_rate']
        if opt['optimizer'] == 'sgd':
            self.optimizers = {self.model_name: optim.SGD(optim_params, lr=lr)}
        elif opt['optimizer'] == 'adam':
            self.optimizers = {self.model_name: optim.Adam(optim_params, lr=lr)}
        elif opt['optimizer'] == 'adadelta':
            self.optimizers = {self.model_name: optim.Adadelta(optim_params, lr=lr)}
        elif opt['optimizer'] == 'adagrad':
            self.optimizers = {self.model_name: optim.Adagrad(optim_params, lr=lr)}
        elif opt['optimizer'] == 'adamax':
            self.optimizers = {self.model_name: optim.Adamax(optim_params, lr=lr)}
        elif opt['optimizer'] == 'rmsprop':
            self.optimizers = {self.model_name: optim.RMSprop(optim_params, lr=lr)}
        else:
            raise NotImplementedError('Optimizer not supported.')

        self.scheduler = ReduceLROnPlateau(self.optimizers[self.model_name], mode='min', \
                    patience=opt['valid_patience'] // 3, verbose=True)

        if opt.get('model_file') and os.path.isfile(opt['model_file']):
            print('Loading existing model parameters from ' + opt['model_file'])
            self.load(opt['model_file'])

    # Evaluate on every xxx steps
    def train(self, train_data, valid_data, seed=1234):        
        print('Training size: {}, Validation size: {}'.format(len(train_data.data), len(valid_data.data)))
        n_incr_error = 0  # nb. of consecutive increase in error
        best_loss = float("inf")
        num_batches = train_data.num_batches(self.opt['batch_size'])
        num_valid_batches = valid_data.num_batches(self.opt['batch_size'])
        global_steps = 0
        train_loss = 0
        eval_count = 0
        temperature = 1
        train_match = 0
        # train_recall = {'@1': 0, '@2': 0, '@5': 0, '@10': 0, '@50': 0, '@100': 0}
        train_recall = {'@1': 0, '@2': 0, '@5': 0}
        train_total = 0
        for epoch in range(1, self.opt['num_epochs'] + 1):
            train_data.shuffle(seed)
            for batch_xs, batch_ys in train_data.make_batches(self.opt['batch_size']):
                train_loss += self.train_batch(batch_xs, batch_ys, temperature=max(0.1, temperature - 0.1 * eval_count)) / self.evaluate_every

                # Evaluate on training set
                _, batch_match, batch_recall, batch_num_samples = self.evaluator.evaluate_batch(self.model, batch_xs, batch_ys)
                train_match += batch_match
                train_total += batch_num_samples
                for k in train_recall:
                    train_recall[k] += batch_recall[k]

                global_steps += 1
                if global_steps % self.evaluate_every == 0:
                    eval_count += 1
                    n_incr_error += 1

                    # Evaluate on validation set
                    valid_loss, valid_precision, valid_recall = self.evaluator.evaluate(self.model, valid_data)
                    # valid_loss = 0
                    # print('Temperature: {}'.format(max(0.1, temperature - 0.1 * eval_count)))
                    # for batch_valid_xs, batch_valid_ys in valid_data.make_batches(self.opt['batch_size']):
                    #     valid_loss += self.train_batch(batch_valid_xs, batch_valid_ys, temperature=max(0.1, temperature - 0.1 * eval_count), is_training=False) / num_valid_batches

                    print('Step {}/{}: Training loss: {:.4}, validation loss: {:.4}'.format(global_steps, self.opt['num_epochs'] * num_batches, train_loss, valid_loss))

                    if train_total == 0:
                        train_match = float('nan')
                    else:
                        train_match = train_match / train_total
                    train_recall = {k: v/train_total for k, v in train_recall.items()}
                    print("Evaluation metrics on the training set: Precision: {}, Recall: {}".format(train_match, train_recall))
                    print("Evaluation metrics on the validation set: Precision: {}, Recall: {}".format(valid_precision, valid_recall))

                    self.scheduler.step(valid_loss)
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        n_incr_error = 0
                        self.save()

                    if n_incr_error >= self.opt['valid_patience']:
                        print('Early stopping occured. Optimization Finished!')
                        return
                    train_loss = 0
                    train_match = 0
                    # train_recall = {'@1': 0, '@2': 0, '@5': 0, '@10': 0, '@50': 0, '@100': 0}
                    train_recall = {'@1': 0, '@2': 0, '@5': 0}
                    train_total = 0

                if self.opt['cuda'] and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()


    # # Evaluate on every epoch
    # def train(self, train_data, valid_data, seed=1234):
    #     print('Training size: {}, Validation size: {}'.format(len(train_data.data), len(valid_data.data)))
    #     n_incr_error = 0  # nb. of consecutive increase in error
    #     best_loss = float("inf")
    #     num_batches = train_data.num_batches(self.opt['batch_size'])
    #     num_valid_batches = valid_data.num_batches(self.opt['batch_size'])
        
    #     for epoch in range(1, self.opt['num_epochs'] + 1):
    #         train_data.shuffle(seed)
    #         n_incr_error += 1
    #         train_loss = 0
    #         for batch_xs, batch_ys in train_data.make_batches(self.opt['batch_size']):
    #             train_loss += self.train_batch(batch_xs, batch_ys) / num_batches

    #         valid_loss = 0
    #         for batch_valid_xs, batch_valid_ys in valid_data.make_batches(self.opt['batch_size']):
    #             valid_loss += self.train_batch(batch_valid_xs, batch_valid_ys, is_training=False) / num_valid_batches

    #         print('Epoch {}/{}: Training loss: {:.4}, validation loss: {:.4}'.format(epoch, self.opt['num_epochs'], train_loss, valid_loss))
    #         self.scheduler.step(valid_loss)
    #         if valid_loss < best_loss:
    #             # Evaluate
    #             precision, recall = self.evaluator.evaluate(self.model, valid_data)
    #             print("Evaluation metrics on the validation set: Precision: {}, Recall: {}".format(precision, recall))

    #             best_loss = valid_loss
    #             n_incr_error = 0
    #             self.save()

    #         if n_incr_error >= self.opt['valid_patience']:
    #             print('Early stopping occured. Optimization Finished!')
    #             break

    # def predict(self, data):
    #     print('Test size: {}'.format(len(data.data)))
    #     predictions = []
    #     for batch_xs, _ in data.make_batches(self.opt['batch_size']):
    #         batch_pred = self.predict_batch(batch_xs)
    #         predictions.extend(batch_pred)
    #     return predictions

    def train_batch(self, xs, ys, temperature=0.1, is_training=True):
        # Sets the module in training mode.
        # This has any effect only on modules such as Dropout or BatchNorm.
        self.model.train(mode=is_training)
        # Organize inputs for network
        ctx, cands, ctx_turns_num, ctx_utter_len, cands_utter_len, db_fields, db_cells, db_rows_num = xs
        ctx = to_var(torch.LongTensor(ctx), self.opt['cuda'], inference_mode=not is_training)
        ctx_utter_len = to_var(torch.LongTensor(ctx_utter_len), self.opt['cuda'], inference_mode=not is_training)
        ctx_turns_num = to_var(torch.LongTensor(ctx_turns_num), self.opt['cuda'], inference_mode=not is_training)
        cands = to_var(torch.LongTensor(cands), self.opt['cuda'], inference_mode=not is_training)
        cands_utter_len = to_var(torch.LongTensor(cands_utter_len), self.opt['cuda'], inference_mode=not is_training)
        if self.model_name == 'advisor' or self.model_name == 'hred_db' or self.model_name == 'hred_db0':
            db_fields = to_var(torch.LongTensor(db_fields), self.opt['cuda'], inference_mode=not is_training)
            db_cells = to_var(torch.LongTensor(db_cells), self.opt['cuda'], inference_mode=not is_training)
            db_rows_num = to_var(torch.LongTensor(db_rows_num), self.opt['cuda'], inference_mode=not is_training)
        
        if MODE_ON:
            targets = to_var(torch.LongTensor(ys[0]), self.opt['cuda'], inference_mode=not is_training)
            act_modes = to_var(torch.LongTensor(ys[1]), self.opt['cuda'], inference_mode=not is_training)
            act_modes_mask = create_mask(ctx_turns_num / 2, ctx.size(1) // 2, self.opt['cuda'])
        else:
            targets = to_var(torch.LongTensor(ys), self.opt['cuda'], inference_mode=not is_training)
        # import pdb;pdb.set_trace()

        scores, thetas, _, turn_loss = self.model(ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num, temperature=temperature)
        
        if self.score_type == 'ranking':
            targets = self.pack_margin_loss_targets(targets, cands.size(1))
            targets = to_var(targets, self.opt['cuda'])
        loss = self.loss_fn(scores, targets)
        # loss += turn_loss
        if MODE_ON and self.model_name == 'advisor':
            mode_loss = -torch.log(thetas.gather(2, act_modes.unsqueeze(-1)).squeeze(-1))
            mode_loss = torch.sum(mode_loss * act_modes_mask) / mode_loss.size(0)
            loss += mode_loss

        if is_training:
            for o in self.optimizers.values():
                o.zero_grad()
            loss.backward()
            for o in self.optimizers.values():
                o.step()
        return float(loss)

    # def predict_batch(self, xs):
    #     self.model.train(mode=False)
    #     # Organize inputs for network
    #     ctx, cands, ctx_turns_num, ctx_utter_len, cands_utter_len, db_fields, db_cells, db_rows_num = xs

    #     ctx = to_var(torch.LongTensor(ctx), self.opt['cuda'])
    #     ctx_utter_len = to_var(torch.LongTensor(ctx_utter_len), self.opt['cuda'])
    #     ctx_turns_num = to_var(torch.LongTensor(ctx_turns_num), self.opt['cuda'])
    #     cands = to_var(torch.LongTensor(cands), self.opt['cuda'])
    #     cands_utter_len = to_var(torch.LongTensor(cands_utter_len), self.opt['cuda'])
    #     db_fields = to_var(torch.LongTensor(db_fields), self.opt['cuda'])
    #     db_cells = to_var(torch.LongTensor(db_cells), self.opt['cuda'])
    #     db_rows_num = to_var(torch.LongTensor(db_rows_num), self.opt['cuda'])

    #     scores = self.model(ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num)
    #     return scores

    def pack_margin_loss_targets(self, x, N, placeholder=-1):
        y = np.ones((len(x), N), dtype='int64') * placeholder
        for i in range(len(x)):
            y[i, :len(x[i])] = x[i]
        return torch.LongTensor(y)

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path
        if path:
            checkpoint = {}
            checkpoint[self.model_name] = self.model.state_dict()
            checkpoint['{}_optim'.format(self.model_name)] = self.optimizers[self.model_name].state_dict()
            with open(path, 'wb') as write:
                torch.save(checkpoint, write)
                print('Saved model to {}'.format(path))

    def load(self, path):
        with open(path, 'rb') as read:
            checkpoint = torch.load(read)
        self.model.load_state_dict(checkpoint[self.model_name])
        self.optimizers[self.model_name].load_state_dict(checkpoint['{}_optim'.format(self.model_name)])


    # def gen_dummy_data(self):
    #     # Dummy data
    #     # ctx: [[x_1, y_1, ..., x_{t-1}, y_{t-1}, ..., x_t], ...], B x T x L
    #     # ctx_utter_len: [[len_x_1, len_y_1, ..., len_x_t], ...], B x T 
    #     # ctx_turns_num: [T_1, ...], B x 1
    #     # cands: [[r_1, ..., r_m], ...], B x m x L
    #     # cands_utter_len: [[len_r_1, ..., len_r_m], ...], B x m
    #     # db_fields: [[f_1, ..., f_k], ...], B x K x Lf
    #     # db_cells: [[[c_{11}, ..., c_{1K}], ..., [c_{N1}, ..., c_{NK}]], ...], B x N x K x Lc
    #     # db_rows_num: [N_1, ...], B x 1
    #     # targets: [[label_1, ..., label_m], ...], B x m
    #     B = 8
    #     T = 11
    #     L = 12
    #     m = 10
    #     N = 16
    #     K = 5
    #     Lc = 6
    #     Lf = 3
    #     vocab_size = self.opt['vocab_size']
    #     ctx = to_var(torch.LongTensor(np.random.randint(vocab_size, size=(B, T, L))), self.opt['cuda'], False)
    #     ctx_utter_len = to_var(torch.LongTensor(np.random.randint(1, L + 1, size=(B, T))), self.opt['cuda'], False)
    #     ctx_turns_num = to_var(torch.LongTensor(np.random.randint(1, T + 1, size=(B, ))), self.opt['cuda'], False)
    #     cands = to_var(torch.LongTensor(np.random.randint(vocab_size, size=(B, m, L))), self.opt['cuda'], False)
    #     cands_utter_len = to_var(torch.LongTensor(np.random.randint(1, L + 1, size=(B, m))), self.opt['cuda'], False)
    #     db_fields = to_var(torch.LongTensor(np.random.randint(vocab_size, size=(B, K, Lf))), self.opt['cuda'], False)
    #     db_cells = to_var(torch.LongTensor(np.random.randint(vocab_size, size=(B, N, K, Lc))), self.opt['cuda'], False)
    #     db_rows_num = to_var(torch.LongTensor(np.random.randint(1, N + 1, size=(B, ))), self.opt['cuda'], False)
    #     # targets = torch.LongTensor(np.random.randint(2, size=(B, m)))
    #     targets = to_var(torch.LongTensor(np.random.randint(-1, m, size=(B, m))), self.opt['cuda'], False)
    #     return ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num, targets

    @staticmethod
    def add_cmdline_args(argparser):
        arg_group = argparser.add_argument_group('Advisor Arguments')
        arg_group.add_argument('-lr', '--learning_rate', type=float, default=0.001,
            help='learning rate')
        arg_group.add_argument('-batch_size', '--batch_size', type=int, default=32,
            help='learning rate')
        arg_group.add_argument('-e', '--num_epochs', type=int, default=100,
            help='num of epochs')
        arg_group.add_argument('-vp', '--valid_patience', type=int, default=10,
            help=('number of iterations of validation where result'
                             ' does not improve before we stop training'))
        arg_group.add_argument('-vocab_size', '--vocab_size', type=int, required=True,
            help='size of token embeddings')
        arg_group.add_argument('-pre_w2v', '--pre_word2vec', type=str,
            help='path to the pretrained word embeddings')
        arg_group.add_argument('-word_emb_size', '--word_emb_size', type=int, default=300,
            help='size of token embeddings')
        arg_group.add_argument('-hidden_size', '--hidden_size', type=int, default=128,
            help='hidden size')
        arg_group.add_argument('-enc_type', '--enc_type', type=str, default='cnn',
            help='utterance encoder type')
        arg_group.add_argument('-rnn_type', '--rnn_type', type=str, default='lstm',
            help='encoder type')
        arg_group.add_argument('--no_bidirectional', action='store_true',
            help='encoder: bidirectional')
        arg_group.add_argument('-utter_enc_dropout', '--utter_enc_dropout', type=float, default=0.3,
            help='dropout probability for utterance encoder')
        arg_group.add_argument('-knowledge_enc_dropout', '--knowledge_enc_dropout', type=float, default=0.3,
            help='dropout probability for knowledge encoder')
        arg_group.add_argument('-att', '--atten_type', type=str, default='add',
            help='memory update: {simple, add, mul}')
        # arg_group.add_argument('-max_utter_len', '--max_utter_len', type=int, default=32,
            # help='max utterance length')
        arg_group.add_argument('-margin', '--margin', type=float, default=1, help='margin loss: margin')
        arg_group.add_argument('-opt', '--optimizer', default='adam',
            help='optimizer type (sgd|adam|adadelta)')
        arg_group.add_argument('-model_name', '--model_name', default='advisor',
            help='model name (advisor|hred_db|hred|bilstm)')
        arg_group.add_argument('-exp', '--experiment', type=int, default=0,
            help='model name (1|2|3)')
        arg_group.add_argument('-eval_every', '--evaluate_every_steps', type=int, default=1000,
            help='hidden size')
        arg_group.add_argument('--no_cuda', action='store_true',
            help='disable GPUs even if available')
        arg_group.add_argument('--gpu', type=int, default=-1,
            help='which GPU device to use')
        arg_group.add_argument('--phase', type=int, default=2,
            help='phase')
