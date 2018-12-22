from __future__ import print_function, division

import numpy as np
import torch

from ..advisor.modules import create_mask
from ..advisor.utils import to_var
from core.config import *


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (torch.NN.CrossEntropyLoss, optional): loss for evaluator (default: torch.NN.CrossEntropyLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss_func, batch_size=64, use_cuda=True, model_name=None):
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.model_name = model_name

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (models.networks): model to evaluate
            data (dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
            accuracy (float): accuracy of the given model on the given dataset
            recall (dict): recall of the given model on the given dataset
        """
        # print('Data size: {}'.format(len(data.data)))
        model.eval()

        loss = 0
        match = 0
        total = 0
        # recall = {'@1': 0, '@2': 0, '@5': 0, '@10': 0, '@50': 0, '@100': 0}
        recall = {'@1': 0, '@2': 0, '@5': 0}

        # device = None if torch.cuda.is_available() else -1
        num_batches = data.num_batches(self.batch_size)
        for batch_xs, batch_ys in data.make_batches(self.batch_size):
            batch_loss, batch_match, batch_recall, batch_num_samples = self.evaluate_batch(model, batch_xs, batch_ys)
            loss += batch_loss / num_batches
            match += batch_match
            total += batch_num_samples
            for k in recall:
                recall[k] += batch_recall[k]

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss, accuracy, {k: v/total for k, v in recall.items()}

    def evaluate_batch(self, model, batch_xs, batch_ys):
        model.eval()
        ctx, cands, ctx_turns_num, ctx_utter_len, cands_utter_len, db_fields, db_cells, db_rows_num = batch_xs

        ctx = to_var(torch.LongTensor(ctx), self.use_cuda, inference_mode=True)
        ctx_utter_len = to_var(torch.LongTensor(ctx_utter_len), self.use_cuda, inference_mode=True)
        ctx_turns_num = to_var(torch.LongTensor(ctx_turns_num), self.use_cuda, inference_mode=True)
        cands = to_var(torch.LongTensor(cands), self.use_cuda, inference_mode=True)
        cands_utter_len = to_var(torch.LongTensor(cands_utter_len), self.use_cuda, inference_mode=True)
        if self.model_name == 'advisor' or self.model_name == 'hred_db' or self.model_name == 'hred_db0':
            db_fields = to_var(torch.LongTensor(db_fields), self.use_cuda, inference_mode=True)
            db_cells = to_var(torch.LongTensor(db_cells), self.use_cuda, inference_mode=True)
            db_rows_num = to_var(torch.LongTensor(db_rows_num), self.use_cuda, inference_mode=True)
        if MODE_ON:
            targets = to_var(torch.LongTensor(batch_ys[0]), self.use_cuda, inference_mode=True)
            act_modes = to_var(torch.LongTensor(batch_ys[1]), self.use_cuda, inference_mode=True)
            act_modes_mask = create_mask(ctx_turns_num / 2, ctx.size(1) // 2, self.use_cuda)
        else:
            targets = to_var(torch.LongTensor(batch_ys), self.use_cuda, inference_mode=True)

        outputs, thetas, _, turn_loss = model(ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num)
        # outputs, thetas, _, turn_loss = model(ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num, targets=targets)

        # Get loss
        loss = self.loss_func(outputs, targets)
        # loss += turn_loss
        if MODE_ON and thetas is not None:
            mode_loss = -torch.log(thetas.gather(2, act_modes.unsqueeze(-1)).squeeze(-1))
            mode_loss = torch.sum(mode_loss * act_modes_mask) / mode_loss.size(0)
            loss += mode_loss

        # Evaluation
        recall = {}
        outputs = outputs.cpu() if self.use_cuda else outputs
        predictions = np.argsort(-outputs.data.numpy(), axis=1) # decreasing order
        num_samples = predictions.shape[0]
        ranks = []
        for i in range(num_samples):
            ranks.append(np.nonzero(predictions[i] == int(targets[i]))[0][0] + 1)
        ranks = np.array(ranks)
        match = sum(ranks == 1)
        recall['@1'] = match
        recall['@2'] = sum(ranks <= 2)
        recall['@5'] = sum(ranks <= 5)
        # recall['@10'] = sum(ranks <= 10)
        # recall['@50'] = sum(ranks <= 50)
        # recall['@100'] = sum(ranks <= 100)
        return float(loss), match, recall, num_samples