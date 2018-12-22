from __future__ import print_function, division

import numpy as np
import torch

from ..advisor.utils import to_var


class Interpreter(object):
    """ Class to interpret models with given datasets.

    Args:
        batch_size (int, optional): batch size for interpreter (default: 64)
    """

    def __init__(self, batch_size=64, use_cuda=True):
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    # def interpret(self, model, data, num_samples=20, out_sampled_dialog='sampled_dialogs.txt'):
    #     """ Interprete a model on given dataset and return results.

    #     Args:
    #         model (models.networks): model to interpret
    #         data (dataset.dataset.Dataset): dataset to interpret

    #     Returns:
    #     """
    #     # print('Data size: {}'.format(len(data.data)))
    #     model.eval()
    #     num_batches = data.num_batches(self.batch_size)
    #     num_samples_per_batch = max(int(num_samples / num_batches), 1)
    #     dialog_idx = 0
    #     with open(out_sampled_dialog, 'w') as f:
    #         for batch_xs, batch_ys in data.make_batches(self.batch_size):
    #             ctx, cands, ctx_turns_num, ctx_utter_len, cands_utter_len, db_fields, db_cells, db_rows_num = batch_xs

    #             ctx = to_var(torch.LongTensor(ctx), self.use_cuda, inference_mode=True)
    #             ctx_utter_len = to_var(torch.LongTensor(ctx_utter_len), self.use_cuda, inference_mode=True)
    #             ctx_turns_num = to_var(torch.LongTensor(ctx_turns_num), self.use_cuda, inference_mode=True)
    #             cands = to_var(torch.LongTensor(cands), self.use_cuda, inference_mode=True)
    #             cands_utter_len = to_var(torch.LongTensor(cands_utter_len), self.use_cuda, inference_mode=True)
    #             db_fields = to_var(torch.LongTensor(db_fields), self.use_cuda, inference_mode=True)
    #             db_cells = to_var(torch.LongTensor(db_cells), self.use_cuda, inference_mode=True)
    #             db_rows_num = to_var(torch.LongTensor(db_rows_num), self.use_cuda, inference_mode=True)
    #             targets = to_var(torch.LongTensor(batch_ys), self.use_cuda, inference_mode=True)

    #             _, all_theta, all_acc = model(ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num)

    #             # Interpret
    #             all_theta = all_theta.cpu() if self.use_cuda else all_theta
    #             all_acc = all_acc.cpu() if self.use_cuda else all_acc
    #             all_theta = all_theta.data.numpy()
    #             all_acc = all_acc.data.numpy()

    #             sampled_dialog_ids = np.random.choice(ctx.size(0), num_samples_per_batch, replace=False)
    #             for i in sampled_dialog_ids: # dialog-level
    #                 dialog_idx += 1
    #                 f.write('Sampled dialog {}\n'.format(dialog_idx))
    #                 for j in range(0, int(ctx_turns_num[i]), 2): # turn-level
    #                     utter_1 = 'Advisor: ' + ' '.join([data.vocab.itos[int(x)] for x in ctx[i][j]]) + '\n\n'
    #                     utter_2 = 'Student: ' + ' '.join([data.vocab.itos[int(x)] for x in ctx[i][j + 1]]) + '\n\n'
    #                     f.write(utter_1 + utter_2)
    #                     theta = all_theta[i][int(j / 2)].tolist()
    #                     f.write('recommend: {}\tquery: {}\tinform: {}\tsocial: {}\n\n'.format(theta[0], theta[1], theta[2], theta[3]))
    #                     f.write('acc: ' + str(all_acc[i][int(j / 2)].tolist()) + '\n\n')
    #                     f.write('argmax acc: {}\n\n'.format(' '.join([data.vocab.itos[int(x)] for x in db_fields[i][np.argmax(all_acc[i][int(j / 2)])]])))
    #                 utter_2 = 'Advisor: ' + ' '.join([data.vocab.itos[int(x)] for x in cands[i][int(targets[i])]]) + '\n\n\n\n'
    #                 f.write(utter_2)

    def interpret(self, model, data, num_samples=20, out_sampled_dialog='sampled_dialogs.txt'):
        """ Interprete a model on given dataset and return results.

        Args:
            model (models.networks): model to interpret
            data (dataset.dataset.Dataset): dataset to interpret

        Returns:
        """
        # print('Data size: {}'.format(len(data.data)))
        model.eval()

        dialogs = []
        for batch_xs, batch_ys in data.make_batches(self.batch_size):
            ctx, cands, ctx_turns_num, ctx_utter_len, cands_utter_len, db_fields, db_cells, db_rows_num = batch_xs

            ctx = to_var(torch.LongTensor(ctx), self.use_cuda, inference_mode=True)
            ctx_utter_len = to_var(torch.LongTensor(ctx_utter_len), self.use_cuda, inference_mode=True)
            ctx_turns_num = to_var(torch.LongTensor(ctx_turns_num), self.use_cuda, inference_mode=True)
            cands = to_var(torch.LongTensor(cands), self.use_cuda, inference_mode=True)
            cands_utter_len = to_var(torch.LongTensor(cands_utter_len), self.use_cuda, inference_mode=True)
            db_fields = to_var(torch.LongTensor(db_fields), self.use_cuda, inference_mode=True)
            db_cells = to_var(torch.LongTensor(db_cells), self.use_cuda, inference_mode=True)
            db_rows_num = to_var(torch.LongTensor(db_rows_num), self.use_cuda, inference_mode=True)
            targets = to_var(torch.LongTensor(batch_ys), self.use_cuda, inference_mode=True)

            outputs, all_theta, all_acc, _ = model(ctx, ctx_utter_len, ctx_turns_num, cands, cands_utter_len, db_fields, db_cells, db_rows_num)

            # Interpret
            outputs = outputs.cpu().data.numpy() if self.use_cuda else outputs.data.numpy()
            predictions = np.argsort(-outputs, axis=1) # decreasing order
            ranks = []
            for i in range(predictions.shape[0]):
                ranks.append(np.nonzero(predictions[i] == batch_ys[i])[0][0] + 1)
            ranks = np.array(ranks)

            all_theta = all_theta.cpu() if self.use_cuda else all_theta
            all_acc = all_acc.cpu() if self.use_cuda else all_acc
            all_theta = all_theta.data.numpy()
            all_acc = all_acc.data.numpy()

            for i in range(ctx.size(0)): # dialog-level
                if ranks[i] <= 10:
                    continue
                dial_str = ''
                for j in range(0, int(ctx_turns_num[i]), 2): # turn-level
                    dial_str += 'Advisor: ' + ' '.join([data.vocab.itos[int(x)] for x in ctx[i][j]]) + '\n\n'
                    dial_str += 'Student: ' + ' '.join([data.vocab.itos[int(x)] for x in ctx[i][j + 1]]) + '\n\n'
                    theta = all_theta[i][int(j / 2)].tolist()
                    dial_str += 'recommend: {}\tquery: {}\tinform: {}\tsocial: {}\n\n'.format(theta[0], theta[1], theta[2], theta[3])
                    dial_str += 'acc: {}\n\n'.format(str(all_acc[i][int(j / 2)].tolist()))
                    dial_str += 'argmax acc: {}\n\n'.format(' '.join([data.vocab.itos[int(x)] for x in db_fields[i][np.argmax(all_acc[i][int(j / 2)])]]))
                # ground truth
                dial_str += 'Advisor (ground truth): ' + ' '.join([data.vocab.itos[int(x)] for x in cands[i][int(targets[i])]]) + '\n\n'
                # prediction
                dial_str += 'Advisor (top 1 prediction): ' + ' '.join([data.vocab.itos[int(x)] for x in cands[i][int(predictions[i][0])]]) + '\n\n'
                dial_str += 'Advisor (top 2 prediction): ' + ' '.join([data.vocab.itos[int(x)] for x in cands[i][int(predictions[i][1])]]) + '\n\n'
                dial_str += 'Advisor (top 3 prediction): ' + ' '.join([data.vocab.itos[int(x)] for x in cands[i][int(predictions[i][2])]]) + '\n\n'
                dial_str += 'Advisor (top 4 prediction): ' + ' '.join([data.vocab.itos[int(x)] for x in cands[i][int(predictions[i][3])]]) + '\n\n'
                dial_str += 'Advisor (top 5 prediction): ' + ' '.join([data.vocab.itos[int(x)] for x in cands[i][int(predictions[i][4])]]) + '\n\n\n\n'
                dialogs.append(dial_str)    
        
        with open(out_sampled_dialog, 'w') as f:
            np.random.shuffle(dialogs)
            for i, dial_str in enumerate(dialogs[:num_samples]):
                f.write('Sampled dialog {}\n'.format(i))
                f.write(dial_str)