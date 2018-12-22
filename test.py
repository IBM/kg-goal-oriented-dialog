'''
Created on July, 2018

@author: hugo

'''
import os
import timeit
import argparse

from core.dataset.dataset import Dataset
from core.advisor.advisor import AdvisorAgent
from core.evaluator.interpreter import Interpreter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
    parser.add_argument('-mf', '--model_file', required=True, type=str, help='path to the output model')
    parser.add_argument('-act', '--action', default='evaluate', type=str, help='action type')
    parser.add_argument('-out_sample', '--out_sampled_dialog', default='out_sampled_dialog.txt', type=str, help='path to sampled dialogs')
    parser.add_argument('-num_sample', '--num_sampled_dialog', default=50, type=int, help='number of sampled dialogs')
    AdvisorAgent.add_cmdline_args(parser)
    opt = vars(parser.parse_args())

    # Load dataset
    data = Dataset.load(os.path.join(opt['data_dir'], 'dev_data.pkl'), os.path.join(opt['data_dir'], 'vocab.txt'))
    data.shuffle(1234)
    start = timeit.default_timer()

    model = AdvisorAgent(opt)
    if opt['action'] == 'evaluate':
        _, precision, recall = model.evaluator.evaluate(model.model, data)
        print("Precision: {}, Recall: {}".format(precision, recall))
    elif opt['action'] == 'interpret':
        interpreter = Interpreter(opt['batch_size'], opt['cuda'])
        interpreter.interpret(model.model, data, num_samples=opt['num_sampled_dialog'], out_sampled_dialog=opt['out_sampled_dialog'])
        print('Saved sampled dialogs to {}'.format(opt['out_sampled_dialog']))
    else:
        raise RuntimeError('Unknow action'.format(opt['action']))
    print('Runtime: %ss' % (timeit.default_timer() - start))