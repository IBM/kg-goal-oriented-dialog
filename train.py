'''
Created on July, 2018

@author: hugo

'''
import os
import random
import timeit
import argparse
from torch.nn import CrossEntropyLoss

from core.dataset.dataset import Dataset
from core.advisor.advisor import AdvisorAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
    parser.add_argument('-mf', '--model_file', required=True, type=str, help='path to the output model')
    AdvisorAgent.add_cmdline_args(parser)
    opt = vars(parser.parse_args())

    # Load dataset
    train_file = 'train_data.pkl'
    train_data = Dataset.load(os.path.join(opt['data_dir'], train_file), vocab_file_name=os.path.join(opt['data_dir'], 'vocab.txt'))
    dev_data = Dataset.load(os.path.join(opt['data_dir'], 'dev_data.pkl'))
    dev_data.vocab = train_data.vocab
    dev_data.shuffle(1234)
    
    if opt['experiment'] == 1 or opt['experiment'] == 3:
        # Train: 90%, 10%
        data = train_data.data
        random.seed(1234)
        random.shuffle(data)
        n_valid = int(0.1 * len(data))
        train_data.data = data[:-n_valid]
        dev_data.data = data[-n_valid:]

    # data = train_data.data + dev_data.data
    # random.seed(1234)
    # random.shuffle(data)
    # n_valid = 500
    # train_data.data = data[:-n_valid]
    # dev_data.data = data[-n_valid:]

    # import numpy as np
    # size = 200
    # n_per = 1
    # data = []
    # np.random.seed(1234)
    # for i in range(0, len(train_data.data), size):
    #     sampled_ids = np.random.choice(range(i, i + size), n_per, replace=False)
    #     for each in sampled_ids:
    #         data.append(train_data.data[each])
    # del train_data.data[:]
    # train_data.data = data


    # size = 5
    # n_per = 1
    # data = []
    # np.random.seed(1234)
    # for i in range(0, len(dev_data.data), size):
    #     sampled_ids = np.random.choice(range(i, i + size), n_per, replace=False)
    #     for each in sampled_ids:
    #         data.append(dev_data.data[each])
    # del dev_data.data[:]
    # dev_data.data = data


    # all_data = train_data.data + dev_data.data

    # np.random.seed(1234)
    # np.random.shuffle(all_data)

    # train_data.data = all_data[:-100]
    # dev_data.data = all_data[-100:]


    # import numpy as np
    # np.random.seed(1234)
    # size = 200
    # data = []
    # n_valid = int(len(train_data.data) * 0.2)
    # for i in range(0, len(train_data.data), size):
    #     data.append(train_data.data[i: i + size])

    # np.random.shuffle(data)
    # del train_data.data[:]
    # del dev_data.data[:]
    # data = [x for y in data for x in y]
    # train_data.data = data[:-n_valid]
    # dev_data.data = data[-n_valid:]


    # import pdb;pdb.set_trace()

    start = timeit.default_timer()

    # Train
    model = AdvisorAgent(opt)
    model.train(train_data, dev_data)

    # Evaluate
    _, precision, recall = model.evaluator.evaluate(model.model, dev_data)
    print("Evaluation metrics on the validation set: Precision: {}, Recall: {}".format(precision, recall))

    print('Runtime: %ss' % (timeit.default_timer() - start))