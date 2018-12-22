import random
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from .vocabulary import Vocabulary
from .utils import prepare_data, read_vocabulary
from core.config import *


class Dataset(object):
    """
    A class that encapsulates a dataset.

    Warning:
        Do not use this constructor directly, use one of the class methods to initialize.

    Note:
        Source or target sequences that are longer than the respective
        max length will be filtered.

    Args:
        max_len (int): maximum source sequence length
    """

    def __init__(self, data=None, vocab=None):
        # Declare vocabulary objects
        self.data = data
        self.vocab = vocab


    @classmethod
    def from_file(cls, path, db_path, vocab=None, max_vocab=100000):
        """
        Initialize a dataset from the file at given path. The file
        must contains a list of TAB-separated pairs of sequences.

        Note:
            Source or target sequences that are longer than the respective
            max length will be filtered.
            As specified by maximum vocabulary size, source and target
            vocabularies will be sorted in descending token frequency and cutoff.
            Tokens that are in the dataset but not retained in the vocabulary
            will be dropped in the sequences.

        Args:
            path (str): path to the dataset file
            db_path (str): path to the database file
            vocab (Vocabulary): pre-populated Vocabulary object or a path of a file containing words for the source language, default `None`. If a pre-populated Vocabulary object, `src_max_vocab` wouldn't be used.
            max_vocab (int): maximum source vocabulary size
        """
        obj = cls()
        pairs = prepare_data(path, db_path)
        return cls._encode(obj, pairs, vocab, max_vocab)

    def _encode(self, pairs, vocab=None, max_vocab=100000):
        """
        Encodes the source and target lists of sequences using source and target vocabularies.

        Note:
            Source or target sequences that are longer than the respective
            max length will be filtered.
            As specified by maximum vocabulary size, source and target
            vocabularies will be sorted in descending token frequency and cutoff.
            Tokens that are in the dataset but not retained in the vocabulary
            will be dropped in the sequences.

        Args:
            pairs (list): list of tuples (source sequences, target sequence)
            vocab (Vocabulary): pre-populated Vocabulary object or a path of a file containing words for the source language,
            default `None`. If a pre-populated Vocabulary object, `src_max_vocab` wouldn't be used.
            max_vocab (int): maximum source vocabulary size
        """
        # Read in vocabularies
        self.vocab = self._init_vocab(pairs, max_vocab, vocab)

        # Translate input sequences to token ids
        self.data = []
        for (context, candidates, course_fields, course_cells), target in pairs:
            c = []
            for utter in context:
                c.append(self.vocab.indices_from_sequence(utter))
            r = []
            for candidate in candidates:
                r.append(self.vocab.indices_from_sequence(candidate))
            f = []
            for field in course_fields:
                f.append(self.vocab.indices_from_sequence(field))
            ce = []
            for row in course_cells:
                ro = []
                for entry in row:
                    ro.append(self.vocab.indices_from_sequence(entry))
                ce.append(ro)
            self.data.append(((c, r, f, ce), target))
        return self

    def _init_vocab(self, data, max_num_vocab, vocab):
        resp_vocab = Vocabulary(max_num_vocab)
        if vocab is None:
            print('Building vocabulary...')
            for (context, candidates, course_fields, course_cells), target in data:
                for utter in context:
                    resp_vocab.add_sequence(utter)
                for candidate in candidates:
                    resp_vocab.add_sequence(candidate)
                for field in course_fields:
                    resp_vocab.add_sequence(field)
                for row in course_cells:
                    for entry in row:
                        resp_vocab.add_sequence(entry)
            resp_vocab.trim()
        elif isinstance(vocab, Vocabulary):
            print('Using pre-built vocabulary...')
            resp_vocab = vocab
        elif isinstance(vocab, str):
            print('Using pre-built vocabulary...')
            for tok in read_vocabulary(vocab, max_num_vocab):
                resp_vocab.add_token(tok)
        else:
            raise AttributeError('{} is not a valid instance on a vocabulary. None, instance of Vocabulary class \
                                 and str are only supported formats for the vocabulary'.format(vocab))
        return resp_vocab

    def _pad(self, data, max_turn_num=float('inf'), max_utter_len=float('inf'), max_cand_utter_len=float('inf')):
        all_c = [pair[0][0] for pair in data]
        all_r = [pair[0][1] for pair in data]
        all_f = [pair[0][2] for pair in data]
        all_ce = [pair[0][3] for pair in data]
        
        # Context
        batch_size = len(all_c)
        max_turn_num = min(max([len(c) for c in all_c], default=2), max_turn_num)
        max_utter_len = min(max([len(utter) for c in all_c for utter in c], default=1), max_utter_len)

        all_ctx = np.zeros([batch_size, max_turn_num, max_utter_len], dtype=int)
        all_ctx.fill(self.vocab.PAD_token_id)
        all_ctx_utter_lengths = np.ones([batch_size, max_turn_num], dtype=int)
        all_ctx_turn_num = np.zeros(batch_size, dtype=int)
        for i, c in enumerate(all_c):
            all_ctx_turn_num[i] = min(max(len(c), 2), max_turn_num) # in case context is empty
            for j, utter in enumerate(c[-max_turn_num:]):
                all_ctx[i, j, :len(utter)] = utter[-max_utter_len:]
                all_ctx_utter_lengths[i, j] = min(max(len(utter), 1), max_utter_len)

        # Candidate
        max_cand_num = max([len(r) for r in all_r], default=1)
        max_cand_utter_len = min(max([len(cand) for r in all_r for cand in r], default=1), max_cand_utter_len)
        all_resp = np.zeros([batch_size, max_cand_num, max_cand_utter_len], dtype=int)
        all_resp.fill(self.vocab.PAD_token_id)
        all_resp_utter_lengths = np.ones([batch_size, max_cand_num], dtype=int)

        for i, r in enumerate(all_r):
            for j, cand in enumerate(r):
                all_resp[i, j, :len(cand)] = cand[-max_cand_utter_len:]
                all_resp_utter_lengths[i, j] = min(max(len(cand), 1), max_cand_utter_len)

        # DB
        max_db_field_num = max([len(fields) for fields in all_f], default=1)
        max_db_field_len = max([len(f) for fields in all_f for f in fields], default=1)
        max_db_row_num = max([len(cells) for cells in all_ce], default=1)
        max_db_cell_len = max([len(entry) for cells in all_ce for row in cells for entry in row], default=1)
        all_fields = np.zeros([batch_size, max_db_field_num, max_db_field_len], dtype=int)
        all_fields.fill(self.vocab.PAD_token_id)
        all_cells = np.zeros([batch_size, max_db_row_num, max_db_field_num, max_db_cell_len], dtype=int)
        all_cells.fill(self.vocab.PAD_token_id)
        all_cells_row_num = np.zeros(batch_size, dtype=int)

        for i, fields in enumerate(all_f):
            for j, f in enumerate(fields):
                all_fields[i, j, :len(f)] = f

        for i, cells in enumerate(all_ce):
            all_cells_row_num[i] = len(cells)
            for j, row in enumerate(cells):
                for k, entry in enumerate(row):
                    all_cells[i, j, k, :len(entry)] = entry

        return (all_ctx, all_resp, all_ctx_turn_num, all_ctx_utter_lengths, all_resp_utter_lengths, all_fields, all_cells, all_cells_row_num)

    def __len__(self):
        return len(self.data)

    def num_batches(self, batch_size):
        """
        Get the number of batches given batch size.

        Args:
            batch_size (int): number of examples in a batch

        Returns:
            (int) : number of batches
        """
        # return len(range(0, len(self.data), batch_size))
        return len(self.data) // batch_size + (len(self.data) % batch_size != 0)

    def make_batches(self, batch_size):
        """
        Create a generator that generates batches in batch_size over data.

        Args:
            batch_size (int): number of pairs in a mini-batch

        Yields:
            (list (str), list (str)): next pair of source and target variable in a batch
        """
        if len(self.data) < batch_size:
            raise OverflowError("batch size = {} cannot be larger than data size = {}".
                                format(batch_size, len(self.data)))
        for i in range(0, len(self.data), batch_size):
            cur_batch = self.data[i:i + batch_size]
            xs = self._pad(cur_batch)
            # target = [pair[1] for pair in cur_batch]
            if MODE_ON:
                target = np.asarray([pair[1][0] for pair in cur_batch])
                max_count = max([len(pair[1][1]) for pair in cur_batch], default=1)
                all_act_modes = np.zeros([target.shape[0], max_count], dtype=int)
                for i, pair in enumerate(cur_batch):
                    for j, mode in enumerate(pair[1][1]):
                        all_act_modes[i, j] = mode
                target = (target, all_act_modes)
            else:
                target = np.asarray([pair[1] for pair in cur_batch])
            yield (xs, target)

    def shuffle(self, seed=None):
        """
        Shuffle the data.

        Args:
            seed (int): provide a value for the random seed; default seed=None is truly random
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)

    def save(self, data_file_name, vocab_file_name=None):
        """
        Writes this Dataset to disk in a pickle file.

        Args:
             file_name (str): path to the target pickle file
        """
        with open(data_file_name, "wb") as data_f:
            pickle.dump(self.data, data_f)

        if vocab_file_name:
            self.vocab.save(vocab_file_name)

    @classmethod
    def load(cls, data_file_name, vocab_file_name=None):
        """
        Loads a Vocabulary from a pickle file on disk.

        Args:
            file_name (str): path to the pickle file

        Returns:
            Vocabulary: loaded Vocabulary
        """
        with open(data_file_name, "rb") as data_f:
            data = pickle.load(data_f)
        vocab = Vocabulary.load(vocab_file_name) if vocab_file_name else None
        dataset = Dataset(data=data, vocab=vocab)
        return dataset