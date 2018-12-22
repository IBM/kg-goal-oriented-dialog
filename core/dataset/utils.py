import os
import logging
import json
import ijson
import shutil
import copy
import datetime
from tqdm import tqdm
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize


from core.config import *

logger = logging.getLogger(__name__)
IGNORE_PUNCT = '"#$&()*,-/;<=>@[\\]^_`{|}~' # keep . ! ? % : + '


def get_NEs_from_db(db):
    all_courses = set()
    all_course_titles = set()
    all_areas = set()
    all_categories = set()
    all_instructors = set()

    for course in db:
        rows = db[course]
        tmp = rows['Course'].lower().split()
        all_courses.update([tmp[0], ' '.join(tmp[1:])])
        all_course_titles.add(rows['CourseTitle'].lower())
        all_areas.add(rows['Area'].lower())
        all_categories.add(rows['Category'].lower())
        all_instructors.update([x.lower() for x in rows['section']])

    all_areas.discard('na')
    all_categories.discard('na')
    all_categories.discard('other')
    all_instructors.discard('na')
    all_nes = {'course': all_courses, 'course title': all_course_titles, 'area': all_areas, 'category': all_categories, 'instructor': all_instructors}
    return all_nes

def add_delexicalization(s, all_nes):
    # Make sure all the strings are in lower case
    for ne_type, nes in all_nes.items():
        for each in nes:
            if not each in s:
                continue
            s = re.sub(r'\b%s\b' % re.escape(each), '{} {}'.format(ne_type, each), s)
    return s


def flatten_list(l): 
    return (flatten_list(l[0]) + flatten_list(l[1:]) if len(l) > 0 else []) if type(l) is list else [l]


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def handle_punct(x):
    return re.sub('[%s]' % re.escape(IGNORE_PUNCT), ' ', re.sub('\.+', '.', x))

def read_json(input_file):
    json_objects_lst = list()
    json_objects = ijson.items(input_file, 'item')
    # i = 0
    for obj in json_objects:
        json_objects_lst.append(obj)
        # i += 1
        # if i >= 256:
        #     print("[ TESTING ]: only kept the first 256 examples")
        #     break

    return json_objects_lst

def prepare_data(path, db_path, tokenize_func=word_tokenize, format='JSON'):
    """
    Reads a tab-separated data file where each line contains a source sentence and a target sentence. Pairs containing
    a sentence that exceeds the maximum length allowed for its language are not added.
    Args:
        path (str): path to the data file
        db_path (str): path to the database file
        tokenize_func (func): function for splitting words in a sentence (default is single-space-delimited)
        format (str): data format for input file. Default is JSON.
    Returns:
        list((str, list(str)), str): list of ((context, list of candidates), target) string pairs
    """

    logger.info("Reading Lines from {}".format(path))
    # Read the file and split into lines
    pairs = []
    with open(path, 'r') as fin:
        if format == 'JSON':
            pairs = process(read_json(fin), json.load(open(db_path, 'r')), tokenize_func)
        elif format == 'CSV':
            pairs = read(fin, ",", tokenize_func)
        elif format == 'TSV':
            pairs = read(fin, ",", tokenize_func)

    logger.info("Number of pairs: %s" % len(pairs))
    return pairs

def read(fin, delimiter, tokenize_func):
    pairs = []
    for line in tqdm(fin):
        try:
            src, dst = line.strip().split(delimiter)
            pair = map(tokenize_func, [src, dst])
            pairs.append(pair)
        except:
            logger.error("Error when reading line: {0}".format(line))
            raise
    return pairs

# def process(records, db, tokenize_func):
#     # all_nes = get_NEs_from_db(db)
#     advisor_str = 'advisor'
#     pairs = []
#     course_dbs = []
#     for record in records:
#         context = []
#         act_modes = []
#         speaker = None
#         first_speaker = None
#         for msg in record['messages-so-far']:
#             ctx_utter = msg['utterance'].lower()
#             # Add delexicalization
#             # ctx_utter = add_delexicalization(ctx_utter, all_nes)
#             if speaker is None:
#                 context.append(ctx_utter + " __eou__ ")
#                 speaker = msg['speaker']
#                 first_speaker = msg['speaker']
#             elif speaker != msg['speaker']:
#                 context[-1] += "__eot__"
#                 context.append(ctx_utter + " __eou__ ")
#                 speaker = msg['speaker']
#                 if MODE_ON and speaker.lower() == advisor_str:
#                     if msg['dialog_act'] in act2mode:
#                         act_modes.append(mode2id[act2mode[msg['dialog_act']]])
#                     else:
#                         act_modes.append(3)
#             else:
#                 context[-1] += ctx_utter + " __eou__ "
#         context[-1] += "__eot__"

#         # Insert an empty turn if the original first speaker is student
#         # to make sure the context looks like this [advisor_1, student_1, advisor_2, student_2, ..., advisor_t-1, student_t-1]
#         # or you can treat it as [advisor_0, student_1, advisor_1, student_2, advisor_2, ..., student_t-1, advisor_t-1, student_t]
#         if first_speaker.lower() != advisor_str:
#             context.insert(0, "__eou__ __eot__")

#         # Ignore the last turn if it is from an advisor
#         # TODO: how to improve this?
#         if speaker.lower() == advisor_str:
#             del context[-1]
#         assert len(context) % 2 == 0

#         # Create the next utterance options and the target label
#         candidates = []
#         assert len(record['options-for-correct-answers']) == 1
#         correct_answer = record['options-for-correct-answers'][0]
#         target_id = correct_answer['candidate-id']
#         if MODE_ON:
#             if correct_answer['dialog_act'] in act2mode:
#                 act_modes.append(mode2id[act2mode[correct_answer['dialog_act']]])
#             else:
#                 act_modes.append(3)

#         tgt = None
#         for i, candidate in enumerate(record['options-for-next']):
#             if candidate['candidate-id'] == target_id:
#                 tgt = i
#             cand_utter = candidate['utterance'].lower()
#             # Add delexicalization
#             # cand_utter = add_delexicalization(cand_utter, all_nes)
#             candidates.append(tokenize_func(handle_punct(cand_utter)))

#         if tgt is None:
#             logger.info(
#                 'Correct answer not found in options-for-next - example {}.'.format(
#                     record['example-id']))
#         else:
#             course_cells = []
#             course_info = {}
#             for course in record['profile']['Courses']['Suggested']:
#                 course_id = course['offering']
#                 course_info = copy.deepcopy(db[course_id])
#                 # Hack
#                 if course['instructor'] == 'Hector Jose Garcia':
#                     course['instructor'] += '-Ramirez'

#                 course_info['Instructor'] = course['instructor']
#                 course_info['DaysOfClass'] = course_info['section'][course['instructor']]['DaysOfClass']
#                 course_info['TimeOfDay'] = course_info['section'][course['instructor']]['TimeOfDay']
#                 del course_info['section']
#                 # Ignore the Description field, it is too long compared to other fields
#                 # TODO: How to import this?
#                 del course_info['Description']
#                 row = []
#                 for field in course_info:
#                     entry = str(course_info[field]) if not isinstance(course_info[field], list) else ' '.join(flatten_list(course_info[field]))
#                     row.append(tokenize_func(handle_punct(entry.lower())))
#                 course_cells.append(row)

#             course_fields = [tokenize_func(handle_punct(' '.join(camel_case_split(field)).lower())) for field in course_info]
#             pairs.append((([tokenize_func(handle_punct(x)) for x in context], candidates, course_fields, course_cells), (tgt, act_modes) if MODE_ON else tgt))
#     return pairs


def process(records, db, tokenize_func):
    # Take 10 candidates
    num_cands = 10
    # all_nes = get_NEs_from_db(db)
    advisor_str = 'advisor'
    pairs = []
    course_dbs = []
    for record in records:
        context = []
        act_modes = []
        speaker = None
        first_speaker = None
        for msg in record['messages-so-far']:
            ctx_utter = msg['utterance'].lower()
            # Add delexicalization
            # ctx_utter = add_delexicalization(ctx_utter, all_nes)
            if speaker is None:
                context.append(ctx_utter + " __eou__ ")
                speaker = msg['speaker']
                first_speaker = msg['speaker']
            elif speaker != msg['speaker']:
                context[-1] += "__eot__"
                context.append(ctx_utter + " __eou__ ")
                speaker = msg['speaker']
                if MODE_ON and speaker.lower() == advisor_str:
                    if msg['dialog_act'] in act2mode:
                        act_modes.append(mode2id[act2mode[msg['dialog_act']]])
                    else:
                        act_modes.append(3)
            else:
                context[-1] += ctx_utter + " __eou__ "
        context[-1] += "__eot__"

        # Insert an empty turn if the original first speaker is student
        # to make sure the context looks like this [advisor_1, student_1, advisor_2, student_2, ..., advisor_t-1, student_t-1]
        # or you can treat it as [advisor_0, student_1, advisor_1, student_2, advisor_2, ..., student_t-1, advisor_t-1, student_t]
        if first_speaker.lower() != advisor_str:
            context.insert(0, "__eou__ __eot__")

        # Ignore the last turn if it is from an advisor
        # TODO: how to improve this?
        if speaker.lower() == advisor_str:
            del context[-1]
        assert len(context) % 2 == 0

        # Create the next utterance options and the target label
        assert len(record['options-for-correct-answers']) == 1
        correct_answer = record['options-for-correct-answers'][0]
        target_id = correct_answer['candidate-id']
        if MODE_ON:
            if correct_answer['dialog_act'] in act2mode:
                act_modes.append(mode2id[act2mode[correct_answer['dialog_act']]])
            else:
                act_modes.append(3)

        tgt = 0
        candidates = [tokenize_func(handle_punct(correct_answer['utterance'].lower()))]
        count = 0
        for i, candidate in enumerate(record['options-for-next']):
            if candidate['candidate-id'] == target_id:
                continue
            cand_utter = candidate['utterance'].lower()
            # Add delexicalization
            # cand_utter = add_delexicalization(cand_utter, all_nes)
            candidates.append(tokenize_func(handle_punct(cand_utter)))
            count += 1
            if count == num_cands - 1:
                break
        assert len(candidates) == num_cands
        
        if tgt is None:
            logger.info(
                'Correct answer not found in options-for-next - example {}.'.format(
                    record['example-id']))
        else:
            course_cells = []
            course_info = {}
            for course in record['profile']['Courses']['Suggested']:
                course_id = course['offering']
                course_info = copy.deepcopy(db[course_id])
                # Hack
                if course['instructor'] == 'Hector Jose Garcia':
                    course['instructor'] += '-Ramirez'

                course_info['Instructor'] = course['instructor']
                course_info['DaysOfClass'] = course_info['section'][course['instructor']]['DaysOfClass']
                course_info['TimeOfDay'] = course_info['section'][course['instructor']]['TimeOfDay']
                del course_info['section']
                # Ignore the Description field, it is too long compared to other fields
                # TODO: How to import this?
                del course_info['Description']
                row = []
                for field in course_info:
                    entry = str(course_info[field]) if not isinstance(course_info[field], list) else ' '.join(flatten_list(course_info[field]))
                    row.append(tokenize_func(handle_punct(entry.lower())))
                course_cells.append(row)

            course_fields = [tokenize_func(handle_punct(' '.join(camel_case_split(field)).lower())) for field in course_info]
            pairs.append((([tokenize_func(handle_punct(x)) for x in context], candidates, course_fields, course_cells), (tgt, act_modes) if MODE_ON else tgt))
    return pairs


def read_vocabulary(path, max_num_vocab=50000):
    """
    Helper function to read a vocabulary file.
    Args:
        path (str): filepath to raw vocabulary file
        max_num_vocab (int): maximum number of words to read from vocabulary file
    Returns:
        set: read words from vocabulary file
    """
    logger.info("Reading vocabulary from {}".format(path))
    # Read the file and create list of tokens in vocabulary
    vocab = set()
    with open(path) as fin:
        for line in fin:
            if len(vocab) >= max_num_vocab:
                break
            try:
                vocab.add(line.strip())
            except:
                logger.error("Error when reading line: {0}".format(line))
                raise

    logger.info("Size of Vocabulary: %s" % len(vocab))
    return vocab


def dump_embeddings(vocab, emb_file, out_path, emb_size=300, binary=False, seed=123):
    vocab_emb = get_embeddings(emb_file, vocab, binary)

    vocab_size = vocab.get_vocab_size()
    np.random.seed(seed)
    embeddings = np.random.uniform(-0.08, 0.08, (vocab_size, emb_size))
    for w, idx in vocab.stoi.items():
        if w in vocab_emb:
            embeddings[int(idx)] = vocab_emb[w]
    embeddings[0] = 0
    dump_ndarray(embeddings, out_path)
    return embeddings


def get_embeddings(emb_file, vocab, binary=False):
    pt = PreTrainEmbedding(emb_file, binary)
    vocab_embs = {}
    i = 0.
    for each in vocab.stoi:
        emb = pt.get_embeddings(each)
        if not emb is None:
            vocab_embs[each] = emb
            i += 1
    print('get_wordemb hit ratio: %s' % (i / vocab.get_vocab_size()))
    return vocab_embs


class PreTrainEmbedding():
    def __init__(self, file, binary=False):
        import gensim
        self.model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=binary)

    def get_embeddings(self, word):
        word_list = [word, word.upper(), word.lower(), word.title(), string.capwords(word, '_')]

        for w in word_list:
            try:
                return self.model[w]
            except KeyError:
                # print('Can not get embedding for ', w)
                continue
        return None


def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def dump_ndarray(data, path_to_file):
    try:
        with open(path_to_file, 'wb') as f:
            np.save(f, data)
    except Exception as e:
        raise e