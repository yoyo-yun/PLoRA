import json
import re
import sys
import csv
import torch
import pandas as pd
import numpy as np
from collections import Counter

csv.field_size_limit(sys.maxsize)


class InputExample(object):
    def __init__(self, guid=None, text=None, user=None, product=None, label=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.user = user
        self.product = product


class SentenceProcessor(object):
    NAME = 'SENTENCE'

    def get_sentences(self):
        raise NotImplementedError

    def _create_examples(self, documents, type):
        examples = []
        for (i, line) in enumerate(documents):
            guid = "%s-%s" % (type, i)
            text = clean_document(line[2])
            examples.append(
                InputExample(guid=guid, user=line[0], product=line[1], text=text, label=int(line[3]) - 1))
        return examples

    def _read_file(self, dataset):
        pd_reader = pd.read_csv(dataset, header=None, skiprows=0, encoding="utf-8", sep='\t\t', engine='python')
        documents = []
        for i in range(len(pd_reader[0])):
            # if i == 1000:
            #     break
            # [ user, product, review, lable]
            document = list([pd_reader[0][i], pd_reader[1][i], pd_reader[3][i], pd_reader[2][i]])
            documents.append(document)
        return documents

    def _creat_sent_doc(self, *datasets):
        documents = []
        for dataset in datasets:
            for id, document in enumerate(dataset):
                user = document[0]
                product = document[1]
                review = document[2]
                label = int(document[3]) - 1
                documents.append(InputExample(user=user, product=product, text=split_sents(clean_document(review)), label=label))
        return documents

    def _creat_sent_doc_(self, *datasets):
        documents = []
        for dataset in datasets:
            for id, document in enumerate(dataset):
                user = document[0]
                product = document[1]
                review = document[2]
                label = int(document[3]) - 1
                documents.append(InputExample(user=user, product=product, text=split_sents(clean_document(review)), label=label))
        return documents


    def _get_attributes(self, *datasets):
        users = Counter()
        products = Counter()
        ATTR_MAP = {
            'user': int(0),
            'product': int(1)
        }
        for dataset in datasets:
            for document in dataset:
                users.update([document[ATTR_MAP["user"]]])
                products.update([document[ATTR_MAP["product"]]])
        return tuple([users, products])


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


class UnknownUPVecCache(object):
    @classmethod
    def unk(cls, tensor):
        return tensor.uniform_(-0.25, 0.25)


def clean_document(document):
    string = re.sub(r"<sssss>", "", document)
    string = re.sub(r" n't", "n't", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'.`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"sssss", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


def split_sents(string):
    string = re.sub(r"[!?]"," ", string)
    string = re.sub(r"\.{2,}", " ", string)
    return string.strip().split('.')


def clean_string_nltk(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    return string.lower().strip().split()


def split_sents_nltk(string):
    from nltk.tokenize import sent_tokenize
    string = re.sub(r"[^A-Za-z().,!?\'`]", " ", string)
    string = re.sub(r"\n{2,}", " ", string)
    string = re.sub(r"\.{2,}", " ", string)
    return sent_tokenize(string.replace('\n',''))


def generate_ngrams(tokens, n=2):
    n_grams = zip(*[tokens[i:] for i in range(n)])
    tokens.extend(['-'.join(x) for x in n_grams])
    return tokens


def load_json(string):
    split_val = json.loads(string)
    return np.asarray(split_val, dtype=np.float32)


def process_labels(string):
    """
    Returns the label string as a list of integers
    """
    return [float(x) for x in string]