import os

PRE_TRAINED_VECTOR_PATH = 'pretrained_Vectors'
if not os.path.exists(PRE_TRAINED_VECTOR_PATH):
    os.makedirs(PRE_TRAINED_VECTOR_PATH)

DATASET_PATH = 'corpus'
CKPTS_PATH = 'ckpts'
LOG_PATH = 'logs'

DATASET_PATH_MAP = {
    "imdb2a": os.path.join(DATASET_PATH, 'imdb2', "a"),
    "imdb2b": os.path.join(DATASET_PATH, 'imdb2', "b"),
    "yelp2a": os.path.join(DATASET_PATH, 'yelp2', "a"),
    "yelp2b": os.path.join(DATASET_PATH, 'yelp2', "b"),
    "gdrda": os.path.join(DATASET_PATH, 'gdrd', "a"),
    "gdrdb": os.path.join(DATASET_PATH, 'gdrd', "b"),
    "ppra": os.path.join(DATASET_PATH, 'ppr', "a"),
    "pprb": os.path.join(DATASET_PATH, 'ppr', "b"),
}

MODEL_MAP = {
    'bert': 'bert-base-uncased',
    'bert_large': 'bert-large-uncased',
    'roberta': 'roberta-base',
    'flan_t5': 'google/flan-t5-base',
    'user': 'bert-base-uncased',
    "ma_bert": 'bert-base-uncased',
}

DATASET_FEW_SHOT_MAP = { # to map few or zero-shot learning from models trained in previous datasets
    "imdb2a": "imdb2a",
    "imdb2b": "imdb2a",
    "yelp2a": "yelp2a",
    "yelp2b": "yelp2a",
    "gdrda": "gdrda",
    "gdrdb": "gdrda",
    "ppra": "ppra",
    "pprb": "ppra",
}
