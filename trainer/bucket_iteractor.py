import math
import torch
import random

from tqdm import tqdm
from transformers import AutoTokenizer
from cfgs.constants import MODEL_MAP

start_map = {
    "bert": "cls_token_id",
    "ma_bert": "cls_token_id",
    "user": "cls_token_id",
    "roberta": "cls_token_id",
    "flan_t5": None,
}
end_map = {
    "bert": "sep_token_id",
    "ma_bert": "sep_token_id",
    "user": "sep_token_id",
    "roberta": "eos_token_id",
    "flan_t5": "eos_token_id",
}
pad_map = {
    "bert": "pad_token_id",
    "ma_bert": "pad_token_id",
    "user": "pad_token_id",
    "roberta": "pad_token_id",
    "flan_t5": "pad_token_id",
}

instruct_text = {
    "bert": None,
    "ma_bert": None,
    "user": None,
    "roberta": None,
    "flan_t5": "Review:",
}

instruct_label = {
    "bert": None,
    "ma_bert": None,
    "user": None,
    "roberta": None,
    "flan_t5": "sentiment score:",
}


def _truncate_and_pad(tokens, start_id, end_id, pad_id, prefix_ids=None, max_length=510, pad_strategy="head"):
    total_length = len(tokens)
    prefix_ids = prefix_ids if prefix_ids is not None else []
    start_id = [start_id] if start_id is not None else []
    end_id = [end_id] if end_id is not None else []
    if total_length > max_length:
        if pad_strategy == 'head':
            return prefix_ids + start_id + tokens[:max_length] + end_id
        if pad_strategy == 'tail':
            return prefix_ids + start_id + tokens[-max_length:] + end_id
        if pad_strategy == 'both':
            # return [start_id] + tokens[:128] + tokens[-max_length+128:] + [end_id]
            return prefix_ids + start_id + tokens[:255] + tokens[-max_length+255:] + end_id
        return
    else:
        return prefix_ids + start_id + tokens + end_id + [pad_id] * (max_length-total_length)


class BucketIteratorForPLMTokenizer(object):
    def __init__(self, data,
                 batch_size,
                 plm_name=None,
                 max_length=512,
                 sort_index=0,
                 shuffle=True,
                 sort=True,
                 device='cpu',
                 trunc=None,
                 description="Train",
                 stoi=None,
                 ):
        self.shuffle = shuffle
        self.stoi = stoi
        self.sort = sort
        self.sort_key = sort_index
        self.max_length = max_length
        self.device = device
        self.description = description
        self.plm_name = plm_name
        self.trunc = trunc
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[self.plm_name])
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        dataloader_tqdm = tqdm(range(num_batch))
        dataloader_tqdm.set_description_str("Processing Dataloader: {}".format(self.description))
        for i in dataloader_tqdm:
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_usr_indices = []
        batch_prd_indices = []
        batch_labels = []

        max_len_words = max([len(t[0]) for t in batch_data])

        for item in batch_data:
            tokens_index, label, user_index, product_index = item
            if self.trunc:
                tokens_index = _truncate_and_pad(
                    tokens=tokens_index,
                    start_id=getattr(self.tokenizer, start_map[self.plm_name]) if start_map[self.plm_name] is not None else None,
                    end_id=getattr(self.tokenizer, end_map[self.plm_name]) if end_map[self.plm_name] is not None else None,
                    pad_id=getattr(self.tokenizer, pad_map[self.plm_name]),
                    prefix_ids=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instruct_text[self.plm_name]))
                    if instruct_text[self.plm_name] is not None else None,
                    pad_strategy=self.trunc,
                    max_length=max_len_words if max_len_words < self.max_length - 2 else self.max_length - 2
                )
            if instruct_label[self.plm_name] is not None:
                label = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instruct_label[self.plm_name] + str(label))) + \
                        [getattr(self.tokenizer, end_map[self.plm_name])]
            batch_text_indices.append(tokens_index)
            batch_labels.append(label)
            batch_usr_indices.append(self.stoi[user_index])
            batch_prd_indices.append(product_index)
        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)
        batch_labels = torch.tensor(batch_labels, device=self.device)
        batch_usr_indices = torch.tensor(batch_usr_indices, device=self.device)


        return {'input_ids': batch_text_indices,
                'attention_mask': batch_text_indices != getattr(self.tokenizer, pad_map[self.plm_name]),
                'labels': batch_labels,
                'p': batch_usr_indices
                }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.batch_len


class BucketIteratorForGloVeTokenizer(object):
    def __init__(self, data,
                 batch_size,
                 max_words=512,
                 max_sencs=512,
                 sort_index=0,
                 shuffle=True,
                 sort=True,
                 device='cpu',
                 trunc=None,
                 description="Train",
                 stoi=None,
                 pad_token_id=1,
                 ):
        self.shuffle = shuffle
        self.stoi = stoi
        self.sort = sort
        self.sort_key = sort_index
        self.max_sencs = max_sencs
        self.max_words = max_words
        self.device = device
        self.description = description
        self.trunc = trunc
        self.pad_token_id = pad_token_id
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        dataloader_tqdm = tqdm(range(num_batch))
        dataloader_tqdm.set_description_str("Processing Dataloader: {}".format(self.description))
        for i in dataloader_tqdm:
            batches.append(self.pad_data(sorted_data[i*batch_size: (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_usr_indices = []
        batch_prd_indices = []
        batch_labels = []

        max_len_sencs = max([len(t[0]) for t in batch_data])
        max_len_words = max([max([len(tt) for tt in t[0]]) for t in batch_data])

        def generate_pads(document_index):
            pads = []
            for sentence_index in document_index:
                pads.append([self.pad_token_id]*(max_len_words-len(sentence_index)))
            pads.extend([[self.pad_token_id]*max_len_words]*(max_len_sencs-len(document_index)))
            return pads

        for item in batch_data:
            document_index, label, user_index, product_index = item
            pads = generate_pads(document_index)
            for i in range(len(document_index)):
                document_index[i].extend(pads[i])
            if len(document_index) < len(pads):
                document_index.extend(pads[len(document_index)-len(pads):])
            if self.trunc:
                if max_len_words > self.max_words:
                    for sentence_index in document_index:
                        del sentence_index[self.max_words:]
                if max_len_sencs > self.max_sencs:
                    del document_index[self.max_sencs:]
            batch_text_indices.append(document_index)
            batch_labels.append(label)
            batch_usr_indices.append(self.stoi[user_index])
            batch_prd_indices.append(product_index)
        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)
        batch_labels = torch.tensor(batch_labels, device=self.device)
        batch_usr_indices = torch.tensor(batch_usr_indices, device=self.device)

        return {'input_ids': batch_text_indices,
                'attention_mask': batch_text_indices != self.pad_token_id,
                'labels': batch_labels,
                'p': batch_usr_indices
                }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.batch_len


class BucketIteratorForFewShot(object):
    def __init__(self, data, # BucketIteratorPLMs
                 batch_size,
                 plm_name=None,
                 shot_number=1,
                 sort_name='input_ids',
                 shuffle=True,
                 sort=True,
                 device='cpu',
                 ):
        self.shuffle = shuffle
        self.sort_key = sort_name
        self.device = device
        self.sort = sort
        self.shot_number = shot_number
        self.plm_name = plm_name
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[self.plm_name])
        self.data = data
        self.newdata = self.generate_new_datasets()
        self.batches = self.sort_and_pad(self.newdata, batch_size)
        self.batch_len = len(self.batches)

    def generate_new_datasets(self):
        data_with_users = {}
        for batch in self.data:
            for i in range(len(batch['p'])):
                if batch['p'][i].item() not in data_with_users.keys():
                    data_with_users[batch['p'][i].item()] = []
                    data_with_users[batch['p'][i].item()].append(
                        {"labels": batch['labels'][i],
                         "input_ids": batch['input_ids'][i],
                         "attention_mask": batch['attention_mask'][i],
                         "p": batch['p'][i]}
                    )
                else:
                    data_with_users[batch['p'][i].item()].append(
                        {"labels": batch['labels'][i],
                         "input_ids": batch['input_ids'][i],
                         "attention_mask": batch['attention_mask'][i],
                         "p": batch['p'][i]}
                    )

        new_train_datasets = []
        for k, v in data_with_users.items():
            new_train_datasets.extend(random.sample(v, self.shot_number))

        return new_train_datasets


    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        dataloader_tqdm = tqdm(range(num_batch))
        dataloader_tqdm.set_description_str("Processing Fewshot Train Dataloader with shot number of {}".format(self.shot_number))
        for i in dataloader_tqdm:
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_usr_indices = []
        batch_labels = []

        max_len_words = max([len(t["input_ids"]) for t in batch_data])

        for item in batch_data:
            tokens_index, label, user_index = item["input_ids"].tolist(), item["labels"], item["p"]
            getattr(self.tokenizer, pad_map[self.plm_name])
            tokens_index += [getattr(self.tokenizer, pad_map[self.plm_name])] * (max_len_words - len(tokens_index))
            batch_text_indices.append(tokens_index)
            batch_labels.append(item["labels"])
            batch_usr_indices.append(item["p"])
        batch_text_indices = torch.tensor(batch_text_indices, device=self.device)
        batch_labels = torch.stack(batch_labels, dim=0)
        batch_usr_indices = torch.stack(batch_usr_indices, dim=0)

        return {'input_ids': batch_text_indices,
                'attention_mask': batch_text_indices != getattr(self.tokenizer, pad_map[self.plm_name]),
                'labels': batch_labels,
                'p': batch_usr_indices
                }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

    def __len__(self):
        return self.batch_len