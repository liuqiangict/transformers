
import argparse
import csv
import logging
import os
import random
import sys
import json
import datetime
from itertools import product

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist
import torch.nn.functional as F
from random import shuffle
from typing import Tuple

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures

def map_to_torch(encoding):
    encoding = torch.LongTensor(encoding)
    encoding.requires_grad_(False)
    return encoding

def map_to_torch_float(encoding):
    encoding = torch.FloatTensor(encoding)
    encoding.requires_grad_(False)
    return encoding

class QADataset(Dataset):
    def __init__(self, tokenizer, data, max_seq_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index % self.len

        pad_on_left=False
        pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        pad_token_segment_id=0
        mask_padding_with_zero = True

        guid, query, passage, label = self.data.all_pairs[i]
        inputs = self.tokenizer.encode_plus(
            query,
            passage,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        label = map_to_torch_float([float(label)])
        #label = map_to_torch_float([float(la) for la in [positive_softmax_logits]])
        
        return tuple([map_to_torch([int(guid)])[0], map_to_torch(input_ids), map_to_torch(attention_mask), map_to_torch(token_type_ids), label[0]])

class QueryPassageFineTuningDataset:
    def __init__(self, path, readin=300000000, mode='eval'):
        all_pairs = []
        with open(path, encoding="utf-8") as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                entities = line.split('\t')
                data = tuple([entities[0], entities[1], entities[2], entities[3]])
                all_pairs.append(data)
                if i > readin:
                    break
        
        # This is done fro deterministic nature for teacher data gen
        if mode == 'train':
            print(str(datetime.datetime.now()), ' Start shuffle.')
            shuffle(all_pairs)
            print(str(datetime.datetime.now()), ' Finish shuffle.')
        self.all_pairs = all_pairs
        self.len = len(self.all_pairs)

    def __len__(self):
        return self.len
