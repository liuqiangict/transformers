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

from .data.processors.utils import InputFeatures

def map_to_torch(encoding):
    encoding = torch.LongTensor(encoding)
    encoding.requires_grad_(False)
    return encoding

def map_to_torch_float(encoding):
    encoding = torch.FloatTensor(encoding)
    encoding.requires_grad_(False)
    return encoding

class DTDataset(Dataset):
    def __init__(self, tokenizer, data, max_seq_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        guid, docs, start_idx, end_idx = self.data.all_pairs[index]
        inputs = self.tokenizer.encode_plus(
            guid,
            docs,
            start_idx,
            end_idx,
            max_length=self.max_seq_len
        )

        #guid = map_to_torch([inputs['guid']])
        #input_ids = map_to_torch(inputs['input_ids'])
        #attention_mask = map_to_torch(inputs['attention_mask'])
        #token_type_ids = map_to_torch(inputs['token_type_ids'])
        #start_pos = map_to_torch([inputs['start_pos']])
        #end_pos = map_to_torch([inputs['end_pos']])

        #return tuple([guid, input_ids, attention_mask, token_type_ids, start_pos, end_pos])
        #return {'guid': guid, 'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'start_positions': start_pos, 'end_positions': end_pos}
        feature = InputFeatures(
                    guids=inputs['guid'],
                    input_ids=inputs['input_ids'], 
                    token_type_ids=inputs['token_type_ids'], 
                    attention_mask=inputs['attention_mask'], 
                    start_positions=inputs['start_pos'],
                    end_positions=inputs['end_pos'])
        return feature



class DeepThinkDataset:
    def __init__(self, path, readin=200000000, mode='train'):
        all_pairs = []
        with open(path, encoding="utf-8") as fd:
            for i, line in enumerate(tqdm(fd)):
                cols = line.strip().split('\t')
                if len(cols) != 6:
                    continue
                guid = int(cols[0])
                docs = [cols[1]]
                json_docs = json.loads(cols[2])
                for doc in json_docs:
                    docs.append(doc['Text'])
                start_idx = int(cols[4]) + 1
                end_idx = int(cols[5]) + 1

                all_pairs.append([guid, docs, start_idx, end_idx])
                if i > readin:
                    break
        
        #if mode == 'train':
        #    shuffle(all_pairs)
        self.all_pairs = all_pairs
        self.len = len(self.all_pairs)

    def __len__(self):
        return self.len


