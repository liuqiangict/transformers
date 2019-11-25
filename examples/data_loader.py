
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

        '''
        query, passage, label = self.data.all_pairs[i]
        label = float(label)

        query_tokens = self.tokenizer.tokenize(query)
        passage_tokens = self.tokenizer.tokenize(passage)

        if(len(query_tokens) > self.max_seq_len // 2):
            query_tokens = query_tokens[0: self.max_seq_len // 2]

        max_passage_tokens = self.max_seq_len - \
            len(query_tokens) - 3  # Removing 3 for SEP and CLS

        if(len(passage_tokens) > max_passage_tokens):
            passage_tokens = passage_tokens[0:max_passage_tokens]

        # print(len(query_tokens) + len(passage_tokens))
        input_ids, input_mask, sequence_ids = encode_sequence(
            query_tokens, passage_tokens, self.max_seq_len, self.tokenizer)
        return tuple([input_ids, input_mask, sequence_ids, map_to_torch_float([label])])
        '''


        '''
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,

        def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):

        '''

        '''
        pad_on_left=False
        pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        pad_token_segment_id=0
        mask_padding_with_zero = True

        guid, query, passage, nlr_score, xlnet_score, roberta_score, turing_roberta_score, albert_score = self.data.all_pairs[i]
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

        #label = map_to_torch_float([float(la) for la in [nlr_score, xlnet_score, roberta_score, turing_roberta_score, albert_score]])
        label = map_to_torch_float([float(la) for la in [albert_score]])
        
        return tuple([map_to_torch([int(guid)]), map_to_torch(input_ids), map_to_torch(attention_mask), map_to_torch(token_type_ids), label])
        '''

        guid, str_input_ids, str_attention_mask, str_token_type_ids, str_label = self.data.all_pairs[i]
        guid = map_to_torch([int(guid)])
        input_ids = map_to_torch([int(id) for id in str_input_ids.split(';')])
        attention_mask = map_to_torch([int(id) for id in str_attention_mask.split(';')])
        token_type_ids = map_to_torch([int(id) for id in str_token_type_ids.split(';')])
        label = map_to_torch_float([float(str_label.split(';')[4])])

        return tuple([guid, input_ids, attention_mask, token_type_ids, label])
        #return InputFeatures(guids=guid, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label)

class QueryPassageFineTuningDataset:
    def __init__(self, path, readin=200000000, mode='eval'):
        all_pairs = []
        with open(path, encoding="utf-8") as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                entities = line.split('\t')
                data = tuple([entities[0], entities[1], entities[2], entities[3], entities[4]])
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