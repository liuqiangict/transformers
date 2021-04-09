# coding=utf-8
# The MIT License (MIT)

# Copyright (c) Microsoft Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Tokenization classes for UniLM."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open
from typing import List, Optional

from ..bert.tokenization_bert import BertTokenizer, whitespace_tokenize
from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES_de = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP_de = {
    'vocab_file':
    {
        'unilm-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm-large-cased-vocab.txt",
        'unilm-base-cased': "https://unilm.blob.core.windows.net/ckpt/unilm-base-cased-vocab.txt",
        'unilm1-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased-vocab.txt",
        'unilm1-base-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased-vocab.txt",
        'unilm1.2-base-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased-vocab.txt", 
        'unilm2-base-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm2-base-uncased-vocab.txt", 
        'unilm2-large-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm2-large-uncased-vocab.txt", 
        'unilm2-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm2-large-cased-vocab.txt", 
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES_de = {
    'unilm-large-cased': 512,
    'unilm-base-cased': 512,
    'unilm1-large-cased': 512,
    'unilm1-base-cased': 512,
    'unilm1.2-base-uncased': 512,
    'unilm2-base-uncased': 512,
    'unilm2-large-cased': 512,
    'unilm2-large-uncased': 512,
}


class UniLMTokenizer_de(BertTokenizer):
    r"""
    Constructs a UnilmTokenizer.
    :class:`~transformers.UnilmTokenizer` is identical to BertTokenizer and runs end-to-end tokenization: punctuation splitting + wordpiece
    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_wordpiece_only=False
    """

    vocab_files_names = VOCAB_FILES_NAMES_de
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP_de
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES_de


UniLM_VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

UniLM_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "unilm3-base-uncased": "https://unilm.blob.core.windows.net/ckpt/unilm3-uncased.model",
        "unilm3-base-cased": "https://unilm.blob.core.windows.net/ckpt/unilm3-cased.model",
        # "unilm3-large-uncased": "https://unilm.blob.core.windows.net/ckpt/unilm3-uncased.model",
        # "unilm3-large-cased": "https://unilm.blob.core.windows.net/ckpt/unilm3-cased.model",
    }
}

UniLM_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "unilm-base-uncased": 512,
    "unilm-base-cased": 512,
}

UniLM_PRETRAINED_INIT_CONFIGURATION = {
    "unilm-base-uncased": {"do_lower_case": True},
    "unilm-base-cased": {"do_lower_case": False},
    # "unilm3-large-uncased": {"do_lower_case": True},
    # "unilm3-large-cased": {"do_lower_case": False},
}


class UniLMTokenizer(XLMRobertaTokenizer):

    vocab_files_names = UniLM_VOCAB_FILES_NAMES
    pretrained_init_configuration = UniLM_PRETRAINED_INIT_CONFIGURATION
    pretrained_vocab_files_map = UniLM_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = UniLM_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        super(UniLMTokenizer, self).__init__(
            vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        addition_tokens = ["<m>", "[X_SEP]"]
        for i, token in enumerate(addition_tokens):
            token_id = len(self.sp_model) + self.fairseq_offset + i + 1
            self.fairseq_tokens_to_ids[token] = token_id
            self.fairseq_ids_to_tokens[token_id] = token

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A XLM-R sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    @property
    def vocab(self):
        return self.get_vocab()

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 3  # Add the <p> <mask> [X_SEP] token

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


class UniLMAutoTokenizer(object):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        if pretrained_model_name_or_path in PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES:
            return UnilmTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        if pretrained_model_name_or_path in UniLMv3_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES:
            logger.info("Use UniLM v3 Tokenizer !")
            return UniLMv3Tokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        tokenizer = None
        try:
            tokenizer = UniLMv3Tokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        except:
            pass
        if tokenizer is None:
            tokenizer = UnilmTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return tokenizer
