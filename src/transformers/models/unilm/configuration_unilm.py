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
""" UniLM model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from ...configuration_utils import PretrainedConfig
from ..bert.configuration_bert import BertConfig

logger = logging.getLogger(__name__)

UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'unilm-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm-large-cased-config.json",
    'unilm-base-cased': "https://unilm.blob.core.windows.net/ckpt/unilm-base-cased-config.json",
    'unilm1-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased-config.json",
    'unilm1-base-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased-config.json",
    'unilm1.2-base-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased-config.json",
    'unilm2-base-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm2-base-uncased-config.json",
    'unilm2-large-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm2-large-uncased-config.json",
    'unilm2-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm2-large-cased-config.json",
}


class UniLMConfig(PretrainedConfig):
    r"""
        :class:`~transformers.UnilmConfig` is the configuration class to store the configuration of a
        `UnilmModel`.
        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `UnilmModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `UnilmModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = UNILM_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size=28996,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=6,
                 initializer_range=0.02,
                 layer_norm_type='post',
                 layer_norm_eps=1e-12,
                 source_type_id=0, 
                 target_type_id=1,
                 rel_pos_bins=0,
                 max_rel_pos=0,
                 cls_dropout_prob=None,
                 expand_qk_head_dim=None,
                 num_ffn_layers=0,
                 **kwargs):
        super(UniLMConfig, self).__init__(**kwargs)
        if isinstance(vocab_size, str) or (sys.version_info[0] == 2
                                           and isinstance(vocab_size, unicode)):
            with open(vocab_size, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size, int):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_type = layer_norm_type
            self.layer_norm_eps = layer_norm_eps
            self.source_type_id = source_type_id
            self.target_type_id = target_type_id
            self.rel_pos_bins = rel_pos_bins
            self.max_rel_pos = max_rel_pos
            self.cls_dropout_prob = cls_dropout_prob if cls_dropout_prob is not None else hidden_dropout_prob
            self.expand_qk_head_dim = expand_qk_head_dim
            self.num_ffn_layers = num_ffn_layers
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")


class UniLMFinetuningConfig(BertConfig):
    def __init__(self, label_smoothing=0.1, source_type_id=0, target_type_id=1, 
                 rel_pos_bins=0, max_rel_pos=0, fix_word_embedding=False,
                 expand_qk_head_dim=None, num_ffn_layers=1, need_pooler=False, **kwargs):
        super(UniLMFinetuningConfig, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fix_word_embedding = fix_word_embedding
        self.need_pooler = need_pooler
        self.expand_qk_head_dim = expand_qk_head_dim
        self.num_ffn_layers = num_ffn_layers

    @classmethod
    def from_exist_config(
            cls, config, label_smoothing=0.1, max_position_embeddings=None,
            need_pooler=False, fix_word_embedding=False,
    ):
        required_keys = [
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "hidden_act", "intermediate_size", "hidden_dropout_prob", "attention_probs_dropout_prob",
            "max_position_embeddings", "type_vocab_size", "initializer_range", "layer_norm_eps", 
        ]

        kwargs = {}
        for key in required_keys:
            assert hasattr(config, key)
            kwargs[key] = getattr(config, key)

        kwargs["vocab_size_or_config_json_file"] = kwargs["vocab_size"]
        if isinstance(config, RobertaConfig):
            kwargs["type_vocab_size"] = 0
            kwargs["max_position_embeddings"] = kwargs["max_position_embeddings"] - 2
        
        additional_keys = [
            "source_type_id", "target_type_id", "rel_pos_bins", "max_rel_pos", "expand_qk_head_dim", "num_ffn_layers",
        ]
        for key in additional_keys:
            if hasattr(config, key):
                kwargs[key] = getattr(config, key)

        if max_position_embeddings is not None and max_position_embeddings > config.max_position_embeddings:
            kwargs["max_position_embeddings"] = max_position_embeddings
            logger.info("  **  Change max position embeddings to %d  ** " % max_position_embeddings)

        return cls(
            label_smoothing=label_smoothing, need_pooler=need_pooler,
            fix_word_embedding=fix_word_embedding, **kwargs)
