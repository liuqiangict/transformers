""" TULRv5 model configuration """


import logging
from math import exp

from ...configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

TULRV5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tulrv5-base": "https://turingdatashare.blob.core.windows.net/projectalexander/languagemodels/tulrv5/base/config.json",
    "tulrv5-large": "https://turingdatashare.blob.core.windows.net/projectalexander/languagemodels/tulrv5/large/config.json"
}

class TULRv5Config(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a :class:`~transformers.TULRv5Model`.
        It is used to instantiate an TULRv5 model according to the specified arguments, defining the model
        architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.


        Args:
            vocab_size (:obj:`int`, optional, defaults to 500002):
                Vocabulary size of the TULRv5 model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
            hidden_size (:obj:`int`, optional, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (:obj:`int`, optional, defaults to 3072):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.TULRv5Model`.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            rel_pos_bins (:obj:`int`, optional, defaults to 32):
                No. of buckets used in relative position bias. 
            max_rel_pos (:obj:`int`, optional, defaults to 128):
                Max relative position length supported.
    """
    pretrained_config_archive_map = TULRV5_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "tulrv5"

    def __init__(self, 
                vocab_size=500002,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=514,
                type_vocab_size=1,
                initializer_range=0.02,
                layer_norm_eps=1e-05,
                expand_qk_dim=96,
                rel_pos_bins=32,
                max_rel_pos=128,
                **kwargs):
        super().__init__(**kwargs)
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
        self.layer_norm_eps = layer_norm_eps
        self.expand_qk_dim = expand_qk_dim
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
