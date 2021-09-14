import logging
from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "tulrv5-base": "https://turingdatashare.blob.core.windows.net/projectalexander/languagemodels/tulrv5/vocab/sentencepiece.bpe.model",
        "tulrv5-large": "https://turingdatashare.blob.core.windows.net/projectalexander/languagemodels/tulrv5/vocab/sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "tulrv5-base": 512,
    "tulrv5-large": 512,
}

class TULRv5Tokenizer(XLMRobertaTokenizer):
    """
        Adapted from RobertaTokenizer and XLNetTokenizer
        SentencePiece based tokenizer. Peculiarities:

        - requires `SentencePiece <https://github.com/google/sentencepiece>`_

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
