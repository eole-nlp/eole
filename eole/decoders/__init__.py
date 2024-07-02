"""Module defining decoders."""
from eole.decoders.rnn_decoder import InputFeedRNNDecoder, StdRNNDecoder
from eole.decoders.transformer_decoder import TransformerDecoder
from eole.decoders.transformer_lm_decoder import TransformerLMDecoder
from eole.decoders.cnn_decoder import CNNDecoder


str2dec = {
    "rnn": StdRNNDecoder,
    "ifrnn": InputFeedRNNDecoder,
    "cnn": CNNDecoder,
    "transformer": TransformerDecoder,
    "transformer_lm": TransformerLMDecoder,
}
