"""Module defining decoders."""

from eole.decoders.rnn_decoder import InputFeedRNNDecoder, StdRNNDecoder
from eole.decoders.transformer import TransformerDecoder
from eole.decoders.cnn_decoder import CNNDecoder


str2dec = {
    "rnn": StdRNNDecoder,
    "ifrnn": InputFeedRNNDecoder,
    "cnn": CNNDecoder,
    "transformer": TransformerDecoder,
}
