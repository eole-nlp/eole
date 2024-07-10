"""Module defining encoders."""
from eole.encoders.transformer import TransformerEncoder
from eole.encoders.rnn_encoder import RNNEncoder
from eole.encoders.cnn_encoder import CNNEncoder
from eole.encoders.mean_encoder import MeanEncoder


str2enc = {
    "rnn": RNNEncoder,
    "brnn": RNNEncoder,
    "cnn": CNNEncoder,
    "transformer": TransformerEncoder,
    "mean": MeanEncoder,
}
