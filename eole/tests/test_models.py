import copy
import unittest
import torch
import pyonmttok
from eole.constants import DefaultTokens
from collections import Counter
import eole
import eole.inputters

from eole.models.model import build_src_emb, build_tgt_emb, build_encoder, build_decoder
from eole.utils.misc import sequence_mask
from eole.config.run import TrainConfig
from eole.config.data import Dataset
from eole.config.models import CustomModelConfig

# In theory we should probably call ModelConfig here,
# but we can't because model building relies on some params
# not currently in model config (dropout, freeze_word_vecs_enc, etc.)

opt = TrainConfig(
    data={
        "dummy": Dataset(path_src="eole/tests/data/src-train.txt")
    },  # actual file path (tested in validate)
    src_vocab="dummy",
    share_vocab=True,
    model=CustomModelConfig(),
)  # now required by validation


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.opt = opt
        self.opt.training.self_attn_backend = "pytorch"

    def get_vocabs(self):
        src_vocab = pyonmttok.build_vocab_from_tokens(
            Counter(),
            maximum_size=0,
            minimum_frequency=1,
            special_tokens=[
                DefaultTokens.UNK,
                DefaultTokens.PAD,
                DefaultTokens.BOS,
                DefaultTokens.EOS,
            ],
        )

        tgt_vocab = pyonmttok.build_vocab_from_tokens(
            Counter(),
            maximum_size=0,
            minimum_frequency=1,
            special_tokens=[
                DefaultTokens.UNK,
                DefaultTokens.PAD,
                DefaultTokens.BOS,
                DefaultTokens.EOS,
            ],
        )

        vocabs = {"src": src_vocab, "tgt": tgt_vocab}
        return vocabs

    def get_batch(self, source_l=3, bsize=1):
        # batch x len
        test_src = torch.ones(bsize, source_l).long()
        test_tgt = torch.ones(bsize, source_l).long()
        test_length = torch.ones(bsize).fill_(source_l).long()
        return test_src, test_tgt, test_length

    def embeddings_forward(self, opt, source_l=3, bsize=1):
        """
        Tests if the embeddings works as expected

        args:
            opt: set of options
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        """
        vocabs = self.get_vocabs()
        emb = build_src_emb(opt.model, vocabs, running_config=opt.training)
        test_src, _, __ = self.get_batch(source_l=source_l, bsize=bsize)
        if opt.model.decoder.decoder_type == "transformer":
            input = torch.cat([test_src, test_src], 1)
            res = emb(input)
            compare_to = torch.zeros(
                bsize, source_l * 2, opt.model.embeddings.src_word_vec_size
            )
        else:
            res = emb(test_src)
            compare_to = torch.zeros(
                bsize, source_l, opt.model.embeddings.src_word_vec_size
            )

        self.assertEqual(res.size(), compare_to.size())

    def encoder_forward(self, opt, source_l=3, bsize=1):
        """
        Tests if the encoder works as expected

        args:
            opt: set of options
            source_l: Length of generated input sentence
            bsize: Batchsize of generated input
        """
        # if opt.hidden_size > 0:
        #     opt.encoder.hidden_size = opt.hidden_size
        vocabs = self.get_vocabs()
        embeddings = build_src_emb(opt.model, vocabs, running_config=opt.training)
        enc = build_encoder(opt.model, running_config=opt.training)

        test_src, test_tgt, test_length = self.get_batch(source_l=source_l, bsize=bsize)
        emb_src = embeddings(test_src)
        mask = sequence_mask(test_length)
        enc_out, hidden_t = enc(emb_src, mask)

        # Initialize vectors to compare size with
        test_hid = torch.zeros(
            opt.model.encoder.layers, bsize, opt.model.encoder.hidden_size
        )
        test_out = torch.zeros(bsize, source_l, opt.model.decoder.hidden_size)

        # Ensure correct sizes and types
        self.assertEqual(test_hid.size(), hidden_t[0].size(), hidden_t[1].size())
        self.assertEqual(test_out.size(), enc_out.size())
        self.assertEqual(type(enc_out), torch.Tensor)

    def model_forward(self, opt, source_l=3, bsize=1):
        """
        Creates a model with a custom opt function.
        Forwards a testbatch and checks output size.

        Args:
            opt: Namespace with options
            source_l: length of input sequence
            bsize: batchsize
        """
        # should be already done in validator
        # if opt.hidden_size > 0:
        #     opt.encoder.hidden_size = opt.hidden_size
        #     opt.decoder.hidden_size = opt.hidden_size
        vocabs = self.get_vocabs()

        src_emb = build_src_emb(opt.model, vocabs, running_config=opt.training)
        enc = build_encoder(opt.model, running_config=opt.training)

        tgt_emb = build_tgt_emb(
            opt.model,
            vocabs,
            running_config=opt.training,
        )
        dec = build_decoder(opt.model, running_config=opt.training)

        model = eole.models.model.EncoderDecoderModel(
            encoder=enc,
            decoder=dec,
            src_emb=src_emb,
            tgt_emb=tgt_emb,
            hidden_size=opt.model.decoder.hidden_size,
        )
        test_src, test_tgt, test_length = self.get_batch(source_l=source_l, bsize=bsize)
        output, attn, estim = model(test_src, test_tgt, test_length)
        outputsize = torch.zeros(bsize, source_l - 1, opt.model.decoder.hidden_size)
        # Make sure that output has the correct size and type
        self.assertEqual(output.size(), outputsize.size())
        self.assertEqual(type(output), torch.Tensor)


def _add_test(param_setting, methodname):
    """
    Adds a Test to TestModel according to settings

    Args:
        param_setting: list of tuples of (param, setting)
        methodname: name of the method that gets called
    """

    def test_method(self):
        opt = copy.deepcopy(self.opt)
        if param_setting:
            for param, setting in param_setting:
                setattr(opt, param, setting)
        getattr(self, methodname)(opt)

    if param_setting:
        name = "test_" + methodname + "_" + "_".join(str(param_setting).split())
    else:
        name = "test_" + methodname + "_standard"
    setattr(TestModel, name, test_method)
    test_method.__name__ = name


"""
TEST PARAMETERS
"""
# opt.brnn = False # deprecated and not used here

test_embeddings = [
    [("model", {"architecture": "rnn"})],
    [("model", {"architecture": "transformer"})],
]

for p in test_embeddings:
    _add_test(p, "embeddings_forward")

tests_encoder = [
    [
        # ("encoder_type", "mean"),
        (
            "model",
            {
                "architecture": "custom",
                "encoder": {"encoder_type": "mean"},
                "decoder": {"decoder_type": "rnn"},
            },
        )
    ],
    # [('encoder_type', 'transformer'),
    # ('word_vec_size', 16), ('hidden_size', 16)],
]

for p in tests_encoder:
    _add_test(p, "encoder_forward")

tests_model = [
    # [("rnn_type", "GRU")],
    [
        (
            "model",
            {
                "architecture": "rnn",
                "encoder": {"rnn_type": "GRU"},
                "decoder": {"rnn_type": "GRU"},
            },
        )
    ],
    [("model", {"architecture": "rnn", "layers": 10})],
    [("model", {"architecture": "rnn", "input_feed": 0})],
    [
        (
            "model",
            {
                "architecture": "transformer",
                "hidden_size": 16,
                "embeddings": {"word_vec_size": 16},
                # "word_vec_size": 16,  # can not be set at model level right now
                # (would need to define field in ModelConfig)
            },
        )
    ],  # not sure what this one is for (and how it works without position_encoding)
    [
        (
            "model",
            {
                "architecture": "transformer",
                "hidden_size": 16,
                "embeddings": {"word_vec_size": 16, "position_encoding": True},
            },
        )
        # ("decoder_type", "transformer"),
        # ("encoder_type", "transformer"),
        # ("src_word_vec_size", 16),
        # ("tgt_word_vec_size", 16),
        # ("position_encoding", True),
    ],
    [
        ("model", {"architecture": "rnn", "decoder": {"coverage_attn": True}})
        # ideally we should not have to pass decoder_type to default to rnn,
        # but pydantic does not seem to agree
    ],
    [("model", {"architecture": "rnn", "decoder": {}})],
    [
        ("model", {"architecture": "rnn", "decoder": {"global_attention": "mlp"}})
        # ("global_attention", "mlp"),
    ],
    [
        ("model", {"architecture": "rnn", "decoder": {"context_gate": "both"}})
        # ("context_gate", "both"),
    ],
    [
        (("model", {"architecture": "rnn", "decoder": {"context_gate": "target"}}))
        # ("context_gate", "target"),
    ],
    [("model", {"architecture": "rnn", "decoder": {"context_gate": "source"}})],
    # [("encoder_type", "brnn"), ("brnn_merge", "sum")],
    # not sure about this one, brnn_merge does not seem to exist in the codebase
    [
        ("model", {"architecture": "rnn", "encoder": {"encoder_type": "brnn"}})
        # ("encoder_type", "brnn"),
    ],
    [
        (
            "model",
            {
                "architecture": "cnn",
            },
        )
        # ("decoder_type", "cnn"), ("encoder_type", "cnn"),
    ],
    [
        ("model", {"architecture": "rnn", "decoder": {"global_attention": None}})
        # ("encoder_type", "rnn"), ("global_attention", None),
    ],
    [
        (
            "model",
            {
                "architecture": "rnn",
                "decoder": {
                    "global_attention": None,
                },
            },
        )
        # ("encoder_type", "rnn"),
        # ("global_attention", None),
    ],
    [
        (
            "model",
            {
                "architecture": "rnn",
                "decoder": {
                    "decoder_type": "rnn",  # not necessary, but adding here to test the case
                    "global_attention": "mlp",
                },
            },
        )
        # ("encoder_type", "rnn"),
        # ("global_attention", "mlp"),
    ],
]


for p in tests_model:
    _add_test(p, "model_forward")
