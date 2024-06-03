import unittest
from eole.modules.embeddings import Embeddings

import itertools
from copy import deepcopy

import torch
import random
from torch.nn.init import xavier_uniform_
from eole.tests.utils_for_tests import product_dict


class TestEmbeddings(unittest.TestCase):
    INIT_CASES = list(
        product_dict(
            word_vec_size=[172],
            word_vocab_size=[319],
            word_padding_idx=[17],
            position_encoding=[False, True],
            dropout=[0, 0.5],
            freeze_word_vecs=[False, True],
        )
    )
    PARAMS = list(product_dict(batch_size=[1, 14], max_seq_len=[23]))

    @classmethod
    def case_is_degenerate(cls, case):
        return False

    @classmethod
    def cases(cls):
        for case in cls.INIT_CASES:
            if not cls.case_is_degenerate(case):
                yield case

    @classmethod
    def dummy_inputs(cls, params, init_case):
        max_seq_len = params["max_seq_len"]
        batch_size = params["batch_size"]
        n_words = init_case["word_vocab_size"]
        voc_sizes = [n_words]
        pad_idxs = [init_case["word_padding_idx"]]
        lengths = [random.randint(0, max_seq_len) for _ in range(batch_size)]
        lengths[0] = max_seq_len
        inps = torch.empty((batch_size, max_seq_len), dtype=torch.long)
        for (voc_size, pad_idx) in zip(voc_sizes, pad_idxs):
            for b, len_ in enumerate(lengths):
                inps[b, :len_] = torch.randint(0, voc_size - 1, (len_,))
                inps[b, len_:] = pad_idx
        return inps

    @classmethod
    def expected_shape(cls, params, init_case):
        wvs = init_case["word_vec_size"]
        size = wvs
        return params["batch_size"], params["max_seq_len"], size

    def test_embeddings_forward_shape(self):
        for params, init_case in itertools.product(self.PARAMS, self.cases()):
            emb = Embeddings(**init_case)
            dummy_in = self.dummy_inputs(params, init_case)
            res = emb(dummy_in)
            expected_shape = self.expected_shape(params, init_case)
            self.assertEqual(res.shape, expected_shape, init_case.__str__())

    def test_embeddings_trainable_params(self):
        for params, init_case in itertools.product(self.PARAMS, self.cases()):
            emb = Embeddings(**init_case)
            trainable_params = {
                n: p for n, p in emb.named_parameters() if p.requires_grad
            }
            # first check there's nothing unexpectedly not trainable
            for key in emb.state_dict():
                if key not in trainable_params:
                    if (
                        key.endswith("embeddings.weight")
                        and init_case["freeze_word_vecs"]
                    ):
                        # ok: word embeddings shouldn't be trainable
                        # if word vecs are freezed
                        continue
                    if key.endswith("pe.pe"):
                        # ok: positional encodings shouldn't be trainable
                        assert init_case["position_encoding"]
                        continue
                    else:
                        self.fail(
                            "Param {:s} is unexpectedly not " "trainable.".format(key)
                        )
            # then check nothing unexpectedly trainable
            if init_case["freeze_word_vecs"]:
                self.assertFalse(
                    any(
                        trainable_param.endswith("embeddings.weight")
                        for trainable_param in trainable_params
                    ),
                    "Word embedding is trainable but word vecs are freezed.",
                )
            if init_case["position_encoding"]:
                self.assertFalse(
                    any(
                        trainable_p.endswith(".pe.pe")
                        for trainable_p in trainable_params
                    ),
                    "Positional encoding is trainable.",
                )

    def test_embeddings_trainable_params_update(self):
        for params, init_case in itertools.product(self.PARAMS, self.cases()):
            emb = Embeddings(**init_case)
            for n, p in emb.named_parameters():
                if "weight" in n and p.dim() > 1:
                    xavier_uniform_(p)
            trainable_params = {
                n: p for n, p in emb.named_parameters() if p.requires_grad
            }
            if len(trainable_params) > 0:
                old_weights = deepcopy(trainable_params)
                dummy_in = self.dummy_inputs(params, init_case)
                res = emb(dummy_in)
                pretend_loss = res.mean()
                pretend_loss.backward()
                dummy_optim = torch.optim.SGD(trainable_params.values(), 1)
                dummy_optim.step()
                for param_name in old_weights.keys():
                    self.assertTrue(
                        trainable_params[param_name].ne(old_weights[param_name]).any(),
                        param_name + " " + init_case.__str__(),
                    )
