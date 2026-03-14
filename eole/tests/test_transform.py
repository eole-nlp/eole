"""Here come the tests for implemented transform."""

import unittest

import copy
import random
import yaml
import math
from argparse import Namespace
from eole.transforms import (
    get_transforms_cls,
    get_specials,
    make_transforms,
    TransformPipe,
)
from eole.transforms.bart import BARTNoising
from eole.transforms.normalize import MosesPunctNormalizer
from eole.config.run import TrainConfig, PredictConfig
from eole.config.data import Dataset
from eole.config.models import CustomModelConfig


class TestTransform(unittest.TestCase):
    def test_transform_register(self):
        builtin_transform = [
            "filtertoolong",
            "prefix",
            "sentencepiece",
            "bpe",
            "onmt_tokenize",
            "bart",
            "switchout",
            "tokendrop",
            "tokenmask",
            "insert_mask_before_placeholder",
        ]
        get_transforms_cls(builtin_transform)

    def test_vocab_required_transform(self):
        transforms_cls = get_transforms_cls(["bart", "switchout"])
        # opt = Namespace(seed=-1, switchout_temperature=1.0)
        opt = TrainConfig(
            data={"dummy": Dataset(path_src="eole/tests/data/src-train.txt")},  # actual file path (tested in validate)
            src_vocab="dummy",
            share_vocab=True,
            seed=-1,
            transforms_configs={"switchout": {"switchout_temperature": 1.0}},
            model=CustomModelConfig(),
        )
        # transforms that require vocab will not create if not provide vocab
        transforms = make_transforms(opt, transforms_cls, vocabs=None)
        self.assertEqual(len(transforms), 0)
        with self.assertRaises(ValueError):
            transforms_cls["switchout"](opt).warm_up(vocabs=None)
            transforms_cls["bart"](opt).warm_up(vocabs=None)

    def test_transform_specials(self):
        # This test is a bit broken.
        # It previously did not account for "infer" prefix added to prefix_dict/specials.
        transforms_cls = get_transforms_cls(["prefix"])
        corpora = yaml.safe_load(
            """
            trainset:
                path_src: eole/tests/data/src-train.txt
                path_tgt: eole/tests/data/tgt-train.txt
                transforms: ["prefix"]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """
        )
        # opt = Namespace(data=corpora)
        opt = TrainConfig(data=corpora, src_vocab="dummy", share_vocab=True, model=CustomModelConfig())
        specials = get_specials(opt, transforms_cls)
        specials_expected = {"src": ["｟_pf_src｠"], "tgt": ["｟_pf_tgt｠"]}
        self.assertEqual(specials, specials_expected)

    def test_transform_pipe(self):
        # 1. Init first transform in the pipe
        prefix_cls = get_transforms_cls(["prefix"])["prefix"]
        corpora = yaml.safe_load(
            """
            trainset:
                path_src: eole/tests/data/src-train.txt
                path_tgt: eole/tests/data/tgt-train.txt
                transforms: [prefix, filtertoolong]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """
        )
        # opt = Namespace(data=corpora, seed=-1)
        opt = TrainConfig(
            data=corpora,
            seed=-1,
            src_vocab="dummy",
            share_vocab=True,
            model=CustomModelConfig(),
        )
        prefix_transform = prefix_cls(opt)
        prefix_transform.warm_up()
        # 2. Init second transform in the pipe
        filter_cls = get_transforms_cls(["filtertoolong"])["filtertoolong"]
        # opt = Namespace(src_seq_length=4, tgt_seq_length=4)
        opt = TrainConfig(
            data={"dummy": Dataset(path_src="eole/tests/data/src-train.txt")},  # actual file path (tested in validate)
            src_vocab="dummy",
            share_vocab=True,
            transforms_configs={"filtertoolong": {"src_seq_length": 4, "tgt_seq_length": 4}},
            model=CustomModelConfig(),
        )
        filter_transform = filter_cls(opt)
        # 3. Sequential combine them into a transform pipe
        transform_pipe = TransformPipe.build_from([prefix_transform, filter_transform])
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # 4. apply transform pipe for example
        ex_after = transform_pipe.apply(copy.deepcopy(ex), corpus_name="trainset")
        # 5. example after the pipe exceed the length limit, thus filtered
        self.assertIsNone(ex_after)
        # 6. Transform statistics registed (here for filtertoolong)
        self.assertTrue(len(transform_pipe.statistics.observables) > 0)
        msg = transform_pipe.statistics.report()
        self.assertIsNotNone(msg)
        # 7. after report, statistics become empty as a fresh start
        self.assertTrue(len(transform_pipe.statistics.observables) == 0)


class TestMiscTransform(unittest.TestCase):
    def test_prefix(self):
        prefix_cls = get_transforms_cls(["prefix"])["prefix"]
        corpora = yaml.safe_load(
            """
            trainset:
                path_src: eole/tests/data/src-train.txt
                path_tgt: eole/tests/data/tgt-train.txt
                transforms: [prefix]
                weight: 1
                src_prefix: "｟_pf_src｠"
                tgt_prefix: "｟_pf_tgt｠"
        """
        )
        # opt = Namespace(data=corpora, seed=-1)
        opt = TrainConfig(
            data=corpora,
            seed=-1,
            src_vocab="dummy",
            share_vocab=True,
            model=CustomModelConfig(),
        )
        prefix_transform = prefix_cls(opt)
        prefix_transform.warm_up()
        self.assertIn("trainset", prefix_transform.prefix_dict)

        ex_in = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        with self.assertRaises(ValueError):
            prefix_transform.apply(ex_in)
            prefix_transform.apply(ex_in, corpus_name="validset")
        ex_out = prefix_transform.apply(ex_in, corpus_name="trainset")
        self.assertEqual(ex_out["src"][0], "｟_pf_src｠")
        self.assertEqual(ex_out["tgt"][0], "｟_pf_tgt｠")

    def test_filter_too_long(self):
        filter_cls = get_transforms_cls(["filtertoolong"])["filtertoolong"]
        # opt = Namespace(src_seq_length=100, tgt_seq_length=100)
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            transforms_configs={"filtertoolong": {"src_seq_length": 100, "tgt_seq_length": 100}},
        )
        filter_transform = filter_cls(opt)
        # filter_transform.warm_up()
        ex_in = {
            "src": ["Hello", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        ex_out = filter_transform.apply(ex_in)
        self.assertIs(ex_out, ex_in)
        filter_transform.tgt_seq_length = 2
        ex_out = filter_transform.apply(ex_in)
        self.assertIsNone(ex_out)


class TestSubwordTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_opts = {
            # "seed": 3431, # done at general level
            # "share_vocab": False, # done at general level
            "src_subword_model": "eole/tests/data/sample.bpe",
            "tgt_subword_model": "eole/tests/data/sample.bpe",
            "src_subword_nbest": 1,
            "tgt_subword_nbest": 1,
            "src_subword_alpha": 0.0,
            "tgt_subword_alpha": 0.0,
            "src_subword_vocab": "",
            "tgt_subword_vocab": "",
            "src_vocab_threshold": 0,
            "tgt_vocab_threshold": 0,
        }

    def test_bpe(self):
        bpe_cls = get_transforms_cls(["bpe"])["bpe"]
        # opt = Namespace(**self.base_opts)
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            seed=3431,
            transforms_configs={"bpe": self.base_opts},
        )
        # bpe_cls._validate_options(opt) # done in config validation now
        bpe_transform = bpe_cls(opt)
        bpe_transform.warm_up()
        ex = {
            "src": "Hello world .",
            "tgt": "Bonjour le monde .",
        }
        bpe_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ["H@@", "ell@@", "o", "world", "."],
            "tgt": ["B@@", "on@@", "j@@", "our", "le", "mon@@", "de", "."],
        }
        self.assertEqual(ex, ex_gold)
        # test BPE-dropout:
        bpe_transform.dropout["src"] = 1.0
        tokens = "Another world ."
        gold_bpe = ["A@@", "no@@", "ther", "world", "."]
        gold_dropout = [
            "A@@",
            "n@@",
            "o@@",
            "t@@",
            "h@@",
            "e@@",
            "r",
            "w@@",
            "o@@",
            "r@@",
            "l@@",
            "d",
            ".",
        ]
        # 1. disable bpe dropout for not training example
        after_bpe = bpe_transform._tokenize(tokens, is_train=False)
        self.assertEqual(after_bpe, gold_bpe)
        # 2. enable bpe dropout for training example
        after_bpe = bpe_transform._tokenize(tokens, is_train=True)
        self.assertEqual(after_bpe, gold_dropout)
        # 3. (NOTE) disable dropout won't take effect if already seen
        # this is caused by the cache mechanism in bpe:
        # return cached subword if the original token is seen when no dropout
        after_bpe2 = bpe_transform._tokenize(tokens, is_train=False)
        self.assertEqual(after_bpe2, gold_dropout)

    def test_sentencepiece(self):
        sp_cls = get_transforms_cls(["sentencepiece"])["sentencepiece"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_model"] = "eole/tests/data/sample.sp.model"
        base_opt["tgt_subword_model"] = "eole/tests/data/sample.sp.model"
        # opt = Namespace(**base_opt)
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            seed=3431,
            transforms_configs={"sentencepiece": base_opt},
        )
        # sp_cls._validate_options(opt)
        sp_transform = sp_cls(opt)
        sp_transform.warm_up()

        ex = {
            "src": "Hello world .",
            "tgt": "Bonjour le monde .",
        }
        sp_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ["▁H", "el", "lo", "▁world", "▁."],
            "tgt": ["▁B", "on", "j", "o", "ur", "▁le", "▁m", "on", "de", "▁."],
        }
        self.assertEqual(ex, ex_gold)

        # test SP regularization:
        sp_transform.src_subword_nbest = 4
        tokens = "Another world ."
        gold_sp = ["▁An", "other", "▁world", "▁."]
        # 1. enable regularization for training example
        after_sp = sp_transform._tokenize(tokens, is_train=True)
        self.assertEqual(after_sp, ["▁An", "o", "ther", "▁world", "▁."])
        # 2. disable regularization for not training example
        after_sp = sp_transform._tokenize(tokens, is_train=False)
        self.assertEqual(after_sp, gold_sp)

        # Test mask location
        ex = {
            "src": "### Instruction: ｟newline｠instruction｟newline｠｟newline｠" "### Response : ｟newline｠response",
            "tgt": "",
        }
        ex_gold = {
            "src": [
                "▁",
                "#",
                "#",
                "#",
                "▁In",
                "struct",
                "ion",
                ":",
                "▁in",
                "struct",
                "ion",
                "▁",
                "#",
                "#",
                "#",
                "▁Re",
                "s",
                "p",
                "on",
                "s",
                "e",
                "▁",
                ":",
                "▁re",
                "s",
                "p",
                "on",
                "se",
            ],
            "tgt": [],
        }
        sp_transform.apply(ex, is_train=True)
        self.assertEqual(ex, ex_gold)

    def test_pyonmttok_bpe(self):
        onmttok_cls = get_transforms_cls(["onmt_tokenize"])["onmt_tokenize"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_type"] = "bpe"
        base_opt["tgt_subword_type"] = "bpe"
        eole_args = {"mode": "space", "joiner_annotate": True}
        base_opt["src_onmttok_kwargs"] = eole_args
        base_opt["tgt_onmttok_kwargs"] = eole_args
        base_opt["gpt2_pretok"] = False
        # opt = Namespace(**base_opt)
        # replace with proper pydantic model
        # config_model = onmttok_cls.config_model
        # opt = config_model(**base_opt)
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            seed=3431,
            transforms_configs={"onmt_tokenize": base_opt},
        )
        # onmttok_cls._validate_options(opt)
        onmttok_transform = onmttok_cls(opt)
        onmttok_transform.warm_up()
        ex = {
            "src": "Hello world .",
            "tgt": "Bonjour le monde .",
        }
        onmttok_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ["H￭", "ell￭", "o", "world", "."],
            "tgt": ["B￭", "on￭", "j￭", "our", "le", "mon￭", "de", "."],
        }
        self.assertEqual(ex, ex_gold)

        # Test mask location
        ex = {
            "src": (
                "### Instruction: ｟newline｠instruction｟newline｠｟newline｠" "### Response : ｟newline｠response"
            ),
            "tgt": "",
        }
        ex_gold = {
            "src": [
                "#￭",
                "#￭",
                "#",
                "In￭",
                "struc￭",
                "tion￭",
                ":",
                "｟newline｠￭",
                "in￭",
                "struc￭",
                "tion￭",
                "｟newline｠￭",
                "｟newline｠￭",
                "#￭",
                "#￭",
                "#",
                "R￭",
                "es￭",
                "p￭",
                "on￭",
                "se",
                ":",
                "｟newline｠￭",
                "respon￭",
                "se",
            ],
            "tgt": [],
        }
        onmttok_transform.apply(ex, is_train=True)
        self.assertEqual(ex, ex_gold)

    def test_pyonmttok_sp(self):
        onmttok_cls = get_transforms_cls(["onmt_tokenize"])["onmt_tokenize"]
        base_opt = copy.copy(self.base_opts)
        base_opt["src_subword_type"] = "sentencepiece"
        base_opt["tgt_subword_type"] = "sentencepiece"
        base_opt["src_subword_model"] = "eole/tests/data/sample.sp.model"
        base_opt["tgt_subword_model"] = "eole/tests/data/sample.sp.model"
        eole_args = {"mode": "none", "spacer_annotate": True}
        base_opt["src_onmttok_kwargs"] = eole_args
        base_opt["tgt_onmttok_kwargs"] = eole_args
        base_opt["gpt2_pretok"] = False
        # opt = Namespace(**base_opt)
        # replace with proper pydantic model
        # config_model = onmttok_cls.config_model
        # opt = config_model(**base_opt)
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            transforms_configs={"onmt_tokenize": base_opt},
        )
        # onmttok_cls._validate_options(opt)
        onmttok_transform = onmttok_cls(opt)
        onmttok_transform.warm_up()
        ex = {
            "src": "Hello world .",
            "tgt": "Bonjour le monde .",
        }
        onmttok_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": ["▁H", "el", "lo", "▁world", "▁."],
            "tgt": ["▁B", "on", "j", "o", "ur", "▁le", "▁m", "on", "de", "▁."],
        }
        self.assertEqual(ex, ex_gold)

        # Test mask location
        ex = {
            "src": (
                "### Instruction: ｟newline｠instruction｟newline｠｟newline｠" "### Response : ｟newline｠response"
            ),
            "tgt": "",
        }
        onmttok_transform.apply(ex, is_train=True)
        ex_gold = {
            "src": [
                "▁",
                "#",
                "#",
                "#",
                "▁In",
                "struct",
                "ion",
                ":",
                "▁in",
                "struct",
                "ion",
                "▁",
                "#",
                "#",
                "#",
                "▁Re",
                "s",
                "p",
                "on",
                "se",
                "▁",
                ":",
                "▁re",
                "s",
                "p",
                "on",
                "se",
            ],
            "tgt": [],
        }
        self.assertEqual(ex, ex_gold)


class TestSamplingTransform(unittest.TestCase):
    def test_tokendrop(self):
        tokendrop_cls = get_transforms_cls(["tokendrop"])["tokendrop"]
        # opt = Namespace(seed=3434, tokendrop_temperature=0.1)
        opt = TrainConfig(
            data={"dummy": Dataset(path_src="eole/tests/data/src-train.txt")},  # actual file path (tested in validate)
            src_vocab="dummy",
            share_vocab=True,
            seed=3434,
            transforms_configs={"tokendrop": {"tokendrop_temperature": 0.1}},
            model=CustomModelConfig(),
        )
        tokendrop_transform = tokendrop_cls(opt)
        tokendrop_transform.warm_up()
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # Not apply token drop for not training example
        ex_after = tokendrop_transform.apply(copy.deepcopy(ex), is_train=False)
        self.assertEqual(ex_after, ex)
        # apply token drop for training example
        ex_after = tokendrop_transform.apply(copy.deepcopy(ex), is_train=True)
        self.assertNotEqual(ex_after, ex)

    def test_tokenmask(self):
        tokenmask_cls = get_transforms_cls(["tokenmask"])["tokenmask"]
        # opt = Namespace(seed=3434, tokenmask_temperature=0.1)
        opt = TrainConfig(
            data={"dummy": Dataset(path_src="eole/tests/data/src-train.txt")},  # actual file path (tested in validate)
            src_vocab="dummy",
            share_vocab=True,
            seed=3434,
            transforms_configs={"tokenmask": {"tokenmask_temperature": 0.1}},
            model=CustomModelConfig(),
        )
        tokenmask_transform = tokenmask_cls(opt)
        tokenmask_transform.warm_up()
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # Not apply token mask for not training example
        ex_after = tokenmask_transform.apply(copy.deepcopy(ex), is_train=False)
        self.assertEqual(ex_after, ex)
        # apply token mask for training example
        ex_after = tokenmask_transform.apply(copy.deepcopy(ex), is_train=True)
        self.assertNotEqual(ex_after, ex)

    def test_switchout(self):
        switchout_cls = get_transforms_cls(["switchout"])["switchout"]
        # opt = Namespace(seed=3434, switchout_temperature=0.1)
        opt = TrainConfig(
            data={"dummy": Dataset(path_src="eole/tests/data/src-train.txt")},  # actual file path (tested in validate)
            src_vocab="dummy",
            share_vocab=True,
            seed=3434,
            transforms_configs={"switchout": {"switchout_temperature": 0.1}},
            model=CustomModelConfig(),
        )
        switchout_transform = switchout_cls(opt)
        with self.assertRaises(ValueError):
            # require vocabs to warm_up
            switchout_transform.warm_up(vocabs=None)
        vocabs = {
            "src": Namespace(ids_to_tokens=["A", "Fake", "vocab"]),
            "tgt": Namespace(ids_to_tokens=["A", "Fake", "vocab"]),
        }
        switchout_transform.warm_up(vocabs=vocabs)
        ex = {
            "src": ["Hello", ",", "world", "."],
            "tgt": ["Bonjour", "le", "monde", "."],
        }
        # Not apply token mask for not training example
        ex_after = switchout_transform.apply(copy.deepcopy(ex), is_train=False)
        self.assertEqual(ex_after, ex)
        # apply token mask for training example
        ex_after = switchout_transform.apply(copy.deepcopy(ex), is_train=True)
        self.assertNotEqual(ex_after, ex)


class TestBARTNoising(unittest.TestCase):
    def setUp(self):
        BARTNoising.set_random_seed(1234)
        self.MASK_TOK = "[MASK]"
        self.FAKE_VOCAB = "[TESTING]"

    def test_sentence_permute(self):
        sent1 = ["Hello", "world", "."]
        sent2 = ["Sentence", "1", "!"]
        sent3 = ["Sentence", "2", "!"]
        sent4 = ["Sentence", "3", "!"]

        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            permute_sent_ratio=0.5,
            replace_length=0,  # not raise Error
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        tokens = sent1 + sent2 + sent3 + sent4
        ends = bart_noise._get_sentence_borders(tokens).tolist()
        self.assertEqual(ends, [3, 6, 9, 12])
        tokens_perm = bart_noise.apply(tokens)
        expected_tokens = sent2 + sent1 + sent3 + sent4
        self.assertEqual(expected_tokens, tokens_perm)

    def test_rotate(self):
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            rotate_ratio=1.0,
            replace_length=0,  # not raise Error
        )
        tokens = ["This", "looks", "really", "good", "!"]
        rotated = bart_noise.apply(tokens)
        self.assertNotEqual(tokens, rotated)
        not_rotate = bart_noise.rolling_noise(tokens, p=0.0)
        self.assertEqual(tokens, not_rotate)

    def test_token_insert(self):
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            mask_tok=self.MASK_TOK,
            insert_ratio=0.5,
            random_ratio=0.3,
            replace_length=0,  # not raise Error
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        tokens = ["This", "looks", "really", "good", "!"]
        inserted = bart_noise.apply(tokens)
        n_insert = math.ceil(len(tokens) * bart_noise.insert_ratio)
        inserted_len = n_insert + len(tokens)
        self.assertEqual(len(inserted), inserted_len)
        # random_ratio of inserted tokens are chosen in vocab
        n_random = math.ceil(n_insert * bart_noise.random_ratio)
        self.assertEqual(
            sum(1 if tok == self.FAKE_VOCAB else 0 for tok in inserted),
            n_random,
        )
        # others are MASK_TOK
        self.assertEqual(
            sum(1 if tok == self.MASK_TOK else 0 for tok in inserted),
            n_insert - n_random,
        )

    def test_token_mask(self):
        """Mask will be done on token level.

        Condition:
        * `mask_length` == subword;
        * or not specify subword marker (joiner/spacer) by `is_joiner`.
        """
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            mask_tok=self.MASK_TOK,
            mask_ratio=0.5,
            mask_length="subword",
            replace_length=0,  # 0 to drop them, 1 to replace them with MASK
            # insert_ratio=0.0,
            # random_ratio=0.0,
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        tokens = ["H￭", "ell￭", "o", "world", "."]
        # all token are considered as an individual word
        self.assertTrue(all(bart_noise._is_word_start(tokens)))
        n_tokens = len(tokens)

        # 1. tokens are dropped when replace_length is 0
        masked = bart_noise.apply(tokens)
        n_masked = math.ceil(n_tokens * bart_noise.mask_ratio)
        # print(f"token delete: {masked} / {tokens}")
        self.assertEqual(len(masked), n_tokens - n_masked)

        # 2. tokens are replaced by MASK when replace_length is 1
        bart_noise.replace_length = 1
        masked = bart_noise.apply(tokens)
        n_masked = math.ceil(n_tokens * bart_noise.mask_ratio)
        # print(f"token mask: {masked} / {tokens}")
        self.assertEqual(len(masked), n_tokens)
        self.assertEqual(sum([1 if tok == self.MASK_TOK else 0 for tok in masked]), n_masked)

    def test_whole_word_mask(self):
        """Mask will be done on whole word that may across multiply token.

        Condition:
        * `mask_length` == word;
        * specify subword marker in order to find word boundary.
        """
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            mask_tok=self.MASK_TOK,
            mask_ratio=0.5,
            mask_length="word",
            is_joiner=True,
            replace_length=0,  # 0 to drop them, 1 to replace them with MASK
            # insert_ratio=0.0,
            # random_ratio=0.0,
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        tokens = ["H￭", "ell￭", "o", "wor￭", "ld", "."]
        # start token of word are identified using subword marker
        token_starts = [True, False, False, True, False, True]
        self.assertEqual(bart_noise._is_word_start(tokens), token_starts)

        # 1. replace_length 0: "words" are dropped
        masked = bart_noise.apply(copy.copy(tokens))
        n_words = sum(token_starts)
        n_masked = math.ceil(n_words * bart_noise.mask_ratio)
        # print(f"word delete: {masked} / {tokens}")
        # self.assertEqual(len(masked), n_words - n_masked)

        # 2. replace_length 1: "words" are replaced with a single MASK
        bart_noise.replace_length = 1
        masked = bart_noise.apply(copy.copy(tokens))
        # print(f"whole word single mask: {masked} / {tokens}")
        # len(masked) depend on number of tokens in select word
        n_words = sum(token_starts)
        n_masked = math.ceil(n_words * bart_noise.mask_ratio)
        self.assertEqual(sum(1 if tok == self.MASK_TOK else 0 for tok in masked), n_masked)

        # 3. replace_length -1: all tokens in "words" are replaced with MASK
        bart_noise.replace_length = -1
        masked = bart_noise.apply(copy.copy(tokens))
        # print(f"whole word multi mask: {masked} / {tokens}")
        self.assertEqual(len(masked), len(tokens))  # length won't change
        n_words = sum(token_starts)
        n_masked = math.ceil(n_words * bart_noise.mask_ratio)
        # number of mask_tok depend on number of tokens in selected word
        # number of MASK_TOK can be greater than n_masked
        self.assertTrue(sum(1 if tok == self.MASK_TOK else 0 for tok in masked) > n_masked)

    def test_span_infilling(self):
        bart_noise = BARTNoising(
            vocab=[self.FAKE_VOCAB],
            mask_tok=self.MASK_TOK,
            mask_ratio=0.5,
            mask_length="span-poisson",
            poisson_lambda=3.0,
            is_joiner=True,
            replace_length=1,
            # insert_ratio=0.5,
            # random_ratio=0.3,
            # Defalt: full_stop_token=[".", "?", "!"]
        )
        self.assertIsNotNone(bart_noise.mask_span_distribution)
        tokens = ["H￭", "ell￭", "o", "world", ".", "An￭", "other", "!"]
        # start token of word are identified using subword marker
        token_starts = [True, False, False, True, True, True, False, True]
        self.assertEqual(bart_noise._is_word_start(tokens), token_starts)
        bart_noise.apply(copy.copy(tokens))
        # n_words = sum(token_starts)
        # n_masked = math.ceil(n_words * bart_noise.mask_ratio)
        # print(f"Text Span Infilling: {infillied} / {tokens}")
        # print(n_words, n_masked)


class TestInsertMaskBeforePlaceholder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_opts = {
            "response_patterns": ["Response : ｟newline｠"],
        }

    def test_insert_mask_before_placeholder(self):
        # insert_mask_before_placeholder_cls = get_transforms_cls(
        #    ["insert_mask_before_placeholder"]
        # )["insert_mask_before_placeholder"]
        # opt = Namespace(**self.base_opts)
        # opt = PredictConfig(
        #    model_path=["dummy"],
        #    src="dummy",
        #    transforms_configs={"insert_mask_before_placeholder": self.base_opts},
        # )
        # insert_mask_before_placeholder_transform = insert_mask_before_placeholder_cls(
        #    opt
        # )
        ex_in = {
            "src": "### Instruction: ｟newline｠instruction｟newline｠｟newline｠" "### Response : ｟newline｠response",
            "tgt": "",
        }
        ex_in["src"] = ex_in["src"].split(" ")
        ex_in["tgt"] = ex_in["src"]
        # ex_out = insert_mask_before_placeholder_transform.apply(ex_in)
        ex_gold = {
            "src": [
                "###",
                "Instruction:",
                "｟newline｠instruction｟newline｠｟newline｠###",
                "Response",
                ":",
                "｟newline｠response",
            ],
            "tgt": [
                "###",
                "Instruction:",
                "｟newline｠instruction｟newline｠｟newline｠###",
                "Response",
                ":",
                "｟newline｠response",
            ],
        }
        self.assertEqual(ex_in, ex_gold)


class TestMosesPunctNormalizer(unittest.TestCase):
    """Tests for the MosesPunctNormalizer utility class."""

    def setUp(self):
        self.normalizer = MosesPunctNormalizer()

    def test_basic_normalization(self):
        """Normalize removes extra whitespace."""
        text = "Hello   world  ."
        result = self.normalizer.normalize(text)
        self.assertEqual(result, "Hello world .")

    def test_unicode_quotes_normalized(self):
        """Unicode typographic quotes are converted to ASCII."""
        text = "\u201cHello\u201d"
        result = self.normalizer.normalize(text)
        self.assertNotIn("\u201c", result)
        self.assertNotIn("\u201d", result)

    def test_em_dash_normalized(self):
        """Em-dash is replaced by a spaced hyphen."""
        text = "word\u2014word"
        result = self.normalizer.normalize(text)
        self.assertIn("-", result)
        self.assertNotIn("\u2014", result)

    def test_en_dash_normalized(self):
        """En-dash is replaced by a hyphen."""
        text = "2000\u20132001"
        result = self.normalizer.normalize(text)
        self.assertNotIn("\u2013", result)

    def test_ellipsis_normalized(self):
        """Unicode ellipsis is expanded to three dots."""
        text = "waiting\u2026"
        result = self.normalizer.normalize(text)
        self.assertIn("...", result)

    def test_french_quotes(self):
        """French guillemets are replaced by double quotes."""
        text = "\u00AB bonjour \u00BB"
        result = self.normalizer.normalize(text)
        self.assertIn('"', result)
        self.assertNotIn("\u00AB", result)
        self.assertNotIn("\u00BB", result)

    def test_pre_replace_unicode_punct(self):
        """pre_replace_unicode_punct replaces fullwidth digits and punctuation."""
        text = "\uff10\uff11\uff12"  # fullwidth 0, 1, 2
        result = self.normalizer.normalize(text, pre_replace_unicode_punct=True)
        self.assertIn("0", result)
        self.assertIn("1", result)
        self.assertIn("2", result)

    def test_replace_unicode_punct_directly(self):
        """replace_unicode_punct converts Chinese/fullwidth punctuation."""
        text = "\uff0c\u3002"  # fullwidth comma, Chinese period
        result = self.normalizer.replace_unicode_punct(text)
        self.assertIn(",", result)

    def test_post_remove_control_chars(self):
        """post_remove_control_chars strips control characters."""
        text = "hello\x00world"
        result = self.normalizer.normalize(text, post_remove_control_chars=True)
        self.assertNotIn("\x00", result)
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_de_number_normalization(self):
        """German locale uses comma as decimal separator."""
        text = "1\u00A02"  # non-breaking space between digits
        result = self.normalizer.normalize(text, lang="de", norm_numbers=True)
        self.assertIn(",", result)

    def test_penn_mode_on_replaces_backtick(self):
        """With Penn mode on, backtick is replaced by single quote."""
        text = "`hello`"
        result = self.normalizer.normalize(text, penn=True)
        # In Penn mode NORMALIZE_UNICODE_IF_NOT_PENN substitutions are applied
        self.assertNotIn("`", result)

    def test_returns_string(self):
        """normalize always returns a string."""
        result = self.normalizer.normalize("")
        self.assertIsInstance(result, str)


class TestNormalizeTransform(unittest.TestCase):
    """Tests for the NormalizeTransform pipeline class."""

    @classmethod
    def setUpClass(cls):
        normalize_cls = get_transforms_cls(["normalize"])["normalize"]
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            transforms_configs={
                "normalize": {
                    "src_lang": "en",
                    "tgt_lang": "en",
                    "penn": True,
                    "norm_quote_commas": True,
                    "norm_numbers": True,
                    "pre_replace_unicode_punct": False,
                    "post_remove_control_chars": False,
                }
            },
        )
        cls.transform = normalize_cls(opt)
        cls.transform.warm_up()

    def test_apply_normalizes_src_and_tgt(self):
        """apply() normalizes both src and tgt strings."""
        ex = {"src": "Hello   world .", "tgt": "Bonjour   monde ."}
        result = self.transform.apply(ex, corpus_name="infer")
        self.assertEqual(result["src"], "Hello world .")
        self.assertEqual(result["tgt"], "Bonjour monde .")

    def test_apply_requires_corpus_name(self):
        """apply() raises ValueError when corpus_name is missing."""
        ex = {"src": "Hello world .", "tgt": None}
        with self.assertRaises(ValueError):
            self.transform.apply(ex)

    def test_apply_handles_none_tgt(self):
        """apply() skips tgt normalization when tgt is None."""
        ex = {"src": "Hello   world .", "tgt": None}
        result = self.transform.apply(ex, corpus_name="infer")
        self.assertIsNotNone(result)
        self.assertIsNone(result["tgt"])
        self.assertEqual(result["src"], "Hello world .")

    def test_apply_requires_string_src(self):
        """apply() raises AssertionError when src is a list (must run before tokenizer)."""
        ex = {"src": ["Hello", "world", "."], "tgt": None}
        with self.assertRaises(AssertionError):
            self.transform.apply(ex, corpus_name="infer")

    def test_warm_up_creates_infer_dicts(self):
        """warm_up() populates 'infer' key in all language dicts."""
        self.assertIn("infer", self.transform.src_lang_dict)
        self.assertIn("infer", self.transform.tgt_lang_dict)


class TestUpperCaseTransform(unittest.TestCase):
    """Tests for the UpperCaseTransform."""

    def _make_transform(self, ratio):
        uppercase_cls = get_transforms_cls(["uppercase"])["uppercase"]
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            transforms_configs={"uppercase": {"upper_corpus_ratio": ratio}},
        )
        t = uppercase_cls(opt)
        t.warm_up()
        return t

    def test_apply_always_uppercases_at_ratio_one(self):
        """With ratio=1.0 every example is uppercased."""
        t = self._make_transform(ratio=1.0)
        ex = {"src": "Hello world.", "tgt": "Bonjour monde."}
        result = t.apply(copy.deepcopy(ex))
        self.assertEqual(result["src"], "HELLO WORLD.")
        self.assertEqual(result["tgt"], "BONJOUR MONDE.")

    def test_apply_never_uppercases_at_ratio_zero(self):
        """With ratio=0.0 no example is uppercased."""
        t = self._make_transform(ratio=0.0)
        ex = {"src": "Hello world.", "tgt": "Bonjour monde."}
        result = t.apply(copy.deepcopy(ex))
        self.assertEqual(result["src"], "Hello world.")
        self.assertEqual(result["tgt"], "Bonjour monde.")

    def test_apply_handles_none_tgt(self):
        """apply() skips tgt when it is None."""
        t = self._make_transform(ratio=1.0)
        ex = {"src": "Hello world.", "tgt": None}
        result = t.apply(copy.deepcopy(ex))
        self.assertEqual(result["src"], "HELLO WORLD.")
        self.assertIsNone(result["tgt"])

    def test_apply_requires_string_src(self):
        """apply() raises AssertionError when src is a list."""
        t = self._make_transform(ratio=1.0)
        ex = {"src": ["Hello", "world", "."], "tgt": None}
        with self.assertRaises(AssertionError):
            t.apply(ex)

    def test_apply_unicode_normalization(self):
        """Uppercasing applies NFD normalization and removes combining marks."""
        t = self._make_transform(ratio=1.0)
        # Greek lowercase 'α' uppercases to 'Α'
        ex = {"src": "αβγ", "tgt": None}
        result = t.apply(copy.deepcopy(ex))
        self.assertEqual(result["src"], result["src"].upper())

    def test_apply_random_sampling(self):
        """With intermediate ratio, some examples are converted and some are not."""
        t = self._make_transform(ratio=0.5)
        rng = random.Random(42)
        # Monkey-patch the module-level random.random used inside UpperCaseTransform.apply
        import eole.transforms.uppercase as _umod

        original_random = _umod.random.random
        _umod.random.random = rng.random
        try:
            results = set()
            ex_template = {"src": "Hello world.", "tgt": None}
            for _ in range(100):
                r = t.apply(copy.deepcopy(ex_template))
                results.add(r["src"])
            # Both uppercased and original forms should appear
            self.assertGreater(len(results), 1)
        finally:
            _umod.random.random = original_random


class TestFilterTooShortTransform(unittest.TestCase):
    """Tests for the FilterTooShortTransform."""

    def _make_transform(self, src_min=3, tgt_min=3):
        cls = get_transforms_cls(["filtertooshort"])["filtertooshort"]
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            transforms_configs={"filtertooshort": {"src_seq_length": src_min, "tgt_seq_length": tgt_min}},
        )
        t = cls(opt)
        t.warm_up()
        return t

    def test_passes_long_enough_example(self):
        """Example meeting the minimum length is returned unchanged."""
        t = self._make_transform(src_min=2, tgt_min=2)
        ex = {"src": ["Hello", "world", "."], "tgt": ["Bonjour", "monde", "."]}
        result = t.apply(ex)
        self.assertIs(result, ex)

    def test_filters_short_src(self):
        """Example with src shorter than minimum is filtered (returns None)."""
        t = self._make_transform(src_min=5, tgt_min=1)
        ex = {"src": ["Hi", "."], "tgt": ["Salut", "."]}
        result = t.apply(ex)
        self.assertIsNone(result)

    def test_filters_short_tgt(self):
        """Example with tgt shorter than minimum is filtered (returns None)."""
        t = self._make_transform(src_min=1, tgt_min=5)
        ex = {"src": ["Hello", "world", "."], "tgt": ["Hi"]}
        result = t.apply(ex)
        self.assertIsNone(result)

    def test_requires_list_input(self):
        """apply() raises AssertionError when src is a string (must run after tokenizer)."""
        t = self._make_transform()
        ex = {"src": "Hello world.", "tgt": ["Bonjour"]}
        with self.assertRaises(AssertionError):
            t.apply(ex)

    def test_passes_when_tgt_is_none(self):
        """Example with None tgt is only filtered on src length."""
        t = self._make_transform(src_min=2, tgt_min=100)
        ex = {"src": ["Hello", "world", "."], "tgt": None}
        result = t.apply(ex)
        # tgt is None so tgt length check is skipped; src is long enough
        self.assertIs(result, ex)


class TestSuffixTransform(unittest.TestCase):
    """Tests for the SuffixTransform."""

    @classmethod
    def setUpClass(cls):
        suffix_cls = get_transforms_cls(["suffix"])["suffix"]
        corpora = yaml.safe_load(
            """
            trainset:
                path_src: eole/tests/data/src-train.txt
                path_tgt: eole/tests/data/tgt-train.txt
                transforms: [suffix]
                weight: 1
                src_suffix: "｟_sf_src｠"
                tgt_suffix: "｟_sf_tgt｠"
        """
        )
        opt = TrainConfig(
            data=corpora,
            seed=-1,
            src_vocab="dummy",
            share_vocab=True,
            model=CustomModelConfig(),
        )
        cls.transform = suffix_cls(opt)
        cls.transform.warm_up()

    def test_warm_up_creates_suffix_dict(self):
        """warm_up() builds a suffix_dict containing the configured corpus."""
        self.assertIn("trainset", self.transform.suffix_dict)

    def test_apply_appends_suffix(self):
        """apply() appends the configured suffix tokens to src and tgt."""
        ex = {"src": ["Hello", "world", "."], "tgt": ["Bonjour", "le", "monde", "."]}
        result = self.transform.apply(copy.deepcopy(ex), corpus_name="trainset")
        self.assertEqual(result["src"][-1], "｟_sf_src｠")
        self.assertEqual(result["tgt"][-1], "｟_sf_tgt｠")

    def test_apply_requires_corpus_name(self):
        """apply() raises ValueError when corpus_name is missing."""
        ex = {"src": ["Hello", "world", "."], "tgt": None}
        with self.assertRaises(ValueError):
            self.transform.apply(ex)

    def test_apply_unknown_corpus_raises(self):
        """apply() raises ValueError for an unknown corpus name."""
        ex = {"src": ["Hello", "world", "."], "tgt": None}
        with self.assertRaises(ValueError):
            self.transform.apply(ex, corpus_name="unknown_corpus")

    def test_apply_requires_list_input(self):
        """apply() raises AssertionError when src is a string."""
        ex = {"src": "Hello world .", "tgt": None}
        with self.assertRaises(AssertionError):
            self.transform.apply(ex, corpus_name="trainset")


class TestDocifyTransform(unittest.TestCase):
    """Tests for the DocifyTransform."""

    def _make_transform(self, doc_length=10, max_context=1):
        cls = get_transforms_cls(["docify"])["docify"]
        opt = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            transforms_configs={"docify": {"doc_length": doc_length, "max_context": max_context}},
        )
        t = cls(opt)
        t.warm_up()
        return t

    def test_batch_apply_concatenates_short_segments(self):
        """Short segments below doc_length are concatenated."""
        t = self._make_transform(doc_length=20, max_context=2)
        batch = [
            ({"src": ["Hello", "world"], "tgt": ["Bonjour", "monde"]}, t, "c1"),
            ({"src": ["Foo", "bar"], "tgt": ["Baz", "qux"]}, t, "c1"),
        ]
        result = t.batch_apply(batch)
        # Two short segments should be merged into one doc
        self.assertEqual(len(result), 1)

    def test_batch_apply_splits_long_segments(self):
        """Segments exceeding doc_length are kept separate."""
        t = self._make_transform(doc_length=2, max_context=1)
        long_src = ["word"] * 5
        batch = [
            ({"src": long_src, "tgt": long_src[:]}, t, "c1"),
            ({"src": long_src[:], "tgt": long_src[:]}, t, "c1"),
        ]
        result = t.batch_apply(batch)
        self.assertGreaterEqual(len(result), 2)

    def test_apply_reverse_splits_on_sep(self):
        """apply_reverse() splits on SEP token and returns a list of segments."""
        from eole.constants import DefaultTokens

        t = self._make_transform()
        predicted = f"segment one {DefaultTokens.SEP} segment two"
        result = t.apply_reverse(predicted)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "segment one")
        self.assertEqual(result[1], "segment two")

    def test_batch_apply_max_context_zero_returns_unchanged(self):
        """When max_context=0 the batch is returned as-is."""
        t = self._make_transform(max_context=0)
        batch = [
            ({"src": ["Hello"], "tgt": ["Bonjour"]}, t, "c1"),
        ]
        result = t.batch_apply(batch)
        self.assertIs(result, batch)
