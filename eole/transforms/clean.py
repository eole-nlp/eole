from eole.transforms import register_transform
from .transform import Transform, TransformConfig
from pydantic import Field
from typing import List
from eole.utils.logging import logger
import os
import urllib.request
import regex as re


class CleanConfig(TransformConfig):
    src_eq_tgt: bool | None = Field(default=False, description="Remove ex src==tgt")
    same_char: bool | None = Field(
        default=False, description="Remove ex with same char more than 4 times"
    )
    same_word: bool | None = Field(
        default=False, description="Remove ex with same word more than 3 times"
    )
    scripts_ok: List[str] | None = Field(
        default=["Latin", "Common"],
        description="list of unicodata scripts not accepted",
    )
    src_tgt_ratio: float | None = Field(
        default=2.0, description="ratio between src and tgt"
    )
    avg_tok_min: float | None = Field(
        default=3.0, description="average length of tokens min"
    )
    avg_tok_max: float | None = Field(
        default=20.0, description="average length of tokens max"
    )
    langid: List[str] | None = Field(
        default=[], description="list of languages accepted"
    )


@register_transform(name="clean")
class CleanTransform(Transform):
    """
    Clean examples according to rules
    """

    config_model = CleanConfig

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        self.src_eq_tgt = self.config.src_eq_tgt
        self.same_char = self.config.same_char
        self.same_word = self.config.same_word
        self.scripts_ok = self.config.scripts_ok
        self.scripts_nok = self.config.scripts_nok
        self.src_tgt_ratio = self.config.src_tgt_ratio
        self.avg_tok_min = self.config.avg_tok_min
        self.avg_tok_max = self.config.avg_tok_max
        self.langid = self.config.langid
        assert (
            self.scripts_ok == [] or self.scripts_nok == []
        ), "Choose either scripts to be included or excluded"

    # this whole logic could probably be improved
    @staticmethod
    def _get_param(corpus, param, def_val):
        """Get param string of a `corpus`."""
        if "clean" in corpus["transforms"]:
            value = corpus.get(param, def_val)
            clean = value
        else:
            clean = None
        return clean

    @classmethod
    def get_config_dict(cls, config, param, def_val):
        """Get clean settings correspond to corpus in `opts`."""
        clean_dict = {}
        # normalize dict src/tgt for each dataset
        if hasattr(config, "data"):
            for c_name, corpus in config.data.items():
                clean = cls._get_param(corpus, param, def_val)
                if clean is not None:
                    logger.debug(f"Get {config} for {c_name}: {clean}")
                    clean_dict[c_name] = clean

        return clean_dict

    def warm_up(self, vocabs=None):
        super().warm_up(None)
        import fasttext

        self.src_eq_tgt_dict = self.get_config_dict(self.config, "src_eq_tgt", True)
        self.same_char_dict = self.get_config_dict(self.config, "same_char", True)
        self.same_word_dict = self.get_config_dict(self.config, "same_word", True)
        self.scripts_ok_dict = self.get_config_dict(
            self.config, "scripts_ok", ["Latin", "Common"]
        )
        self.scripts_nok_dict = self.get_config_dict(self.config, "scripts_nok", [])
        self.src_tgt_ratio_dict = self.get_config_dict(self.config, "src_tgt_ratio", 2)
        self.avg_tok_min_dict = self.get_config_dict(self.config, "avg_tok_min", 3)
        self.avg_tok_max_dict = self.get_config_dict(self.config, "avg_tok_max", 20)
        self.langid_dict = self.get_config_dict(self.config, "langid", [])
        fasttext_loc = f"{os.path.dirname(os.path.abspath(__file__))}/lid.176.ftz"

        if not os.path.exists(fasttext_loc):
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/"
                + "fasttext/supervised-models/lid.176.ftz",
                fasttext_loc,
            )
        self.id_func = fasttext.load_model(fasttext_loc)

    def batch_apply(self, batch, is_train=False, stats=None, **kwargs):
        """Convert source and target examples to doc level segments."""

        def _id(string):
            res = self.id_func.predict(string, k=1)
            res = res[0][0].replace("__label__", "")
            return res

        trf_batch = []

        for ex, _, cid in batch:
            if self.scripts_ok_dict[cid]:
                ok_regex = (
                    "[^"
                    + "".join(r"\p{%s}" % sc for sc in self.scripts_ok_dict[cid])
                    + "]"
                )

            if self.scripts_nok_dict[cid]:
                nok_regex = (
                    "["
                    + "".join(r"\p{%s}" % sc for sc in self.scripts_nok_dict[cid])
                    + "]"
                )

            src_str = " ".join(ex["src"])
            if len(src_str) == 0:
                # print("src empty")
                continue
            if self.same_char_dict[cid] and re.search(r"([^0-9])\1{3}", src_str):
                # print("too many same char in src")
                continue
            if self.same_word_dict[cid] and re.search(r"(\ .*|.*\ )\1{2}", src_str):
                # print("too many same word in src")
                continue
            if len(src_str) / len(ex["src"]) < self.avg_tok_min_dict[cid]:
                # print("avg token min", len(src_str) / len(ex['src']))
                continue
            if len(src_str) / len(ex["src"]) > self.avg_tok_max_dict[cid]:
                # print("avg token max", len(src_str) / len(ex['src']))
                continue

            if self.scripts_ok_dict[cid] and re.search(ok_regex, src_str):
                # print("text does not fully belong to wanted script")
                continue
            if self.scripts_nok_dict[cid] and re.search(nok_regex, src_str):
                # print("Some text belong to unwanted scripts")
                continue

            if (
                self.langid_dict[cid] != []
                and _id(src_str) not in self.langid_dict[cid]
            ):
                # print("langid does not match", _id(src_str))
                continue

            if ex["tgt"] is not None:
                tgt_str = " ".join(ex["tgt"])
                if self.src_eq_tgt_dict[cid] and src_str == tgt_str:
                    # print("src = tgt")
                    continue
                if len(tgt_str) == 0:
                    # print("tgt empty")
                    continue
                if (len(ex["src"]) + 1) / (
                    len(ex["tgt"]) + 1
                ) > self.src_tgt_ratio_dict[cid] or (len(ex["src"]) + 1) / (
                    len(ex["tgt"]) + 1
                ) < (
                    1 / self.src_tgt_ratio_dict[cid]
                ):
                    # print("src / tgt ratio ", len(src_str) / len(tgt_str))
                    continue
                if self.same_char_dict[cid] and re.search(r"([^0-9])\1{3}", tgt_str):
                    # print("too many same char in tgt")
                    continue
                if self.same_word_dict[cid] and re.search(r"(\ .*|.*\ )\1{2}", tgt_str):
                    # print("too many same word in tgt")
                    continue
                if len(tgt_str) / len(ex["tgt"]) < self.avg_tok_min_dict[cid]:
                    # print("avg token min", len(tgt_str) / len(ex['tgt']))
                    continue
                if len(tgt_str) / len(ex["tgt"]) > self.avg_tok_max_dict[cid]:
                    # print("avg token max", len(tgt_str) / len(ex['tgt']))
                    continue

                if self.scripts_ok_dict[cid] and re.search(ok_regex, tgt_str):
                    # print("text does not fully belong to wanted script")
                    continue
                if self.scripts_nok_dict[cid] and re.search(nok_regex, tgt_str):
                    # print("Some text belong to unwanted scripts")
                    continue

                if (
                    self.langid_dict[cid] != []
                    and _id(tgt_str) not in self.langid_dict[cid]
                ):
                    # print("langid does not match", _id(tgt_str))
                    continue

            trf_batch.append((ex, self, cid))

        return trf_batch
