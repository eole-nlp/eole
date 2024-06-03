from eole.constants import DefaultTokens
from eole.transforms import register_transform
from .transform import Transform, TransformConfig
from pydantic import Field
import copy


class DocifyConfig(TransformConfig):
    doc_length: int | None = Field(default=200, description="Number of tokens per doc.")
    max_context: int | None = Field(default=1, description="Max context segments.")


@register_transform(name="docify")
class DocifyTransform(Transform):
    """
    Convert source and target examples to doc level segments.

    It concatenates segments with a DefaultTokens.SEP
    until it reaches --doc_length tokens

    """

    config_model = DocifyConfig

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        if hasattr(self.config, "num_workers") and hasattr(self.config, "world_size"):
            self.stride = self.config.num_workers * self.config.world_size
        else:
            self.stride = 1
        self.doc_length = self.config.doc_length
        self.max_context = self.config.max_context

    @classmethod
    def get_specials(cls, config):
        """Add newline tag to src and tgt vocabs."""

        src_specials, tgt_specials = [DefaultTokens.SEP], [DefaultTokens.SEP]
        return (src_specials, tgt_specials)

    def warm_up(self, vocabs=None):
        super().warm_up(None)
        if self.stride != 1:
            assert (
                self.stride % (self.max_context + 1) == 0
            ), "(max_context+1) must be a multiple \
                 of num_workers * world_size"

    def batch_apply(self, batch, is_train=False, stats=None, **kwargs):
        """Convert source and target examples to doc level segments."""
        if self.max_context == 0:
            return batch
        trf_batch = []
        doc = {}
        doc["src"] = []
        doc["tgt"] = []
        doc["cid"] = ""
        doc["cid_line_number"] = 0

        for ex, _, cid in batch:
            if ex["tgt"] is not None:
                cur_len = max(len(doc["src"] + ex["src"]), len(doc["tgt"] + ex["tgt"]))

                if len(ex["src"]) == 0 and len(ex["tgt"]) == 0:
                    # doc break we add it, restart new doc
                    trf_batch.append((doc, self, cid))
                    doc = {}
                    doc["src"] = []
                    doc["tgt"] = []
                    doc["cid"] = ex["cid"]
                    doc["cid_line_number"] = ex["cid_line_number"]
                elif cur_len > self.doc_length:
                    if len(doc["src"]) == 0:
                        # case 1st ex is already longer
                        trf_batch.append((ex, self, cid))
                    else:
                        # adding cur ex is too long we add cur doc
                        # and reset doc to cur ex
                        trf_batch.append((doc, self, cid))
                        doc = copy.deepcopy(ex)
                else:
                    if len(doc["src"]) == 0:
                        # we start the new doc with cur ex
                        doc = copy.deepcopy(ex)
                    else:
                        # we cumulate cur ex to cur doc
                        doc["src"] += [DefaultTokens.SEP] + ex["src"]
                        doc["src_original"] += [DefaultTokens.SEP] + ex["src_original"]
                        doc["tgt"] += [DefaultTokens.SEP] + ex["tgt"]
                        doc["tgt_original"] += [DefaultTokens.SEP] + ex["tgt_original"]
                        nb_ctx = doc["src"].count(DefaultTokens.SEP)
                        if nb_ctx >= self.max_context:
                            trf_batch.append((doc, self, cid))
                            doc = {}
                            doc["src"] = []
                            doc["tgt"] = []
                            doc["cid"] = ex["cid"]
                            doc["cid_line_number"] = ex["cid_line_number"]
            else:
                cur_len = len(doc["src"] + ex["src"])
                doc["tgt"] = None
                if len(ex["src"]) == 0:
                    trf_batch.append((doc, self, cid))
                    doc = {}
                    doc["src"] = []
                    doc["indices"] = ex["indices"]
                elif cur_len > self.doc_length:
                    if len(doc["src"]) == 0:
                        trf_batch.append((ex, self, cid))
                    else:
                        trf_batch.append((doc, self, cid))
                        doc = copy.deepcopy(ex)
                else:
                    if len(doc["src"]) == 0:
                        doc = copy.deepcopy(ex)
                    else:
                        doc["src"] += [DefaultTokens.SEP] + ex["src"]
                        doc["src_original"] += [DefaultTokens.SEP] + ex["src_original"]
                        nb_ctx = doc["src"].count(DefaultTokens.SEP)
                        if nb_ctx >= self.max_context:
                            trf_batch.append((doc, self, cid))
                            doc = {}
                            doc["src"] = []
                            doc["cid"] = ex["cid"]
                            doc["cid_line_number"] = ex["cid_line_number"]
        if len(doc["src"]) > 0:
            trf_batch.append((doc, self, cid))
        return trf_batch

    def apply_reverse(self, predicted):
        segments = predicted.split(DefaultTokens.SEP)
        segments = [segment.strip(" ") for segment in segments]
        return segments
