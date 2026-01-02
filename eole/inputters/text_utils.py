import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

# import torch.nn.functional as F
# import math
from eole.constants import DefaultTokens, CorpusTask, ModelType
from torch.nn.utils.rnn import pad_sequence
from eole.utils.logging import logger


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if ex.get("tgt", None) is not None:
        return len(ex["src"]["src_ids"]), len(ex["tgt"]["tgt_ids"])
    return len(ex["src"]["src_ids"])


def clean_example(maybe_example):
    if isinstance(maybe_example["src"], list):
        maybe_example["src"] = {"src": " ".join(maybe_example["src"])}
    else:
        maybe_example["src"] = {"src": maybe_example["src"]}
    if maybe_example.get("tgt", None) is not None:
        maybe_example["tgt"] = {"tgt": " ".join(maybe_example["tgt"])}
    if "align" in maybe_example:
        maybe_example["align"] = " ".join(maybe_example["align"])
    if "sco" not in maybe_example:
        maybe_example["sco"] = 1
    return maybe_example


def transform_bucket(task, bucket, threshold=0):
    """Returns valid transformed bucket from bucket."""
    transform_cid_to_examples = {}
    for example in bucket:
        transform_cid = (example[1], example[2])
        if transform_cid not in transform_cid_to_examples:
            transform_cid_to_examples[transform_cid] = []
        transform_cid_to_examples[transform_cid].append(example)

    transformed_bucket = []
    # careful below it will return a bucket sorted by corpora
    # but we sort by length later and shuffle batches
    for (transform, cid), sub_bucket in transform_cid_to_examples.items():
        transf_bucket = transform.batch_apply(sub_bucket, is_train=(task == CorpusTask.TRAIN), corpus_name=cid)
        for example, transform, cid in transf_bucket:
            example = clean_example(example)
            if len(example["src"]["src"]) > 0 and example.get("sco", 1) >= threshold:
                transformed_bucket.append(example)

        # at this point an example looks like:
        # {'src': {'src': ..., 'feats': [....]},
        #  'tgt': {'tgt': ...},
        #  'src_original': ['tok1', ...'tokn'],
        #  'tgt_original': ['tok1', ...'tokm'],
        #  'cid': corpus id
        #  'cid_line_number' : cid line number
        #  'align': ...,
        #  if tokenize_id is used then it will include:
        #  'src_ids' and 'tgt_ids'
        # }
    if len(transformed_bucket) > 0:
        return transformed_bucket
    else:
        return None


@dataclass
class TokenIds:
    """Helper class to manage special token IDs"""

    decoder_start: Optional[int]
    eos: Optional[int]
    bos: Optional[int]

    @classmethod
    def from_vocabs(cls, vocabs: Dict, model_type=ModelType.ENCODER_DECODER) -> "TokenIds":
        """Extract token IDs from vocabulary"""
        decoder_start_token = vocabs["decoder_start_token"]
        eos_token = vocabs["specials"].get("eos_token", "")
        bos_token = vocabs["specials"].get("bos_token", "")

        side = "tgt" if model_type == ModelType.ENCODER_DECODER else "src"
        return cls(
            decoder_start=vocabs[side].tokens_to_ids.get(decoder_start_token) if decoder_start_token else None,
            eos=vocabs["src"].tokens_to_ids.get(eos_token) if eos_token else None,
            bos=vocabs["src"].tokens_to_ids.get(bos_token) if bos_token else None,
        )


class Numericalizer:
    """Handles conversion of text tokens to numeric IDs for different model architectures"""

    def __init__(self, vocabs, model_type=ModelType.ENCODER_DECODER, task=CorpusTask.INFER):
        self.vocabs = vocabs
        self.token_ids = TokenIds.from_vocabs(vocabs, model_type)
        self.is_hf_tokenized = False
        self.model_type = model_type
        self.task = task
        if self.model_type == ModelType.ENCODER_DECODER:
            self._handle = self._handle_encoder_decoder
        elif self.model_type == ModelType.DECODER:
            self._handle = self._handle_decoder
        elif self.model_type == ModelType.ENCODER:
            self._handle = self._handle_encoder
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def __call__(self, example: Dict) -> Dict:
        """
        Numericalize the (src / tgt) to (src_ids / tgt_ids)
        """
        self.is_hf_tokenized = "src_ids" in example

        example["src"]["src_ids"] = self._get_src_ids(example)

        example = self._handle(example)

        return example

    def _get_src_ids(self, example: Dict) -> List[int]:
        """Extract or generate source token IDs"""
        if self.is_hf_tokenized:
            return example["src_ids"]
        src_text = example["src"]["src"].split(" ")
        return self.vocabs["src"](src_text)

    def _get_tgt_ids(self, example: Dict) -> List[int]:
        """Extract or generate target token IDs"""
        if self.is_hf_tokenized:
            return example["tgt_ids"]

        tgt_text = example["tgt"]["tgt"].split(" ")
        return self.vocabs["tgt"](tgt_text)

    def _add_decoder_start_token(self, ids: List[int]) -> List[int]:
        """Prepend decoder start token if needed"""
        if self.token_ids.decoder_start is None:
            return ids
        if ids and ids[0] == self.token_ids.decoder_start:
            return ids
        return [self.token_ids.decoder_start] + ids

    def _add_eos_token(self, ids: List[int]) -> List[int]:
        """Append EOS token if needed"""
        if self.token_ids.eos is None:
            return ids
        if ids and ids[-1] == self.token_ids.eos:
            return ids
        return ids + [self.token_ids.eos]

    def _create_lm_target(self, src_ids: List[int]) -> List[int]:
        """Create language model target by shifting source"""
        shifted = src_ids[1:]
        return self._add_eos_token(shifted)

    def _handle_encoder_decoder(self, example: Dict) -> Dict:
        """
        Handle encoder-decoder models (e.g., standard NMT).

        Training format:
            src: stok1 stok2 stok3
            tgt: <decoder_start> ttok1 ttok2 ttok3 <eos>
        """
        has_target = example.get("tgt") is not None

        if not has_target:
            if self.task in [CorpusTask.TRAIN, CorpusTask.VALID]:
                # Language model: use shifted source as target
                example["tgt"] = {"tgt_ids": self._create_lm_target(example["src"]["src_ids"])}
            # else: inference with no target - leave tgt as None
        else:
            # NMT or prefix LM with explicit target
            tgt_ids = self._get_tgt_ids(example)
            tgt_ids = self._add_decoder_start_token(tgt_ids)
            tgt_ids = self._add_eos_token(tgt_ids)
            example["tgt"] = {"tgt_ids": tgt_ids}

        return example

    def _handle_decoder(self, example: Dict) -> Dict:
        """
        Handle decoder-only models (e.g., GPT-style).

        Training format:
            src: <decoder_start> stok1 stok2 stok3
            tgt: stok1 stok2 stok3 <eos>
        """
        # Add decoder start token to source
        example["src"]["src_ids"] = self._add_decoder_start_token(example["src"]["src_ids"])
        example["src"]["prefix_len"] = len(example["src"]["src_ids"])

        has_target = example.get("tgt") is not None

        if not has_target:
            if self.task in [CorpusTask.TRAIN, CorpusTask.VALID]:
                # Language model: use shifted source as target
                example["tgt"] = {"tgt_ids": self._create_lm_target(example["src"]["src_ids"])}
            # else: inference with no target - leave tgt as None
        else:
            # Prompt-response format: concatenate prompt and answer
            tgt_ids = self._get_tgt_ids(example)
            tgt_ids = self._add_eos_token(tgt_ids)

            # Concatenate prompt (src) and response (tgt)
            full_sequence = example["src"]["src_ids"] + tgt_ids

            # For decoder-only: input is full_seq[:-1], target is full_seq[1:]
            example["src"]["src_ids"] = full_sequence[:-1]
            example["tgt"] = {"tgt_ids": full_sequence[1:]}

        return example

    def _add_bos_token(self, ids: List[int]) -> List[int]:
        """Prepend BOS token if needed"""
        if self.token_ids.bos is None:
            return ids
        if ids and ids[0] == self.token_ids.bos:
            return ids
        return [self.token_ids.bos] + ids

    def _handle_encoder(self, example: Dict) -> Dict:
        """
        Handle encoder-only models (e.g., BERT-style).

        Format: <bos> [tgt <eos> <eos>] src <eos>
        Where tgt section is optional.
          if there, then COMET format for training comet-like encoder
          if not, regular encoder training
        """
        has_target = example.get("tgt") is not None

        if has_target:
            # Get tokenized target
            tgt_ids = self._get_tgt_ids(example)
            # Add double EOS after target
            tgt_ids = self._add_eos_token(tgt_ids)
            tgt_ids = tgt_ids + [self.token_ids.eos] if self.token_ids.eos is not None else tgt_ids

            # Concatenate: tgt + src
            combined_ids = tgt_ids + example["src"]["src_ids"]

            # Add BOS at start and EOS at end
            combined_ids = self._add_bos_token(combined_ids)
            combined_ids = self._add_eos_token(combined_ids)

            example["src"]["src_ids"] = combined_ids
            example["tgt"]["tgt_ids"] = combined_ids[:]
        else:
            # Just source: <bos> src <eos>
            example["src"]["src_ids"] = self._add_bos_token(example["src"]["src_ids"])
            example["src"]["src_ids"] = self._add_eos_token(example["src"]["src_ids"])

        return example


def parse_align_idx(align_pharaoh):
    """
    Parse Pharaoh alignment into [[<src>, <tgt>], ...]
    """
    align_list = align_pharaoh.strip().split(" ")
    flatten_align_idx = []
    for align in align_list:
        try:
            src_idx, tgt_idx = align.split("-")
        except ValueError:
            logger.warning("{} in `{}`".format(align, align_pharaoh))
            logger.warning("Bad alignement line exists. Please check file!")
            raise
        flatten_align_idx.append([int(src_idx), int(tgt_idx)])
    return flatten_align_idx


def tensorify(vocabs, minibatch, device, left_pad=False):
    """
    This function transforms a batch of example in tensors
    Each example looks like
    {'src': {'src': ..., 'feats': [...], 'src_ids': ...},
     'tgt': {'tgt': ..., 'tgt_ids': ...},
     'src_original': ['tok1', ...'tokn'],
     'tgt_original': ['tok1', ...'tokm'],
     'cid': corpus id
     'cid_line_number' : corpus id line number
     'ind_in_bucket': index in bucket
     'align': ...,
    }
    Returns  Dict of batch Tensors
        {'src': [seqlen, batchsize, n_feats+1],
         'tgt' : [seqlen, batchsize, n_feats=1],
         'cid': [batchsize],
         'cid_line_number' : [batchsize],
         'ind_in_bucket': [batchsize],
         'srclen': [batchsize],
         'tgtlen': [batchsize],
         'align': alignment sparse tensor
        }
    """
    tensor_batch = {}
    if left_pad:
        tbatchsrc = [
            torch.tensor(ex["src"]["src_ids"], dtype=torch.long, device=device).flip(dims=[0])
            for ex, indice in minibatch
        ]
    else:
        tbatchsrc = [torch.tensor(ex["src"]["src_ids"], dtype=torch.long, device=device) for ex, indice in minibatch]
    padidx = vocabs["src"][vocabs["specials"].get("pad_token", DefaultTokens.PAD)]
    tbatchsrc = pad_sequence(tbatchsrc, batch_first=True, padding_value=padidx)
    """
    This removes some recompiles in torch.dynamo, but slows down and make inference tricky
    tbatchsrc = F.pad(
        tbatchsrc,
        (0, max(0, math.ceil(tbatchsrc.size(1) / 8) * 8 - tbatchsrc.size(1))),
        value=padidx,
    )
    """
    if left_pad:
        tensor_batch["src"] = tbatchsrc.flip(dims=[1])
    else:
        tensor_batch["src"] = tbatchsrc

    tensor_batch["srclen"] = torch.tensor(
        [len(ex["src"]["src_ids"]) for ex, indice in minibatch],
        dtype=torch.long,
        device=device,
    )
    if "prefix_len" in minibatch[0][0]["src"].keys():
        tensor_batch["prefix_len"] = torch.tensor(
            [ex["src"]["prefix_len"] for ex, indice in minibatch],
            dtype=torch.long,
            device=device,
        )
        if left_pad:
            tensor_batch["prefix_len"] += tbatchsrc.eq(padidx).sum(dim=1)
    else:
        tensor_batch["prefix_len"] = None

    if minibatch[0][0].get("tgt", None) is not None:
        if left_pad:
            tbatchtgt = [
                torch.tensor(ex["tgt"]["tgt_ids"], dtype=torch.long, device=device).flip(dims=[0])
                for ex, indice in minibatch
            ]
        else:
            tbatchtgt = [
                torch.tensor(ex["tgt"]["tgt_ids"], dtype=torch.long, device=device) for ex, indice in minibatch
            ]

        padidx = vocabs["tgt"][vocabs["specials"].get("pad_token", DefaultTokens.PAD)]
        tbatchtgt = pad_sequence(tbatchtgt, batch_first=True, padding_value=padidx)

        tbatchtgtlen = torch.tensor(
            [len(ex["tgt"]["tgt_ids"]) for ex, indice in minibatch],
            dtype=torch.long,
            device=device,
        )
        if left_pad:
            tensor_batch["tgt"] = tbatchtgt.flip(dims=[1])
        else:
            tensor_batch["tgt"] = tbatchtgt
        tensor_batch["tgtlen"] = tbatchtgtlen

    if "align" in minibatch[0][0].keys() and minibatch[0][0]["align"] is not None:
        sparse_idx = []
        for i, (ex, indice) in enumerate(minibatch):
            for src, tgt in parse_align_idx(ex["align"]):
                sparse_idx.append([i, tgt + 1, src])
        tbatchalign = torch.tensor(sparse_idx, dtype=torch.long, device=device)
        tensor_batch["align"] = tbatchalign

    if "src_map" in minibatch[0][0].keys():
        src_vocab_size = max([max(ex["src_map"]) for ex, indice in minibatch]) + 1
        src_map = torch.zeros(
            len(tensor_batch["srclen"]),
            tbatchsrc.size(1),
            src_vocab_size,
            device=device,
        )
        for i, (ex, indice) in enumerate(minibatch):
            for j, t in enumerate(ex["src_map"]):
                src_map[i, j, t] = 1
        tensor_batch["src_map"] = src_map

    if "alignment" in minibatch[0][0].keys():
        alignment = torch.zeros(
            len(tensor_batch["srclen"]),
            tbatchtgt.size(1),
            dtype=torch.long,
            device=device,
        )
        for i, (ex, indice) in enumerate(minibatch):
            alignment[i, : len(ex["alignment"])] = torch.tensor(ex["alignment"], dtype=torch.long, device=device)
        tensor_batch["alignment"] = alignment

    if "images" in minibatch[0][0].keys():
        tensor_batch["images"] = [
            torch.tensor(v, device=device, dtype=torch.float32)
            for ex, indice in minibatch
            for k, v in ex["images"].items()
            # BATCH > 1 not supported yet
            # [
            #     torch.tensor(v, device=device, dtype=torch.float32)
            #     for k, v in ex["images"].items()
            # ]
            # for ex, indice in minibatch
        ]
    else:
        tensor_batch["images"] = None

    tensor_batch["ind_in_bucket"] = [indice for ex, indice in minibatch]

    tensor_batch["cid"] = [ex["cid"] for ex, indice in minibatch]
    tensor_batch["cid_line_number"] = [ex["cid_line_number"] for ex, indice in minibatch]

    if minibatch[0][0]["cid"] != "infer":
        tensor_batch["sco"] = torch.tensor([ex["sco"] for ex, indice in minibatch], device=device)

    tensor_batch["left_pad"] = left_pad

    return tensor_batch
