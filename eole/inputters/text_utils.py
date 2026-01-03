import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

# import torch.nn.functional as F
# import math
from eole.constants import DefaultTokens, CorpusTask, ModelType
from torch.nn.utils.rnn import pad_sequence
from eole.utils.logging import logger


def text_sort_key(example):
    """
    Sort key based on sequence lengths.

    Returns tuple of (src_len, tgt_len) if target exists, otherwise just src_len.
    This allows sorting by source length primarily, then target length.
    """
    src_len = len(example["src"]["src_ids"])
    if example.get("tgt") is not None:
        tgt_len = len(example["tgt"]["tgt_ids"])
        return (src_len, tgt_len)
    return src_len


def clean_example(example):
    """
    Normalize example structure to ensure a consistent dict-based format.
    Wraps ``src`` and, if present, ``tgt`` values in a dict structure,
    converts an ``align`` list (if any) to a space-separated string, and
    ensures a ``sco`` field exists with a default value.
    Args:
        example: Raw example dict
    Returns:
        Cleaned example dict with standardized structure.
    """

    example["src"] = {"src": example["src"]}
    if example.get("tgt") is not None:
        example["tgt"] = {"tgt": example["tgt"]}
    if "align" in example:
        example["align"] = " ".join(example["align"])
    if "sco" not in example:
        example["sco"] = 1

    return example


def transform_bucket(task, bucket, threshold=0):
    """
    Apply transforms to bucket and filter by score threshold.

    Groups examples by (transform, corpus_id), applies batch transforms,
    then filters by score threshold and empty source.

    Args:
        task: CorpusTask enum value
        bucket: List of (example, transform, cid) tuples
        threshold: Minimum score threshold (default: 0)

    Returns:
        List of transformed and cleaned examples, or None if empty
    """
    # Group examples by their transform and corpus ID
    transform_groups = {}
    for example_tuple in bucket:
        example, transform, cid = example_tuple
        key = (transform, cid)
        if key not in transform_groups:
            transform_groups[key] = []
        transform_groups[key].append(example_tuple)

    # Apply transforms and collect results
    transformed_examples = []
    is_train = task == CorpusTask.TRAIN

    for (transform, cid), examples_group in transform_groups.items():
        # Apply batch transform
        transformed_group = transform.batch_apply(examples_group, is_train=is_train, corpus_name=cid)

        # Clean and filter each example
        for example, _, _ in transformed_group:
            example = clean_example(example)

            # Filter by non-empty source and score threshold
            if len(example["src"]["src"]) > 0 and example.get("sco", 1) >= threshold:
                transformed_examples.append(example)

    return transformed_examples if transformed_examples else None


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
        src = example["src"]["src"]
        if isinstance(src, list):
            return self.vocabs["src"](src)
        # src is a space-separated string (no tokenizer transform applied)
        return self.vocabs["src"](src.split(" "))

    def _get_tgt_ids(self, example: Dict) -> List[int]:
        """Extract or generate target token IDs"""
        if self.is_hf_tokenized:
            return example["tgt_ids"]
        tgt = example["tgt"]["tgt"]
        if isinstance(tgt, list):
            return self.vocabs["tgt"](tgt)
        return self.vocabs["tgt"](tgt.split(" "))

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
    Parse Pharaoh-format alignment string into list of [src_idx, tgt_idx] pairs.

    Args:
        align_pharaoh: Space-separated alignment string (e.g., "0-0 1-2 2-1")

    Returns:
        List of [src_idx, tgt_idx] integer pairs

    Raises:
        ValueError: If alignment format is invalid
    """
    align_pairs = align_pharaoh.strip().split(" ")
    parsed_alignments = []

    for align_pair in align_pairs:
        try:
            src_idx, tgt_idx = align_pair.split("-")
            parsed_alignments.append([int(src_idx), int(tgt_idx)])
        except ValueError:
            logger.warning(f"Invalid alignment pair '{align_pair}' in '{align_pharaoh}'")
            logger.warning("Invalid alignment line found. Please check the file!")
            raise

    return parsed_alignments


def _create_padded_tensor(sequences, pad_idx, device, left_pad=False):
    """
    Helper to create padded tensor from list of sequences.

    Args:
        sequences: List of token ID lists
        pad_idx: Padding token index
        device: Target device
        left_pad: Whether to pad on the left side

    Returns:
        Padded tensor of shape [batch_size, max_seq_len]
    """
    # Convert to tensors, flip if left padding
    if left_pad:
        tensors = [torch.tensor(seq, dtype=torch.long, device=device).flip(dims=[0]) for seq in sequences]
    else:
        tensors = [torch.tensor(seq, dtype=torch.long, device=device) for seq in sequences]

    # Pad sequences
    padded = pad_sequence(tensors, batch_first=True, padding_value=pad_idx)

    # Flip back if left padding
    if left_pad:
        padded = padded.flip(dims=[1])

    return padded


def tensorify(vocabs, minibatch, device, left_pad=False):
    """
    Transform a batch of examples into tensors.

    Args:
        vocabs: Vocabulary dictionaries
        minibatch: List of (example, index) tuples
        device: Target device for tensors
        left_pad: Whether to pad sequences on the left

    Returns:
        Dictionary of batch tensors with keys:
            - src: [batch_size, max_src_len] source sequences
            - srclen: [batch_size] source lengths
            - prefix_len: [batch_size] prefix lengths (if applicable)
            - tgt: [batch_size, max_tgt_len] target sequences (if present)
            - tgtlen: [batch_size] target lengths (if present)
            - align: sparse alignment tensor (if present)
            - src_map: [batch_size, max_src_len, vocab_size] (if present)
            - alignment: [batch_size, max_tgt_len] (if present)
            - images: list of image tensors (if present)
            - ind_in_bucket: list of indices
            - cid: list of corpus IDs
            - cid_line_number: list of line numbers
            - sco: [batch_size] scores (if not inference)
            - left_pad: boolean flag
    """
    tensor_batch = {}
    examples = [ex for ex, _ in minibatch]
    indices = [idx for _, idx in minibatch]

    # Process source sequences
    pad_token = vocabs["specials"].get("pad_token", DefaultTokens.PAD)
    src_pad_idx = vocabs["src"][pad_token]
    src_sequences = [ex["src"]["src_ids"] for ex in examples]

    tensor_batch["src"] = _create_padded_tensor(src_sequences, src_pad_idx, device, left_pad)
    tensor_batch["srclen"] = torch.tensor(
        [len(seq) for seq in src_sequences],
        dtype=torch.long,
        device=device,
    )

    # Handle prefix lengths for decoder-only models
    if "prefix_len" in examples[0]["src"]:
        prefix_lens = torch.tensor(
            [ex["src"]["prefix_len"] for ex in examples],
            dtype=torch.long,
            device=device,
        )
        # Adjust for left padding
        if left_pad:
            num_pads = tensor_batch["src"].eq(src_pad_idx).sum(dim=1)
            prefix_lens += num_pads
        tensor_batch["prefix_len"] = prefix_lens
    else:
        tensor_batch["prefix_len"] = None

    # Process target sequences if present
    if examples[0].get("tgt") is not None:
        tgt_pad_idx = vocabs["tgt"][pad_token]
        tgt_sequences = [ex["tgt"]["tgt_ids"] for ex in examples]

        tensor_batch["tgt"] = _create_padded_tensor(tgt_sequences, tgt_pad_idx, device, left_pad)
        tensor_batch["tgtlen"] = torch.tensor(
            [len(seq) for seq in tgt_sequences],
            dtype=torch.long,
            device=device,
        )

    # Process alignments if present
    if "align" in examples[0] and examples[0]["align"] is not None:
        sparse_indices = []
        for batch_idx, ex in enumerate(examples):
            for src_idx, tgt_idx in parse_align_idx(ex["align"]):
                # Store as [batch_idx, tgt_idx+1, src_idx]
                sparse_indices.append([batch_idx, tgt_idx + 1, src_idx])

        tensor_batch["align"] = torch.tensor(sparse_indices, dtype=torch.long, device=device)

    # Process source vocabulary mapping if present
    if "src_map" in examples[0]:
        max_vocab_idx = max(max(ex["src_map"]) for ex in examples)
        src_vocab_size = max_vocab_idx + 1

        src_map = torch.zeros(
            len(examples),
            tensor_batch["src"].size(1),
            src_vocab_size,
            device=device,
        )
        for i, ex in enumerate(examples):
            for j, token_idx in enumerate(ex["src_map"]):
                src_map[i, j, token_idx] = 1

        tensor_batch["src_map"] = src_map

    # Process word alignment if present
    if "alignment" in examples[0]:
        alignment = torch.zeros(
            len(examples),
            tensor_batch["tgt"].size(1),
            dtype=torch.long,
            device=device,
        )
        for i, ex in enumerate(examples):
            align_len = len(ex["alignment"])
            alignment[i, :align_len] = torch.tensor(ex["alignment"], dtype=torch.long, device=device)

        tensor_batch["alignment"] = alignment

    # Process images if present
    if "images" in examples[0]:
        # Flatten all images from all examples
        # Note: Currently supports batch_size=1 for images
        tensor_batch["images"] = [
            torch.tensor(img_data, device=device, dtype=torch.float32)
            for ex in examples
            for img_data in ex["images"].values()
        ]
    else:
        tensor_batch["images"] = None

    # Add metadata
    tensor_batch["ind_in_bucket"] = indices
    tensor_batch["cid"] = [ex["cid"] for ex in examples]
    tensor_batch["cid_line_number"] = [ex["cid_line_number"] for ex in examples]

    # Add scores if not inference
    if examples[0]["cid"] != "infer":
        tensor_batch["sco"] = torch.tensor([ex["sco"] for ex in examples], device=device)

    tensor_batch["left_pad"] = left_pad

    return tensor_batch
