import torch
from eole.constants import DefaultTokens, CorpusTask, ModelType
from torch.nn.utils.rnn import pad_sequence
from eole.utils.logging import logger


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if ex["tgt"]:
        return len(ex["src"]["src_ids"]), len(ex["tgt"]["tgt_ids"])
    return len(ex["src"]["src_ids"])


def clean_example(maybe_example):
    maybe_example["src"] = {"src": " ".join(maybe_example["src"])}
    if maybe_example["tgt"] is not None:
        maybe_example["tgt"] = {"tgt": " ".join(maybe_example["tgt"])}
    if "align" in maybe_example:
        maybe_example["align"] = " ".join(maybe_example["align"])
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
        transf_bucket = transform.batch_apply(
            sub_bucket, is_train=(task == CorpusTask.TRAIN), corpus_name=cid
        )
        for example, transform, cid in transf_bucket:
            example = clean_example(example)
            if len(example["src"]["src"]) > 0 and example["sco"] > threshold:
                transformed_bucket.append(example)

        # at this point an example looks like:
        # {'src': {'src': ..., 'feats': [....]},
        #  'tgt': {'tgt': ...},
        #  'src_original': ['tok1', ...'tokn'],
        #  'tgt_original': ['tok1', ...'tokm'],
        #  'cid': corpus id
        #  'cid_line_number' : cid line number
        #  'align': ...,
        # }
    if len(transformed_bucket) > 0:
        return transformed_bucket
    else:
        return None


def numericalize(vocabs, example, model_type=ModelType.ENCODER_DECODER):
    """ """
    decoder_start_token = vocabs["decoder_start_token"]
    numeric = example
    numeric["src"]["src_ids"] = []
    if model_type == ModelType.ENCODER_DECODER:
        src_text = example["src"]["src"].split(" ")
        numeric["src"]["src_ids"] = vocabs["src"](src_text)
        if example["tgt"] is not None:
            numeric["tgt"]["tgt_ids"] = []
            tgt_text = example["tgt"]["tgt"].split(" ")
            numeric["tgt"]["tgt_ids"] = vocabs["tgt"](
                [decoder_start_token] + tgt_text + [DefaultTokens.EOS]
            )

    elif model_type == ModelType.DECODER:
        src_text = example["src"]["src"].split(" ")
        if decoder_start_token != "":
            src_text = [decoder_start_token] + src_text
        numeric["src"]["src_ids"] = vocabs["src"](src_text)
        if example["tgt"] is not None:
            numeric["tgt"]["tgt_ids"] = []
            tgt_text = example["tgt"]["tgt"].split(" ")
            numeric["tgt"]["tgt_ids"] = vocabs["tgt"](tgt_text + [DefaultTokens.EOS])
            if decoder_start_token == "":
                numeric["tgt"]["tgt_ids"] = numeric["tgt"]["tgt_ids"][1:]

    elif model_type == ModelType.ENCODER:
        src_text = example["src"]["src"].split(" ")
        if example["tgt"] is not None:  # TO BE DISCUSSED
            tgt_text = example["tgt"]["tgt"].split(" ")
            txt = (
                [DefaultTokens.BOS]
                + tgt_text
                + [DefaultTokens.EOS]
                + [DefaultTokens.EOS]
                + src_text
                + [DefaultTokens.EOS]
            )
            numeric["src"]["src_ids"] = vocabs["src"](txt)
            numeric["tgt"]["tgt_ids"] = vocabs["src"](txt)
        else:
            txt = [DefaultTokens.BOS] + src_text + [DefaultTokens.EOS]
            numeric["src"]["src_ids"] = vocabs["src"](txt)

    else:
        raise ValueError(f"Something went wrong with model_type {model_type}")

    return numeric


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
            torch.tensor(ex["src"]["src_ids"], dtype=torch.long, device=device).flip(
                dims=[0]
            )
            for ex, indice in minibatch
        ]
    else:
        tbatchsrc = [
            torch.tensor(ex["src"]["src_ids"], dtype=torch.long, device=device)
            for ex, indice in minibatch
        ]
    padidx = vocabs["src"][DefaultTokens.PAD]
    tbatchsrc = pad_sequence(tbatchsrc, batch_first=True, padding_value=padidx)

    if left_pad:
        tensor_batch["src"] = tbatchsrc.flip(dims=[1])
    else:
        tensor_batch["src"] = tbatchsrc

    tensor_batch["srclen"] = torch.tensor(
        [len(ex["src"]["src_ids"]) for ex, indice in minibatch],
        dtype=torch.long,
        device=device,
    )

    if minibatch[0][0]["tgt"] is not None:
        if left_pad:
            tbatchtgt = [
                torch.tensor(
                    ex["tgt"]["tgt_ids"], dtype=torch.long, device=device
                ).flip(dims=[0])
                for ex, indice in minibatch
            ]
        else:
            tbatchtgt = [
                torch.tensor(ex["tgt"]["tgt_ids"], dtype=torch.long, device=device)
                for ex, indice in minibatch
            ]

        padidx = vocabs["tgt"][DefaultTokens.PAD]
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
            alignment[i, : len(ex["alignment"])] = torch.tensor(
                ex["alignment"], dtype=torch.long, device=device
            )
        tensor_batch["alignment"] = alignment

    if "src_ex_vocab" in minibatch[0][0].keys():
        tensor_batch["src_ex_vocab"] = [ex["src_ex_vocab"] for ex, indice in minibatch]

    tensor_batch["ind_in_bucket"] = [indice for ex, indice in minibatch]

    tensor_batch["cid"] = [ex["cid"] for ex, indice in minibatch]
    tensor_batch["cid_line_number"] = [
        ex["cid_line_number"] for ex, indice in minibatch
    ]

    if minibatch[0][0]["cid"] != "infer":
        tensor_batch["sco"] = torch.tensor(
            [ex["sco"] for ex, indice in minibatch], device=device
        )

    return tensor_batch
