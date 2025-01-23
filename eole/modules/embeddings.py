""" Embeddings module """

import math
import torch
import torch.nn as nn
from torch.nn.utils import skip_init

from eole.utils.logging import logger
from eole.constants import PositionEncodingType


class SequenceTooLongError(Exception):
    pass


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dim (int): embedding size
    """

    def __init__(self, dim, enc_type, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(dim))
        if enc_type == PositionEncodingType.SinusoidalInterleaved:
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
        elif enc_type == PositionEncodingType.SinusoidalConcat:
            half_dim = dim // 2
            pe = math.log(10000) / (half_dim - 1)
            pe = torch.exp(torch.arange(half_dim, dtype=torch.float) * -pe)
            pe = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * pe.unsqueeze(0)
            pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1).view(max_len, -1)
        else:
            raise ValueError("Choice of Position encoding is SinusoidalInterleaved or" " SinusoidalConcat.")
        pe = pe.unsqueeze(1)  # we keep pe (len x batch x dim) for back comp
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        pe = self.pe.transpose(0, 1)  # (batch x len x dim)
        emb = emb * math.sqrt(self.dim)
        step = step or 0
        if pe.size(1) < step + emb.size(1):
            raise SequenceTooLongError(
                f"Sequence is {emb.size(1) + step} but PositionalEncoding is"
                f" limited to {self.pe.size(1)}. See max_len argument."
            )
        emb = emb + pe[:, step : emb.size(1) + step, :]

        return emb


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        word_padding_idx (int): padding index for words in the embeddings.
        position_encoding_type (str): see :constants:`~eole.constants.PositionEncodingType`
        position_shift (int): patch int for xlm-roberta-xl
        dropout (float): dropout probability.
        sparse (bool): sparse embbedings default False
        freeze_word_vecs (bool): freeze weights of word vectors.
        n_positions (int): number of positions for Learned position embeddings
    """

    def __init__(
        self,
        word_vec_size,
        word_vocab_size,
        word_padding_idx,
        position_encoding_type="SinusoidalInterleaved",
        position_shift=0,
        dropout=0,
        sparse=False,
        freeze_word_vecs=False,
        n_positions=1024,
        normalize=False,
    ):
        super(Embeddings, self).__init__()
        self._validate_args()

        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        self.embeddings = skip_init(
            nn.Embedding,
            num_embeddings=word_vocab_size,
            embedding_dim=word_vec_size,
            padding_idx=word_padding_idx,
            sparse=sparse,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

        self.position_encoding_type = position_encoding_type
        self.position_shift = position_shift
        self.normalize = normalize

        if self.position_encoding_type == PositionEncodingType.Learned:
            self.pe = nn.Embedding(n_positions, word_vec_size)
            self.past_length = 0
        elif self.position_encoding_type in [
            PositionEncodingType.SinusoidalInterleaved,
            PositionEncodingType.SinusoidalConcat,
        ]:
            self.pe = PositionalEncoding(word_vec_size, position_encoding_type)
        else:
            # Rotary, Alibi, Relative are handled in MHA
            pass

        if freeze_word_vecs:
            self.embeddings.weight.requires_grad = False

    def _validate_args(
        self,
    ):
        pass

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """

        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.embeddings.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.embeddings.weight.data.copy_(pretrained[:, : self.word_vec_size])
            else:
                self.embeddings.weight.data.copy_(pretrained)

    def forward(self, source, step=None):
        """Computes the embeddings for words.

        Args:
            source (LongTensor): index tensor ``(batch, len)``

        Returns:
            FloatTensor: Word embeddings ``(batch, len, embedding_size)``
        """
        emb = self.embeddings(source)
        if self.position_encoding_type == PositionEncodingType.Learned:
            if step == 0 or step is None:
                # reset
                self.past_length = 0
            position_ids = torch.arange(
                self.past_length,
                source.size(-1) + self.past_length,
                dtype=torch.long,
                device=source.device,
            )
            position_ids = position_ids.unsqueeze(0) + self.position_shift
            position_emb = self.pe(position_ids)
            emb += position_emb
            if self.past_length == 0:
                self.past_length += source.size(-1)
            else:
                self.past_length += 1
        elif self.position_encoding_type in [
            PositionEncodingType.SinusoidalInterleaved,
            PositionEncodingType.SinusoidalConcat,
        ]:
            emb = self.pe(emb, step)

        if self.normalize:
            normalizer = torch.tensor(self.word_vec_size**0.5, dtype=emb.dtype)
            emb = emb * normalizer

        if self.dropout_p > 0:
            return self.dropout(emb)
        else:
            return emb

    def update_dropout(self, dropout):
        self.dropout.p = dropout


# Some utilitary functions for pretrained embeddings


def read_embeddings(path, skip_lines=0, filter_set=None):
    """
    Read an embeddings file in the glove format.
    """
    embs = dict()
    total_vectors_in_file = 0
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.decode("utf8").strip().split(" ")
            if len(l_split) == 2:
                continue
            total_vectors_in_file += 1
            if filter_set is not None and l_split[0] not in filter_set:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    return embs, total_vectors_in_file


def calc_vocab_load_stats(vocab, loaded_embed_dict):
    matching_count = len(set(vocab.ids_to_tokens) & set(loaded_embed_dict.keys()))
    missing_count = len(vocab) - matching_count
    percent_matching = matching_count / len(vocab) * 100
    return matching_count, missing_count, percent_matching


def convert_to_torch_tensor(word_to_float_list_dict, vocab):
    dim = len(next(iter(word_to_float_list_dict.values())))
    tensor = torch.zeros((len(vocab), dim))
    for word, values in word_to_float_list_dict.items():
        tensor[vocab.tokens_to_ids[word]] = torch.Tensor(values)
    return tensor


def prepare_pretrained_embeddings(config, vocabs):
    if all(
        [
            config.both_embeddings is None,
            config.src_embeddings is None,
            config.tgt_embeddings is None,
        ]
    ):
        return

    assert (
        config.save_data
    ), "-save_data is required when using \
        pretrained embeddings."

    vocs = []
    for side in ["src", "tgt"]:
        vocab = vocabs[side]
        vocs.append(vocab)
    enc_vocab, dec_vocab = vocs

    skip_lines = 1 if config.embeddings_type == "word2vec" else 0
    if config.both_embeddings is not None:
        set_of_src_and_tgt_vocab = set(enc_vocab.ids_to_tokens) | set(dec_vocab.ids_to_tokens)
        logger.info("Reading encoder and decoder embeddings from {}".format(config.both_embeddings))
        src_vectors, total_vec_count = read_embeddings(config.both_embeddings, skip_lines, set_of_src_and_tgt_vocab)
        tgt_vectors = src_vectors
        logger.info("\tFound {} total vectors in file".format(total_vec_count))
    else:
        if config.src_embeddings is not None:
            logger.info("Reading encoder embeddings from {}".format(config.src_embeddings))
            src_vectors, total_vec_count = read_embeddings(
                config.src_embeddings,
                skip_lines,
                filter_set=set(enc_vocab.ids_to_tokens),
            )
            logger.info("\tFound {} total vectors in file.".format(total_vec_count))
        else:
            src_vectors = None
        if config.tgt_embeddings is not None:
            logger.info("Reading decoder embeddings from {}".format(config.tgt_embeddings))
            tgt_vectors, total_vec_count = read_embeddings(
                config.tgt_embeddings,
                skip_lines,
                filter_set=set(dec_vocab.ids_to_tokens),
            )
            logger.info("\tFound {} total vectors in file".format(total_vec_count))
        else:
            tgt_vectors = None
    logger.info("After filtering to vectors in vocab:")
    if config.src_embeddings is not None or config.both_embeddings is not None:
        logger.info("\t* enc: %d match, %d missing, (%.2f%%)" % calc_vocab_load_stats(enc_vocab, src_vectors))
    if config.tgt_embeddings is not None or config.both_embeddings is not None:
        logger.info("\t* dec: %d match, %d missing, (%.2f%%)" % calc_vocab_load_stats(dec_vocab, tgt_vectors))

    # Write to file
    enc_output_file = config.save_data + ".enc_embeddings.pt"
    dec_output_file = config.save_data + ".dec_embeddings.pt"
    if config.src_embeddings is not None or config.both_embeddings is not None:
        logger.info("\nSaving encoder embeddings as:\n\t* enc: %s" % enc_output_file)
        torch.save(convert_to_torch_tensor(src_vectors, enc_vocab), enc_output_file)
        # set the opt in place
        config.pre_word_vecs_enc = enc_output_file
    if config.tgt_embeddings is not None or config.both_embeddings is not None:
        logger.info("\nSaving decoder embeddings as:\n\t* dec: %s" % dec_output_file)
        torch.save(convert_to_torch_tensor(tgt_vectors, dec_vocab), dec_output_file)
        # set the opt in place
        config.pre_word_vecs_dec = dec_output_file
