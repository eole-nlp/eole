"""Ensemble decoding.

Decodes using multiple models simultaneously,
combining their prediction distributions by averaging.
All models in the ensemble must share a target vocabulary.
"""

import torch
import torch.nn as nn
import copy
from eole.encoders.encoder import EncoderBase
from eole.decoders.decoder import DecoderBase
from eole.models.model import EncoderDecoderModel, BaseModel


class EnsembleDecoderOutput(object):
    """Wrapper around multiple decoder final hidden states."""

    def __init__(self, model_dec_outs):
        self.model_dec_outs = tuple(model_dec_outs)

    def squeeze(self, dim=None):
        """Delegate squeeze to avoid modifying
        :func:`eole.predict.translator.Translator.predict_batch()`
        """
        return EnsembleDecoderOutput([x.squeeze(dim) for x in self.model_dec_outs])

    def __getitem__(self, index):
        return self.model_dec_outs[index]


class EnsembleSrcEmb(nn.Module):
    """Dummy SrcEmb that delegates to individual real SrcEmb."""

    def __init__(self, model_src_embs):
        super(EnsembleSrcEmb, self).__init__()
        self.model_src_embs = nn.ModuleList(model_src_embs)

    def forward(self, src):
        src_emb = [model_src_emb(src) for model_src_emb in self.model_src_embs]
        return src_emb


class EnsembleEncoder(EncoderBase):
    """Dummy Encoder that delegates to individual real Encoders."""

    def __init__(self, model_encoders):
        super(EnsembleEncoder, self).__init__()
        self.model_encoders = nn.ModuleList(model_encoders)

    def forward(self, emb, mask=None):
        enc_out, enc_final_hs = zip(
            *[
                model_encoder(emb[i], mask)
                for i, model_encoder in enumerate(self.model_encoders)
            ]
        )
        return enc_out, enc_final_hs


class EnsembleTgtEmb(nn.Module):
    """Dummy TgtEmb that delegates to individual real TgtEmb."""

    def __init__(self, model_tgt_embs):
        super(EnsembleTgtEmb, self).__init__()
        self.model_tgt_embs = nn.ModuleList(model_tgt_embs)

    def forward(self, tgt, step=None):
        tgt_emb = [model_tgt_emb(tgt, step) for model_tgt_emb in self.model_tgt_embs]
        return tgt_emb


class EnsembleDecoder(DecoderBase):
    """Dummy Decoder that delegates to individual real Decoders."""

    def __init__(self, model_decoders):
        model_decoders = nn.ModuleList(model_decoders)
        attentional = any([dec.attentional for dec in model_decoders])
        super(EnsembleDecoder, self).__init__(attentional)
        self.model_decoders = model_decoders

    def forward(self, emb, enc_out=None, src_len=None, step=None, **kwargs):
        """See :func:`eole.decoders.decoder.DecoderBase.forward()`."""
        # src_len is a single tensor shared between all models.
        # This assumption will not hold if Translator is modified
        # to calculate src_len as something other than the length
        # of the input.
        dec_outs, attns = zip(
            *[
                model_decoder(
                    emb[i],
                    enc_out=None if enc_out is None else enc_out[i],
                    src_len=src_len,
                    step=step,
                    **kwargs
                )
                for i, model_decoder in enumerate(self.model_decoders)
            ]
        )
        if attns[0]["std"] is not None:
            mean_attns = self.combine_attns(attns)
        else:
            mean_attns = attns
        return EnsembleDecoderOutput(dec_outs), mean_attns

    def combine_attns(self, attns):
        result = {}
        for key in attns[0].keys():
            result[key] = torch.stack(
                [attn[key] for attn in attns if attn[key] is not None]
            ).mean(0)
        return result

    def init_state(self, enc_out=None, src=None, enc_final_hs=None):
        """See :obj:`RNNDecoderBase.init_state()`"""
        for i, model_decoder in enumerate(self.model_decoders):
            if enc_out is not None:
                enc_out_i = enc_out[i]
            else:
                enc_out_i = None
            if enc_final_hs is not None:
                enc_final_hs_i = enc_final_hs[i]
            else:
                enc_final_hs_i = None
            model_decoder.init_state(
                src=src, enc_out=enc_out_i, enc_final_hs=enc_final_hs_i
            )

    def map_state(self, fn):
        for model_decoder in self.model_decoders:
            model_decoder.map_state(fn)


class EnsembleGenerator(nn.Module):
    """
    Dummy Generator that delegates to individual real Generators,
    and then averages the resulting target distributions.
    """

    def __init__(self, model_generators, raw_probs=False):
        super(EnsembleGenerator, self).__init__()
        self.model_generators = nn.ModuleList(model_generators)
        self._raw_probs = raw_probs

    def forward(self, hidden, attn=None, src_map=None):
        """
        Compute a distribution over the target dictionary
        by averaging distributions from models in the ensemble.
        All models in the ensemble must share a target vocabulary.
        """
        distributions = torch.stack(
            [
                mg(h) if attn is None else mg(h, attn, src_map)
                for h, mg in zip(hidden, self.model_generators)
            ]
        )
        if self._raw_probs:
            return torch.log(torch.exp(distributions).mean(0))
        else:
            return distributions.mean(0)


class EnsembleModel(EncoderDecoderModel):
    """Dummy EncoderDecoderModel wrapping individual real EncoderDecoderModels."""

    def __init__(self, models, raw_probs=False):
        src_emb = EnsembleSrcEmb(model.src_emb for model in models)
        encoder = EnsembleEncoder(model.encoder for model in models)
        tgt_emb = EnsembleTgtEmb(model.tgt_emb for model in models)
        decoder = EnsembleDecoder(model.decoder for model in models)
        hidden_size = models[0].hidden_size
        super(EnsembleModel, self).__init__(
            encoder=encoder,
            decoder=decoder,
            src_emb=src_emb,
            tgt_emb=tgt_emb,
            hidden_size=hidden_size,
        )
        self.generator = EnsembleGenerator(
            [model.generator for model in models], raw_probs
        )
        self.models = nn.ModuleList(models)


def load_test_model(config, device_id=0):
    """Read in multiple models for ensemble."""
    shared_vocabs = None
    shared_model_config = None
    models = []
    config2 = copy.deepcopy(config)
    for i, model_path in enumerate(config.model_path):
        config2.model_path = [config.model_path[i]]
        vocabs, model, model_config = BaseModel.load_test_model(
            config2, device_id, model_path=model_path
        )
        if shared_vocabs is None:
            shared_vocabs = vocabs
        else:
            assert (
                shared_vocabs["src"].tokens_to_ids == vocabs["src"].tokens_to_ids
            ), "Ensemble models must use the same vocabs "
        models.append(model)
        if shared_model_config is None:
            shared_model_config = model_config
    ensemble_model = EnsembleModel(models, config.avg_raw_probs)
    return shared_vocabs, ensemble_model, shared_model_config
