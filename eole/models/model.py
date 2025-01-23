import torch
import torch.nn as nn
from glob import glob
from collections import defaultdict
import os

from eole.utils.logging import logger

# might not be useful after full refactor
from eole.modules.lora import (
    replace_lora_linear,
    replace_lora_embedding,
    mark_only_lora_as_trainable,
)
from torch.nn.utils import skip_init
from torch.nn.init import xavier_uniform_, zeros_, uniform_, normal_
from eole.utils.misc import use_gpu, get_device
from eole.inputters.inputter import dict_to_vocabs

# copied from model_builder to facilitate tests, but should not live there in the end
from eole.encoders import str2enc
from eole.decoders import str2dec
from eole.constants import DefaultTokens, PositionEncodingType
from eole.modules.embeddings import Embeddings
from eole.modules.rope import RotaryPosition
from eole.models.model_saver import load_checkpoint
from eole.modules.estimator import FeedForward


class NoOpPosition:
    """A no-op position encoding callable."""

    def update(self, *args, **kwargs):
        return None


def build_rope(model_config):
    if model_config.embeddings.position_encoding_type == PositionEncodingType.Rotary:
        return RotaryPosition(model_config)
    else:
        return NoOpPosition()


def build_encoder(model_config, running_config=None):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
    """
    enc_type = model_config.encoder.encoder_type
    return str2enc[enc_type].from_config(model_config.encoder, running_config=running_config)  # full config for now


def build_decoder(model_config, running_config=None, with_cross_attn=False):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
    """
    dec_type = (
        "ifrnn"
        if model_config.decoder.decoder_type == "rnn" and model_config.input_feed
        else model_config.decoder.decoder_type
    )
    return str2dec[dec_type].from_config(
        model_config.decoder,
        running_config=running_config,
        with_cross_attn=with_cross_attn,
    )


def build_src_emb(model_config, vocabs, running_config=None):
    # Build embeddings.
    pad_token = vocabs["specials"].get("pad_token", DefaultTokens.PAD)
    src_emb = Embeddings(
        word_vec_size=model_config.embeddings.src_word_vec_size,
        position_encoding_type=model_config.embeddings.position_encoding_type,
        position_shift=model_config.embeddings.position_shift,
        dropout=getattr(running_config, "dropout", [0.0])[0],
        word_padding_idx=vocabs["tgt"][pad_token],
        word_vocab_size=len(vocabs["src"]),
        sparse=getattr(running_config, "optim", None) == "sparseadam",
        freeze_word_vecs=model_config.embeddings.freeze_word_vecs_enc,
        n_positions=model_config.embeddings.n_positions,
        normalize=model_config.embeddings.normalize,
    )
    return src_emb


def build_tgt_emb(model_config, vocabs, running_config=None, share_embeddings=False, src_emb=None):
    # Build embeddings.
    pad_token = vocabs["specials"].get("pad_token", DefaultTokens.PAD)
    tgt_emb = Embeddings(
        word_vec_size=model_config.embeddings.tgt_word_vec_size,
        position_encoding_type=model_config.embeddings.position_encoding_type,
        position_shift=model_config.embeddings.position_shift,
        dropout=getattr(running_config, "dropout", [0.0])[0],
        word_padding_idx=vocabs["tgt"][pad_token],
        word_vocab_size=len(vocabs["tgt"]),
        sparse=getattr(running_config, "optim", None) == "sparseadam",
        freeze_word_vecs=model_config.embeddings.freeze_word_vecs_dec,
        n_positions=model_config.embeddings.n_positions,
        normalize=model_config.embeddings.normalize,
    )

    if share_embeddings:
        tgt_emb.embeddings.weight = src_emb.embeddings.weight  # not sure if we need pe.weight too

    return tgt_emb


class BaseModel(nn.Module):
    """Core trainable object. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.

    Args:
      encoder (eole.encoders.EncoderBase): an encoder object
      decoder (eole.decoders.DecoderBase): a decoder object"""

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        self.encoder = kwargs.get("encoder", None)
        self.decoder = kwargs.get("decoder", None)
        self.src_emb = kwargs.get("src_emb", None)
        self.tgt_emb = kwargs.get("tgt_emb", None)
        self.add_estimator = kwargs.get("add_estimator", False)
        self.hidden_size = kwargs.get("hidden_size", None)
        self.rope = kwargs.get("rope", None)
        self.share_decoder_embeddings = False
        if self.encoder is not None and self.src_emb is None:
            raise ValueError("An Encoder needs source Embeddings")
        if self.decoder is not None and self.tgt_emb is None:
            raise ValueError("A Decoder needs target Embeddings")
        if self.encoder is None and self.decoder is None:
            raise ValueError("A Model need an Encoder or a Decoder")

    @classmethod
    def build_blocks(cls, model_config, vocabs, running_config=None):
        """Where the blocks (encoder/decoder/etc) are actually instantiated,
        depending on the actual subclass."""
        # TODO we might want to put the share_embeddings assert here (previous model_builder L228)
        raise NotImplementedError

    def maybe_quantize(self, running_config):
        # these are not supposed to be "model" opts, but rather "chekpoint" opts
        # -> at some point we need to retrieve such opts from checkpoint
        nonlora_to_quant = [
            layer
            for layer in getattr(running_config, "quant_layers", [])
            if layer not in getattr(running_config, "lora_layers", [])
        ]
        if hasattr(running_config, "quant_layers") and len(nonlora_to_quant) > 0:
            if running_config.quant_type in ["bnb_8bit", "bnb_FP4", "bnb_NF4"]:
                logger.info("%s compression of layer %s" % (running_config.quant_type, nonlora_to_quant))
                try:
                    from eole.modules.bnb_linear import replace_bnb_linear
                except ImportError:
                    raise ImportError("Install bitsandbytes to use 4/8bit compression")
                # try to do this inplace, not sure it'll work
                replace_bnb_linear(
                    self,
                    module_to_convert=nonlora_to_quant,
                    q_type=running_config.quant_type,
                )
            elif running_config.quant_type in ["awq_gemm", "awq_gemv"]:
                logger.info("%s compression of layer %s" % (running_config.quant_type, nonlora_to_quant))
                try:
                    from eole.modules.awq_linear import replace_awq_linear
                except ImportError:
                    raise ImportError("Install AutoAWQ to use awq quantized model")
                # try to do this inplace, not sure it'll work
                replace_awq_linear(
                    self,
                    module_to_convert=nonlora_to_quant,
                    w_bit=running_config.w_bit,
                    group_size=running_config.group_size,
                    q_type=running_config.quant_type,
                )
            else:
                logger.info("compression type %s not supported." % running_config.quant_type)

    def maybe_lora(self, running_config):
        mark_lora = False
        if hasattr(running_config, "lora_layers") and len(running_config.lora_layers) > 0:
            # I think we need to allow encoder freezing while training LoRa
            # if running_config.freeze_encoder or running_config.freeze_decoder:
            #    raise ValueError("Cannot use LoRa with Enc/Dec-oder freezing")
            for layer in running_config.lora_layers:
                if hasattr(running_config, "quant_layers") and layer in running_config.quant_layers:
                    quant_type = running_config.quant_type
                else:
                    quant_type = None
                logger.info("Adding LoRa layers for %s quant %s" % (layer, quant_type))
                # try to do this inplace, not sure it'll work
                # alternative could be to make without recursion if possible
                replace_lora_linear(
                    self,
                    r=running_config.lora_rank,
                    lora_alpha=running_config.lora_alpha,
                    lora_dropout=running_config.lora_dropout,
                    layer=layer,
                    quant_type=quant_type,
                    use_ckpting=running_config.use_ckpting,
                )
            mark_lora = True
        if hasattr(running_config, "lora_embedding") and running_config.lora_embedding:
            if running_config.freeze_encoder or running_config.freeze_decoder:
                raise ValueError("Cannot use LoRa with Enc/Dec-oder freezing")  # TODO move this to config validation
            logger.info("Adding LoRa Embeddings")
            # same, trying inplace
            replace_lora_embedding(self, r=running_config.lora_rank, lora_alpha=running_config.lora_alpha)
            mark_lora = True

        if mark_lora:
            mark_only_lora_as_trainable(self, bias="lora_only")

    def build_generator(self, model_config, running_config, vocabs):
        # patched to make it work with decoder config, but might be improved
        self.generator = skip_init(
            nn.Linear,
            in_features=model_config.decoder.hidden_size,
            out_features=len(vocabs["tgt"]),
            bias=model_config.generator_bias,
        )
        if model_config.share_decoder_embeddings:
            self.generator.weight = self.tgt_emb.embeddings.weight
        elif hasattr(running_config, "lora_embedding") and running_config.lora_embedding:
            logger.info("Generator and decoder not tied Adding LoRa Generator")
            replace_lora_linear(
                self,
                r=running_config.lora_rank,
                lora_alpha=running_config.lora_alpha,
                lora_dropout=running_config.lora_dropout,
                layer="generator",
                quant_type=None,
                use_ckpting=running_config.use_ckpting,
            )
            mark_only_lora_as_trainable(self, bias="lora_only")

    def load_checkpoint(
        self,
        running_config,
        vocabs,
        checkpoint,
        device,
        offset,
        model_path=None,
    ):
        if model_path is None:
            model_path = running_config.get_model_path()
        # when using LoRa or updating the vocab (no more embeddings in ckpt)
        # => strict=False when loading state_dict
        strict = not getattr(running_config, "update_vocab", False)
        self.load_safe_state_dict(
            model_path,
            running_config=running_config,
            vocabs=vocabs,
            checkpoint=checkpoint,
            device=device,
            offset=offset,
            strict=strict,
        )

    # probably good to refactor in some way, but working for now
    def training_logic(self, running_config, vocabs, checkpoint, device_id):
        if running_config.world_size > 1 and running_config.parallel_mode == "tensor_parallel":
            device = get_device()
            offset = device_id
        else:
            if use_gpu(running_config):
                device = get_device()
            else:
                device = torch.device("cpu")
            offset = 0

        if checkpoint is not None:
            self.load_checkpoint(running_config, vocabs, checkpoint, device, offset)
        else:
            self.to(running_config.storage_dtype)
            self.to(device)

        # currently in TrainingConfig which makes more sense
        if running_config.freeze_encoder:
            self.encoder.requires_grad_(False)

        if running_config.freeze_decoder:
            self.decoder.requires_grad_(False)
            self.generator.requires_grad_(False)

    @classmethod
    def inference_logic(self, checkpoint, running_config, vocabs, device_id=None):
        model_config = running_config.model  # loaded in PredictConfig validation
        if running_config.world_size > 1 and running_config.parallel_mode == "tensor_parallel":
            # same, probably not needed anymore
            # training_config.world_size = running_config.world_size
            # training_config.parallel_mode = running_config.parallel_mode
            # training_config.gpu_ranks = running_config.gpu_ranks
            device = torch.device("cuda", device_id)
            offset = device_id
        else:
            if use_gpu(running_config):
                if len(running_config.gpu_ranks) > 0:
                    device_id = running_config.gpu_ranks[0]
                device = get_device(device_id=device_id)
            else:
                device = torch.device("cpu")
            offset = 0

        vocabs = dict_to_vocabs(checkpoint["vocab"])

        return (
            model_config,
            running_config,
            vocabs,
            device,
            offset,
        )

    @classmethod
    def build_base_model(cls, config, vocabs, running_config=None):
        model = cls.build_blocks(config, vocabs, running_config=running_config)
        model.maybe_quantize(running_config)
        model.maybe_lora(running_config)
        if config.decoder is not None:
            model.build_generator(config, running_config, vocabs)
        else:
            model.generator = None
        return model

    @classmethod
    def load_test_model(cls, config, device_id=0, model_path=None):
        """
        Moved from legacy model_builder.
        We can probably refactor better this whole logic at some point.
        """
        if model_path is None:
            model_path = config.model_path[0]
        checkpoint = load_checkpoint(model_path)

        checkpoint_model_config = checkpoint["config"].model
        # we actually need to merge inference opts and config here
        update_dict = {
            "model": checkpoint_model_config,
        }
        # if not set in inference config, override with checkpoint (generalize to more fields?)
        if "quant_type" not in config.model_fields_set:
            update_dict["quant_type"] = checkpoint["config"].training.quant_type
        if "quant_layers" not in config.model_fields_set:
            update_dict["quant_layers"] = checkpoint["config"].training.quant_layers
        config.update(**update_dict)
        vocabs = None
        # not super clean inheritance anti-pattern,
        # might be enhanced if checkpoint loading is split from model instanciation
        model, vocabs, model_config = get_model_class(config.model).from_config(
            config, vocabs, checkpoint, device_id, running_config=config
        )

        return vocabs, model, model_config

    def init_weights(self, running_config):
        match running_config.param_init_method:
            case "normal":
                for module in self.modules():
                    for param_name, param in module.named_parameters():
                        if param_name == "weight" and param.dim() > 1:
                            normal_(module.weight, mean=0.0, std=running_config.param_init)
                        elif param_name == "bias":
                            zeros_(param)
            case "uniform":
                # taken from legacy code, we might want to zero bias for coherence
                for param in self.parameters():
                    uniform_(param, -running_config.param_init, running_config.param_init)
            case "xavier_uniform":
                for module in self.modules():
                    for param_name, param in module.named_parameters():
                        if param_name == "weight" and param.dim() > 1:
                            xavier_uniform_(param)
                        elif param_name == "bias":
                            zeros_(param)

    @classmethod
    def from_config(
        cls,
        model_config,  # ModelConfig
        vocabs,
        checkpoint,
        device_id,
        training=False,
        running_config=None,  # TrainingConfig / PredictConfig
    ):
        # shall we split in two classes InferenceModel/TrainingModel ?
        # opt = running_config  # patch to make it work, need to diffuse change downstream
        # patch for inference mode
        # if hasattr(config, "model"):
        #     model_opt = config.model
        # else:
        #     model_opt = opt
        # model_opt = config
        logger.info("Building model...")
        # moving up in build_model chain, step by step
        if not training:
            (
                model_config,
                running_config,
                vocabs,
                device,
                offset,
            ) = cls.inference_logic(checkpoint, running_config, vocabs, device_id)
        model = cls.build_blocks(
            model_config, vocabs, running_config=running_config
        )  # corresponds to build_task_specific_model
        # TODO: handle this better at some point?
        model.share_decoder_embeddings = model_config.share_decoder_embeddings
        # generator -> shall it be called within build_blocks?
        if model_config.decoder is not None:
            model.build_generator(model_config, running_config, vocabs)
        else:
            model.generator = None
        # 1 build_base_model
        # quantization stuff
        model.maybe_quantize(running_config)  # not sure about this one, as quantize opts are not in model_config
        # -> it's not a model config we need here but a running config potentially updated with checkpoint stuff # noqa: E501
        # lora stuff
        model.maybe_lora(running_config)  # same
        # 0 build_model # TODO
        # init params & update vocab stuff
        # If new training initialize the model params
        # If update_vocab init also but checkpoint will overwrite old weights
        if checkpoint is None or getattr(running_config, "update_vocab", False):
            model.init_weights(running_config)

            if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
                model.src_emb.load_pretrained_vectors(running_config.pre_word_vecs_enc)
            if hasattr(model.decoder, "embeddings"):
                model.tgt_emb.load_pretrained_vectors(running_config.pre_word_vecs_dec)
        # dtype/amp/apex stuff
        # ONLY for legacy fusedam with amp pytorch requires NOT to half the model
        if training:
            model.training_logic(running_config, vocabs, checkpoint, device_id)
            logger.info(model)
        else:
            # actually load state dict here
            logger.info("Loading data into the model")
            model.load_checkpoint(
                running_config,
                vocabs,
                checkpoint,
                device,
                offset,
            )
            if running_config.compute_dtype == torch.int8:
                torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
            del checkpoint
            model.eval()
            for name, module in model.named_modules():
                if hasattr(module, "dropout_p"):
                    module.dropout_p = 0.0

        return model, vocabs, model_config

    def forward(self, src, tgt, src_len, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.

        Args:
            src (Tensor): A source sequence passed to encoder.
                Typically for input this will be a padded `LongTensor`
                of size ``(batch, len)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(batch, tgt_len)``.
            src_len(LongTensor): The src lengths, pre-padding ``(batch,)``.
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(batch, tgt_len, hidden)``
            * dictionary of attention weights ``(batch, tgt_len, src_len)``"""

        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        raise NotImplementedError

    def _load_param(self, name, module, param_name, param, buf_list, ckpt_t, offset):
        if module.__class__.__name__ == "WQLinear_GEMM":
            # ugly patch because in_feat and out_feat are reversed in WQLinear_GEMM
            param.data = param.data.transpose(0, 1)
            ckpt_t = ckpt_t.transpose(0, 1)
        if name.split(".")[-1] in [
            "linear_keys",
            "linear_values",
            "linear_query",
            "gate_up_proj",
            "up_proj",
        ]:
            col_slice_start = param.data.size(0) * offset
            col_slice_end = param.data.size(0) * (offset + 1)
        else:
            col_slice_start = 0
            col_slice_end = param.data.size(0)
        if param.data.dim() == 2:
            if name.split(".")[-1] in ["final_linear", "down_proj"]:
                row_slice_start = param.data.size(1) * offset
                row_slice_end = param.data.size(1) * (offset + 1)
            else:
                row_slice_start = 0
                row_slice_end = param.data.size(1)
            assert (
                param.data.size()
                == ckpt_t[
                    col_slice_start:col_slice_end,
                    row_slice_start:row_slice_end,
                ].size()
            ), (
                "An error in model's partition and checkpoint's slice was detected, "
                f"[{name}, {module}, {param_name}, {param.data.size()}, {ckpt_t.size()}]"
            )
            if name + "." + param_name in buf_list:
                if module.__class__.__name__ == "WQLinear_GEMM":
                    module.register_buffer(
                        param_name,
                        ckpt_t[
                            col_slice_start:col_slice_end,
                            row_slice_start:row_slice_end,
                        ].transpose(0, 1),
                    )
                else:
                    module.register_buffer(
                        param_name,
                        ckpt_t[
                            col_slice_start:col_slice_end,
                            row_slice_start:row_slice_end,
                        ],
                    )
            else:
                param.data = ckpt_t[
                    col_slice_start:col_slice_end,
                    row_slice_start:row_slice_end,
                ]
        else:
            assert (
                param.data.size() == ckpt_t[col_slice_start:col_slice_end].size()
            ), "An error in model's partition and checkpoint's slice was detected"
            if name + "." + param_name in buf_list:
                module.register_buffer(param_name, ckpt_t[col_slice_start:col_slice_end])
            else:
                param.data = ckpt_t[col_slice_start:col_slice_end]

    def load_safe_state_dict(
        self,
        model_path,
        running_config=None,
        vocabs=None,
        checkpoint=None,
        device=torch.device("cpu"),
        offset=0,
        strict=False,
    ):
        """Custom state_dict loading to enable moving module on device as they are loaded

        Args:
            model_path: Model path
            running_config: RunningConfig (either training or predict)
            vocabs: src/tgt vocabs
            checkpoint: config/vocabs loaded from checkpoint
            device: device to move parameters to
            offset: for tensor_parallel
            strict: stric loading of all parameters
        """
        # bitsandbytes quantize weights when .cuda() is called
        # for huge models we need to save Ram
        # so we load the weights  module by module and transfer them to GPU for quantization
        try:
            import safetensors
        except ImportError:
            raise ImportError("run: pip install safetensors, to use safetensors")
        enc_emb_name = "src_emb.embeddings.weight"
        dec_emb_name = "tgt_emb.embeddings.weight"
        keyfound = {}
        shards = glob(os.path.join(model_path, "model.*.safetensors"))
        if len(shards) == 0:
            raise ValueError("No safetensors file found")
        f = []
        keys_shard = {}
        for i, shard in enumerate(shards):
            f.append(safetensors.safe_open(shard, framework="pt", device="cpu"))
            for key in f[i].keys():
                keys_shard[key] = i
        if device == torch.device("cpu"):
            offset = 0
        buf_list = []
        for buf_name, buf in self.named_buffers():
            buf_list.append(buf_name)
        # special handling for update_vocab
        updated_params = {}
        if getattr(running_config, "update_vocab", False):
            updated_params["generator.weight"] = self.generator.state_dict()["weight"]
            updated_params["generator.bias"] = self.generator.state_dict()["bias"]
            for side, emb_name in [("src", enc_emb_name), ("tgt", dec_emb_name)]:
                if emb_name not in keys_shard.keys():
                    continue
                else:
                    # initialize
                    updated_params[emb_name] = self.state_dict()[emb_name]
                new_tokens = []
                checkpoint_vocabs = dict_to_vocabs(checkpoint["vocab"])
                for i, tok in enumerate(vocabs[side].ids_to_tokens):
                    if tok in checkpoint_vocabs[side]:
                        old_i = checkpoint_vocabs[side].lookup_token(tok)
                        checkpoint_emb = f[keys_shard[emb_name]].get_tensor(emb_name)
                        updated_params[emb_name][i] = checkpoint_emb[old_i]
                        if side == "tgt":
                            generator_weight = f[keys_shard["generator.weight"]].get_tensor("generator.weight")
                            updated_params["generator.weight"][i] = generator_weight[old_i]
                            generator_bias = f[keys_shard["generator.bias"]].get_tensor("generator.bias")
                            updated_params["generator.bias"][i] = generator_bias[old_i]
                    else:
                        new_tokens.append(tok)
                logger.info("%s: %d new tokens" % (side, len(new_tokens)))

        for name, module in self.named_modules():
            named_buf_and_param = list(module.named_buffers()) + list(module.named_parameters())
            for param_name, param in named_buf_and_param:
                # special handling for update vocab related stuff
                # if param_name in [enc_emb_name, dec_emb_name, ]
                if len(param_name.split(".")) == 1:  # only last key
                    if name + "." + param_name in updated_params.keys():
                        ckpt_t = updated_params[name + "." + param_name]
                        self._load_param(name, module, param_name, param, buf_list, ckpt_t, offset)
                        keyfound[name + "." + param_name] = True
                    elif name + "." + param_name in keys_shard.keys():

                        ckpt_t = f[keys_shard[name + "." + param_name]].get_tensor(name + "." + param_name)
                        self._load_param(name, module, param_name, param, buf_list, ckpt_t, offset)
                        keyfound[name + "." + param_name] = True
                    elif strict and ("lora" not in param_name and "slopes" not in param_name and "rope" not in name):
                        # Let's warn instead of just passing
                        logger.info("Missing key in safetensors checkpoint: %s" % name + "." + param_name)
                        if (
                            f"{name}.{param_name}" in ["generator.weight", "generator.bias"]
                            and self.share_decoder_embeddings
                        ):
                            logger.info(
                                "└─> Sharing from embeddings matrix since "
                                "`share_decoder_embeddings` flag is enabled."
                            )
                    if getattr(running_config, "compute_dtype", None) == torch.int8:
                        torch.quantization.quantize_dynamic(module, inplace=True)
                    else:
                        module.to(getattr(running_config, "storage_dtype", torch.float32))
                    module.to(device)
        for key in keys_shard.keys():
            if key not in keyfound.keys() and key not in buf_list:
                raise ValueError("Extra keys in model state_dict do not match the model config %s" % key)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns: (int, int)
            encoder side parameter count
            decoder side parameter count"""
        trainable = defaultdict(int)
        non_trainable = defaultdict(int)
        emb, enc, dec, gen, other = 0, 0, 0, 0, 0
        for name, param in self.named_parameters():
            if "encoder" in name:
                enc += param.nelement()
            elif "decoder" in name:
                dec += param.nelement()
            elif "embeddings" in name:
                emb += param.nelement()
            elif "generator" in name:
                gen += param.nelement()
            else:
                # mostly zero, but might be useful to grab edge cases
                other += param.nelement()
            if param.requires_grad:
                trainable[str(param.dtype)] += param.numel()
            else:
                non_trainable[str(param.dtype)] += param.numel()

        if callable(log):
            log("embeddings: {}".format(emb))
            log("encoder: {}".format(enc))
            log("decoder: {}".format(dec))
            log("generator: {}".format(gen))
            log("other: {}".format(other))
            log("* number of parameters: {}".format(emb + enc + dec + gen + other))
            log("Trainable parameters = %s" % str(dict(trainable)))
            log("Non trainable parameters = %s" % str(dict(non_trainable)))


class EncoderDecoderModel(BaseModel):
    """EncoderDecoderModel Class
    See :class:`~eole.models.BaseModel` for options."""

    def __init__(self, **kwargs):
        super(EncoderDecoderModel, self).__init__(**kwargs)
        self.tgt_shift = 1
        self.src_pad_idx = self.src_emb.word_padding_idx
        self.tgt_pad_idx = self.tgt_emb.word_padding_idx
        # we might want to disable this constructor some way
        if self.encoder is None or self.decoder is None:
            raise ValueError("A EncoderDecoderModel requires both an Encoder and a Decoder")
        if self.add_estimator:
            self.estimator = FeedForward(self.hidden_size)

    @classmethod
    def build_blocks(cls, model_config, vocabs, running_config=None):
        src_emb = build_src_emb(model_config, vocabs, running_config=running_config)
        encoder = build_encoder(model_config, running_config=running_config)
        tgt_emb = build_tgt_emb(
            model_config,
            vocabs,
            running_config=running_config,
            share_embeddings=model_config.share_embeddings,
            src_emb=src_emb,
        )
        decoder = build_decoder(model_config, running_config=running_config, with_cross_attn=True)
        return cls(
            encoder=encoder,
            decoder=decoder,
            src_emb=src_emb,
            tgt_emb=tgt_emb,
            add_estimator=model_config.add_estimator,
            hidden_size=model_config.decoder.hidden_size,
            rope=build_rope(model_config),
        )
        # from there, the base blocks exist, and the rest is done in the from_opt from base class

    def forward(self, src, tgt, src_len, with_align=False):
        """An EncoderDecoderModel forward the src side to the encoder.
        Then the output of encoder ``enc_out`` is forwarded to the
        decoder along with the target excluding the last token.
        The decoder state is initiliazed with:
        * enc_final_hs in the case of RNNs
        * enc_out + enc_final_hs in the case of CNNs
        * src in the case of Transformer"""
        src_pad_mask = src.eq(self.src_pad_idx).unsqueeze(1)  # [B, 1, T_src]
        position_embeddings = self.rope.update(src.size(1), step=0)
        enc_out, enc_final_hs = self.encoder(
            self.src_emb(src),
            pad_mask=src_pad_mask,
            position_embeddings=position_embeddings,
        )
        self.decoder.init_state(src=src, enc_out=enc_out, enc_final_hs=enc_final_hs)
        dec_in = tgt[:, :-1]
        tgt_pad_mask = dec_in.eq(self.tgt_pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        dec_out, attns = self.decoder(
            self.tgt_emb(dec_in),
            enc_out=enc_out,
            with_align=with_align,
            src_pad_mask=src_pad_mask,
            tgt_pad_mask=tgt_pad_mask,
            position_embeddings=position_embeddings,
        )

        if self.add_estimator:  # we take the average of dec_out using the pad mask
            pad_mask2 = ~dec_in.eq(self.tgt_pad_idx)
            in_estim2 = (dec_out * pad_mask2.unsqueeze(-1).float()).sum(dim=1) / pad_mask2.sum(
                dim=1, keepdim=True
            ).float()
            estim = self.estimator(in_estim2.to(dec_out.dtype)).squeeze(-1)
        else:
            estim = None

        return dec_out, attns, estim

    def update_dropout(self, dropout, attention_dropout):
        self.encoder.update_dropout(dropout, attention_dropout)
        self.src_emb.update_dropout(dropout)
        self.decoder.update_dropout(dropout, attention_dropout)
        self.tgt_emb.update_dropout(dropout)


class DecoderModel(BaseModel):
    """DecoderModel Class
    Currently TransformerLMDecoder is the only LM decoder implemented

    Args:
        decoder (eole.decoders.TransformerLMDecoder): a transformer decoder"""

    def __init__(self, **kwargs):
        super(DecoderModel, self).__init__(**kwargs)
        self.tgt_shift = 0
        self.pad_idx = self.tgt_emb.word_padding_idx
        if self.encoder is not None:
            raise ValueError("DecoderModel should not be used" "with an encoder")
        if self.decoder is None:
            raise ValueError("DecoderModel requires a Decoder")
        if self.add_estimator:
            self.estimator = FeedForward(self.hidden_size)

    @classmethod
    def build_blocks(cls, model_config, vocabs, running_config=None):
        tgt_emb = build_tgt_emb(model_config, vocabs, running_config=running_config)
        decoder = build_decoder(model_config, running_config=running_config, with_cross_attn=False)
        return cls(
            decoder=decoder,
            tgt_emb=tgt_emb,
            add_estimator=model_config.add_estimator,
            hidden_size=model_config.decoder.hidden_size,
            rope=build_rope(model_config),
        )
        # from there, the base blocks exist, and the rest is done in the from_opt from base class

    def forward(self, src, _, src_len, with_align=False):
        """A DecoderModel forward the src side to the decoder along
        with the source lengths vector. It is a decoder only LM (cf GPT-2)"""

        self.decoder.init_state()
        position_embeddings = self.rope.update(src.size(1), step=0)
        dec_out, attns = self.decoder(
            self.tgt_emb(src),
            tgt_pad_mask=src.eq(self.pad_idx).unsqueeze(1),
            position_embeddings=position_embeddings,
        )

        if self.add_estimator:  # we take the average of dec_out using the pad mask
            pad_mask2 = ~src.eq(self.pad_idx)
            in_estim2 = (dec_out * pad_mask2.unsqueeze(-1).float()).sum(dim=1) / pad_mask2.sum(
                dim=1, keepdim=True
            ).float()
            estim = self.estimator(in_estim2.to(dec_out.dtype)).squeeze(-1)
        else:
            estim = None

        return dec_out, attns, estim

    def update_dropout(self, dropout, attention_dropout):
        self.decoder.update_dropout(dropout, attention_dropout)
        self.tgt_emb.update_dropout(dropout)


class EncoderModel(BaseModel):
    """EncoderModel Class
    Currently TransformerEncoder is the only Encoder implemented

    Args:
        encoder (eole.encoders.TransformerEncoder): a transformer encoder"""

    def __init__(self, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.tgt_shift = 1
        self.pad_idx = self.src_emb.word_padding_idx
        if self.decoder is not None:
            raise ValueError("EncoderModel should not be used" "with a decoder")
        if self.encoder is None:
            raise ValueError("EncoderModel requires an Encoder")
        if self.add_estimator:
            self.estimator = FeedForward(self.hidden_size)

    @classmethod
    def build_blocks(cls, model_config, vocabs, running_config=None):
        src_emb = build_src_emb(model_config, vocabs, running_config=running_config)
        encoder = build_encoder(model_config, running_config=running_config)
        return cls(
            encoder=encoder,
            src_emb=src_emb,
            add_estimator=model_config.add_estimator,
            hidden_size=model_config.encoder.hidden_size,
            rope=build_rope(model_config),
        )
        # from there, the base blocks exist, and the rest is done in the from_opt from base class

    def forward(self, src, _, src_len, with_align=False):
        """An EncoderModel encodes the source sentence to build hidden states"""

        pad_mask = src.eq(self.pad_idx).unsqueeze(1)  # [B, 1, T_src]
        position_embeddings = self.rope.update(src.size(1), step=0)
        enc_out, enc_final_hs = self.encoder(
            self.src_emb(src),
            pad_mask=pad_mask,
            position_embeddings=position_embeddings,
        )
        if self.add_estimator:
            # Version with average
            """
            pad_mask1 = ~src.eq(self.pad_idx)
            in_estim1 = (enc_out * pad_mask1.unsqueeze(-1).float()).sum(
                dim=1
            ) / pad_mask1.sum(dim=1, keepdim=True).float()
            estim = self.estimator(in_estim1.half()).squeeze(-1)
            """
            # Version with first token
            estim = self.estimator(enc_out[:, 0, :]).squeeze(-1)
        else:
            estim = None

        return enc_out, enc_final_hs, estim

    def update_dropout(self, dropout, attention_dropout):
        self.encoder.update_dropout(dropout, attention_dropout)


def get_model_class(model_config):
    # might have more cases later
    if model_config.decoder is None:
        return EncoderModel
    elif model_config.encoder is None:
        return DecoderModel
    else:
        return EncoderDecoderModel
