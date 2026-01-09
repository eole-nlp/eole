import torch
import torch.nn as nn
from glob import glob
from collections import defaultdict
import os
import itertools

from eole.utils.logging import logger

# might not be useful after full refactor
from eole.modules.lora import (
    replace_lora_linear,
    replace_lora_embedding,
    mark_only_lora_as_trainable,
)
from torch.nn.utils import skip_init
from torch.nn.init import xavier_uniform_, zeros_, uniform_, normal_
from eole.utils.misc import get_device
from eole.inputters.inputter import dict_to_vocabs

# copied from model_builder to facilitate tests, but should not live there in the end
from eole.encoders import str2enc
from eole.decoders import str2dec
from eole.adapters import str2adapter
from eole.constants import DefaultTokens, LayerNormFP32
from eole.modules.embeddings import Embeddings
from eole.models.model_saver import get_metadata
from eole.modules.estimator import FeedForward

from eole.encoders.vision import VisionEncoder


def build_encoder(model_config, running_config=None):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
    """
    enc_type = model_config.encoder.encoder_type
    return str2enc[enc_type](model_config.encoder, running_config=running_config)


def build_adapter(model_config, running_config=None):
    """
    Various adapter dispatcher function.
    """
    adapter_type = model_config.adapter
    return str2adapter[adapter_type](model_config, running_config=running_config)


def build_decoder(model_config, running_config=None):
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
    return str2dec[dec_type](
        model_config.decoder,
        running_config=running_config,
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
        self.adapter = kwargs.get("adapter", None)
        self.decoder = kwargs.get("decoder", None)
        self.src_emb = kwargs.get("src_emb", None)
        self.tgt_emb = kwargs.get("tgt_emb", None)
        self.add_estimator = kwargs.get("add_estimator", False)
        self.estimator_type = kwargs.get("estimator_type", "average")
        self.hidden_size = kwargs.get("hidden_size", None)
        self.share_decoder_embeddings = False
        if self.encoder is not None and self.src_emb is None:
            if not isinstance(self.encoder, VisionEncoder):
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

    def _setup_device_and_offset(self, running_config, device_id=None):
        """Determine device and offset for tensor parallel or single device setup."""
        device = get_device(device_id=device_id)
        if running_config.world_size > 1 and running_config.parallel_mode == "tensor_parallel":
            tp_offset = device_id if device_id is not None else 0
        else:
            tp_offset = 0
        return device, tp_offset

    def _apply_freezing(self, running_config):
        """Apply encoder/decoder freezing based on config."""
        if running_config.freeze_encoder:
            self.encoder.requires_grad_(False)
            if hasattr(self, "adapter"):
                self.adapter.requires_grad_(False)

        if running_config.freeze_decoder:
            self.decoder.requires_grad_(False)
            self.generator.requires_grad_(False)

    def _initialize_weights_and_embeddings(self, running_config):
        """Initialize model weights and load pretrained embeddings."""
        self.init_weights(running_config)

        if hasattr(self, "encoder") and hasattr(self.encoder, "embeddings"):
            self.src_emb.load_pretrained_vectors(running_config.pre_word_vecs_enc)
        if hasattr(self.decoder, "embeddings"):
            self.tgt_emb.load_pretrained_vectors(running_config.pre_word_vecs_dec)

    def _prepare_for_inference(self, running_config):
        """Apply inference-specific transformations."""
        if running_config.compute_dtype == torch.int8:
            torch.quantization.quantize_dynamic(self, dtype=torch.qint8, inplace=True)

        self.eval()
        for name, module in self.named_modules():
            if hasattr(module, "dropout_p"):
                module.dropout_p = 0.0

    def _prepare_for_training(self, running_config, vocabs, metadata, device, tp_offset):
        """Apply training-specific setup."""
        should_load = metadata is not None
        should_init = metadata is None or getattr(running_config, "update_vocab", False)

        if should_init:
            self._initialize_weights_and_embeddings(running_config)

        if should_load:
            strict = not getattr(running_config, "update_vocab", False)
            self.load_safe_state_dict(
                running_config.get_model_path(),
                running_config=running_config,
                vocabs=vocabs,
                metadata=metadata,
                device=device,
                tp_offset=tp_offset,
                strict=strict,
            )
        else:
            self.to(running_config.storage_dtype)
            self.to(device)

        # Ensure LayerNormFP32 stays in float32
        for name, module in self.named_modules():
            if isinstance(module, LayerNormFP32):
                module.to(torch.float32)

        self._apply_freezing(running_config)

    @classmethod
    def build(cls, model_config, vocabs, running_config):
        """Build a complete model with all components."""
        logger.info("Building model...")

        # Build core blocks
        model = cls.build_blocks(model_config, vocabs, running_config=running_config)
        model.share_decoder_embeddings = model_config.share_decoder_embeddings

        # Build generator if needed
        if model_config.decoder is not None:
            model.build_generator(model_config, running_config, vocabs)
        else:
            model.generator = None

        # Apply quantization and LoRA
        model.maybe_quantize(running_config)
        model.maybe_lora(running_config)

        return model

    @classmethod
    def for_training(cls, model_config, vocabs, running_config, metadata=None, device_id=0):
        """Create and configure a model for training.

        Args:
            model_config: Model architecture configuration
            vocabs: Source and target vocabularies
            running_config: Training configuration
            metadata: Optional checkpoint metadata
            device_id: Device ID for multi-GPU setup

        Returns:
            Configured model ready for training
        """
        # Build model
        model = cls.build(model_config, vocabs, running_config)

        # Setup device
        device, tp_offset = model._setup_device_and_offset(running_config, device_id)

        # Prepare for training
        model._prepare_for_training(running_config, vocabs, metadata, device, tp_offset)

        logger.info(model)
        return model

    @classmethod
    def for_inference(cls, running_config, device_id=0, model_path=None):
        """Create and configure a model for inference.

        Args:
            running_config: Inference configuration (PredictConfig)
            device_id: Device ID for inference
            model_path: Optional path to model checkpoint

        Returns:
            tuple: (model, vocabs, model_config)
        """
        # Load metadata and setup
        if model_path is None:
            model_path = running_config.model_path[0]

        metadata = get_metadata(model_path)
        model_config = metadata["config"].model

        # Merge inference config with checkpoint config
        update_dict = {"model": model_config}
        if "quant_type" not in running_config.model_fields_set:
            update_dict["quant_type"] = metadata["config"].training.quant_type
        if "quant_layers" not in running_config.model_fields_set:
            update_dict["quant_layers"] = metadata["config"].training.quant_layers
        running_config.update(**update_dict)

        # Build vocabs
        vocabs = dict_to_vocabs(metadata["vocab"])

        # Build model
        model = cls.build(model_config, vocabs, running_config)

        # Setup device
        device, tp_offset = model._setup_device_and_offset(running_config, device_id)

        # Load weights
        logger.info("Loading data into the model")
        model.load_safe_state_dict(
            model_path,
            running_config=running_config,
            vocabs=vocabs,
            metadata=metadata,
            device=device,
            tp_offset=tp_offset,
            strict=True,
        )

        # Apply inference-specific setup
        model._prepare_for_inference(running_config)

        del metadata
        return model, vocabs, model_config

    def maybe_quantize(self, running_config):
        # -> at some point we need to retrieve such opts from metadata
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
        # TODO: when share_decoder_embeddings generator should be adjusted to lora_embeddings A/B
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

    def init_weights(self, running_config):
        match running_config.param_init_method:
            case "normal":
                for name, module in self.named_modules():
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

    def forward(self, src, tgt, src_len, with_align=False, **kwargs):
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

    def _reset_lora_to_fp32(self):
        """Reset LoRA parameters to FP32 after dtype conversion"""
        for name, param in self.named_parameters():
            if "lora" in name:
                module_name, param_name = name.rsplit(".", 1)
                module = self.get_submodule(module_name)
                setattr(module, param_name, nn.Parameter(param.to(torch.float32)))

    def _report_extra_keys(self, keys_shard, keyfound, buf_list):
        """Report any extra keys found in checkpoint"""
        for key in keys_shard:
            if key not in keyfound and key not in buf_list:
                logger.warning(f"Extra key in checkpoint: {key}")

    def _get_tp_slices(self, param, base_name, tp_offset):
        """
        Returns a tuple of size 2 or 4 depending on param
        """
        col_start, col_end = 0, param.size(0)
        if base_name in {
            "linear_keys",
            "linear_values",
            "linear_query",
            "gate_up_proj",
            "up_proj",
        }:
            col_start = param.size(0) * tp_offset
            col_end = param.size(0) * (tp_offset + 1)
        if param.dim() == 2:
            row_start, row_end = 0, param.size(1)
            if base_name in {"final_linear", "down_proj"}:
                row_start = param.size(1) * tp_offset
                row_end = param.size(1) * (tp_offset + 1)
            return col_start, col_end, row_start, row_end
        return col_start, col_end

    def _load_param(self, module_name, module, param_name, param, buf_list, ckpt_t, tp_offset):
        base_name = module_name.split(".")[-1]
        is_buffer = f"{module_name}.{param_name}" in buf_list
        is_wq = module.__class__.__name__ == "WQLinear_GEMM"

        # WQLinear_GEMM fix MUST happen before slicing
        if is_wq:
            param.data = param.data.transpose(0, 1)
            ckpt_t = ckpt_t.transpose(0, 1)

        slices = self._get_tp_slices(param, base_name, tp_offset)

        if param.dim() == 2:
            col_start, col_end, row_start, row_end = slices
            sliced = ckpt_t[col_start:col_end, row_start:row_end]
            assert param.size() == sliced.size(), (
                "An error in model's partition and checkpoint's slice was detected, "
                f"[{module_name}, {module}, {param_name}, {param.size()}, {ckpt_t.size()}]"
            )
            if is_buffer:
                if is_wq:
                    module.register_buffer(param_name, sliced.transpose(0, 1).contiguous())
                else:
                    module.register_buffer(param_name, sliced.contiguous())
            else:
                param.data = sliced.contiguous()
        else:
            col_start, col_end = slices
            sliced = ckpt_t[col_start:col_end]
            assert param.size() == sliced.size(), (
                "An error in model's partition and checkpoint's slice was detected, "
                f"[{module_name}, {module}, {param_name}, {param.size()}, {ckpt_t.size()}]"
            )
            if is_buffer:
                module.register_buffer(param_name, sliced.contiguous())
            else:
                param.data = sliced.contiguous()

    def _try_load_parameter(
        self, module_name, module, param_name, param, f, keys_shard, updated_params, buf_list, keyfound, tp_offset
    ):
        """Try to load a parameter from updated_params or keys_shard."""
        # in the case of "adapter.new_line" adapter is not a module hence module_name is ""
        full_name = f"{module_name}.{param_name}" if module_name else f"{param_name}"
        ckpt_t = updated_params.get(full_name)
        if ckpt_t is None and full_name in keys_shard:
            ckpt_t = f[keys_shard[full_name]].get_tensor(full_name)
        if ckpt_t is None:
            return False
        self._load_param(module_name, module, param_name, param, buf_list, ckpt_t, tp_offset)
        keyfound[full_name] = True
        return True

    def _maybe_warn_missing_parameter(self, module_name, param_name, module, buffers_list):
        """Conditionally warn about a missing parameter with helpful context."""
        full_name = f"{module_name}.{param_name}"
        skip_patterns = ["lora", "slopes", "rope"]
        if any(p in full_name or p in param_name for p in skip_patterns):
            return
        logger.info(f"Missing key in safetensors checkpoint: {full_name}")
        if buffers_list and param_name in [n for n, _ in module.named_buffers()]:
            logger.info("└─> Found in buffers, probably ok")
        if full_name in ("generator.weight", "generator.bias") and self.share_decoder_embeddings:
            logger.info("└─> Sharing from embeddings matrix since " "`share_decoder_embeddings` flag is enabled.")
            self.generator.weight = self.tgt_emb.embeddings.weight

    def _load_module_parameters(
        self, module_name, module, f, keys_shard, updated_params, buf_list, keyfound, tp_offset, strict
    ):
        """Load all parameters for a single module"""
        buffers_list = list(module.named_buffers(recurse=False))
        for param_name, param in itertools.chain(
            module.named_buffers(recurse=False), module.named_parameters(recurse=False)
        ):
            loaded = self._try_load_parameter(
                module_name, module, param_name, param, f, keys_shard, updated_params, buf_list, keyfound, tp_offset
            )
            if not loaded and strict:
                self._maybe_warn_missing_parameter(module_name, param_name, module, buffers_list)

    def _load_parameters_by_module(
        self, f, keys_shard, updated_params, buf_list, device, dtype, tp_offset, strict, running_config
    ):
        """Load parameters module by module, moving to device as we go"""
        keyfound = {}
        # Load leaf modules
        for module_name, module in self.named_modules():
            has_children = any(module.children())
            has_own_params = next(module.parameters(recurse=False), None) is not None
            if not has_children or has_own_params:
                self._load_module_parameters(
                    module_name, module, f, keys_shard, updated_params, buf_list, keyfound, tp_offset, strict
                )
                module.to(device=device, dtype=dtype)
                if getattr(running_config, "compute_dtype", None) == torch.int8:
                    torch.quantization.quantize_dynamic(module, inplace=True)
        return keyfound

    def _handle_vocab_updates(self, running_config, vocabs, metadata, f, keys_shard):
        """Handle vocabulary updates when update_vocab flag is set"""
        updated_params = {}
        if not getattr(running_config, "update_vocab", False):
            return updated_params
        updated_params["generator.weight"] = self.generator.state_dict()["weight"]
        updated_params["generator.bias"] = self.generator.state_dict()["bias"]
        # Update embeddings for src and tgt
        emb_names = {"src": "src_emb.embeddings.weight", "tgt": "tgt_emb.embeddings.weight"}
        metadata_vocabs = dict_to_vocabs(metadata["vocab"])
        for side, emb_name in emb_names.items():
            if emb_name not in keys_shard:
                continue
            # Initialize embedding tensor
            updated_params[emb_name] = self.state_dict()[emb_name]
            new_tokens = []
            checkpoint_emb = f[keys_shard[emb_name]].get_tensor(emb_name)
            # Copy embeddings for existing tokens
            for i, tok in enumerate(vocabs[side].ids_to_tokens):
                if tok in metadata_vocabs[side]:
                    old_i = metadata_vocabs[side].lookup_token(tok)
                    updated_params[emb_name][i] = checkpoint_emb[old_i]
                    # Update generator for target side
                    if side == "tgt":
                        generator_weight = f[keys_shard["generator.weight"]].get_tensor("generator.weight")
                        updated_params["generator.weight"][i] = generator_weight[old_i]
                        generator_bias = f[keys_shard["generator.bias"]].get_tensor("generator.bias")
                        updated_params["generator.bias"][i] = generator_bias[old_i]
                else:
                    new_tokens.append(tok)
            logger.info(f"{side}: {len(new_tokens)} new tokens")
        return updated_params

    def _load_safetensors_shards(self, model_path):
        """Load all safetensors shards and build key->shard mapping"""
        try:
            import safetensors
        except ImportError:
            raise ImportError("run: pip install safetensors, to use safetensors")

        shards = glob(os.path.join(model_path, "model.*.safetensors"))
        if len(shards) == 0:
            raise ValueError("No safetensors file found")
        f = []
        keys_shard = {}
        for i, shard in enumerate(shards):
            f.append(safetensors.safe_open(shard, framework="pt", device="cpu"))
            for key in f[i].keys():
                keys_shard[key] = i
        return f, keys_shard

    def load_safe_state_dict(
        self,
        model_path,
        running_config=None,
        vocabs=None,
        metadata=None,
        device=torch.device("cpu"),
        tp_offset=0,
        strict=False,
    ):
        """Custom state_dict loading to enable moving module on device as they are loaded

        Args:
            model_path: Model path
            running_config: RunningConfig (either training or predict)
            vocabs: src/tgt vocabs
            metadata: config/vocabs
            device: device to move parameters to
            tp_offset: for tensor_parallel
            strict: strict loading of all parameters
        """
        # bitsandbytes quantize weights when .cuda() is called
        # for huge models we need to save Ram
        # so we load the weights  module by module and transfer them to GPU for quantization
        dtype = getattr(running_config, "storage_dtype", torch.float32)
        f, keys_shard = self._load_safetensors_shards(model_path)
        if device == torch.device("cpu"):
            tp_offset = 0
        buf_list = [name for name, _ in self.named_buffers()]
        updated_params = self._handle_vocab_updates(running_config, vocabs, metadata, f, keys_shard)
        keyfound = self._load_parameters_by_module(
            f, keys_shard, updated_params, buf_list, device, dtype, tp_offset, strict, running_config
        )
        self._report_extra_keys(keys_shard, keyfound, buf_list)
        self._reset_lora_to_fp32()

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
        # Not sure if there is a better way to handle this
        model_config.decoder.with_cross_attn = True
        decoder = build_decoder(model_config, running_config=running_config)
        return cls(
            encoder=encoder,
            decoder=decoder,
            src_emb=src_emb,
            tgt_emb=tgt_emb,
            add_estimator=model_config.add_estimator,
            hidden_size=model_config.decoder.hidden_size,
        )
        # from there, the base blocks exist, and the rest is done in the from_opt from base class

    def forward(self, src, tgt, src_len, with_align=False, **kwargs):
        """An EncoderDecoderModel forward the src side to the encoder.
        Then the output of encoder ``enc_out`` is forwarded to the
        decoder along with the target excluding the last token.
        The decoder state is initiliazed with:
        * enc_final_hs in the case of RNNs
        * enc_out + enc_final_hs in the case of CNNs
        * src in the case of Transformer"""
        src_pad_mask = src.eq(self.src_pad_idx).unsqueeze(1)  # [B, 1, T_src]
        enc_out, enc_final_hs = self.encoder(
            self.src_emb(src),
            pad_mask=src_pad_mask,
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
        )

        if self.add_estimator:  # we take the average of dec_out using the pad mask
            if self.estimator_type == "average":
                pad_mask2 = ~dec_in.eq(self.tgt_pad_idx)
                in_estim2 = (dec_out * pad_mask2.unsqueeze(-1).float()).sum(dim=1) / pad_mask2.sum(
                    dim=1, keepdim=True
                ).float()
            elif self.estimator == "last_token":
                # assumes left padding so -1 is the real last token (not eos)
                in_estim2 = dec_out[:, -1, :]
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
        decoder = build_decoder(model_config, running_config=running_config)
        return cls(
            decoder=decoder,
            tgt_emb=tgt_emb,
            add_estimator=model_config.add_estimator,
            hidden_size=model_config.decoder.hidden_size,
        )
        # from there, the base blocks exist, and the rest is done in the from_opt from base class

    def forward(self, src, _, src_len, with_align=False, **kwargs):
        """A DecoderModel forward the src side to the decoder along
        with the source lengths vector. It is a decoder only LM (cf GPT-2)"""

        self.decoder.init_state()
        dec_out, attns = self.decoder(
            self.tgt_emb(src),
            tgt_pad_mask=src.eq(self.pad_idx).unsqueeze(1),
            **kwargs,
        )

        if self.add_estimator:  # we take the average of dec_out using the pad mask
            if self.estimator_type == "average":
                pad_mask2 = ~src.eq(self.pad_idx)
                in_estim2 = (dec_out * pad_mask2.unsqueeze(-1).float()).sum(dim=1) / pad_mask2.sum(
                    dim=1, keepdim=True
                ).float()
            elif self.estimator_type == "last_token":
                # assumes left padding so -1 is the real last token (not eos)
                in_estim2 = dec_out[:, -1, :]
            else:
                raise ValueError("Estimator type should be average or last token")
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
        )
        # from there, the base blocks exist, and the rest is done in the from_opt from base class

    def forward(self, src, _, src_len, with_align=False, **kwargs):
        """An EncoderModel encodes the source sentence to build hidden states"""

        pad_mask = src.eq(self.pad_idx).unsqueeze(1)  # [B, 1, T_src]
        enc_out, enc_final_hs = self.encoder(
            self.src_emb(src),
            pad_mask=pad_mask,
        )
        if self.add_estimator:
            if self.estimator_type == "average":
                pad_mask1 = ~src.eq(self.pad_idx)
                in_estim1 = (enc_out * pad_mask1.unsqueeze(-1).float()).sum(dim=1) / pad_mask1.sum(
                    dim=1, keepdim=True
                ).float()
            elif self.estimator_type == "first_token":
                in_estim1 = enc_out[:, 0, :]
            else:
                raise ValueError("Estimator type should be average or first token")
            estim = self.estimator(in_estim1.to(enc_out.dtype)).squeeze(-1)
        else:
            estim = None

        return enc_out, enc_final_hs, estim

    def update_dropout(self, dropout, attention_dropout):
        self.encoder.update_dropout(dropout, attention_dropout)


class VisionEncoderDecoderModel(BaseModel):
    """VisionEncoderDecoderModel Class
    See :class:`~eole.models.BaseModel` for options."""

    def __init__(self, **kwargs):
        super(VisionEncoderDecoderModel, self).__init__(**kwargs)
        self.tgt_shift = 1
        self.image_token_id = kwargs.get("image_token_id", None)
        self.patch_size = kwargs.get("patch_size", None)
        self.spatial_merge_size = kwargs.get("spatial_merge_size", None)
        if self.encoder is None or self.decoder is None:
            raise ValueError("A EncoderDecoderModel requires both an Encoder and a Decoder")
        if self.encoder.sam is not None:
            embed_std = 1 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
            self.image_newline = nn.Parameter(torch.randn(self.hidden_size) * embed_std)
            self.view_separator = nn.Parameter(torch.randn(self.hidden_size) * embed_std)
        if self.add_estimator:
            self.estimator = FeedForward(self.hidden_size)

    @classmethod
    def build_blocks(cls, model_config, vocabs, running_config=None):
        encoder = build_encoder(model_config, running_config=running_config)
        adapter = build_adapter(model_config, running_config=running_config)
        tgt_emb = build_tgt_emb(
            model_config,
            vocabs,
            running_config=running_config,
            share_embeddings=model_config.share_embeddings,
        )
        decoder = build_decoder(model_config, running_config=running_config)
        return cls(
            encoder=encoder,
            decoder=decoder,
            adapter=adapter,
            tgt_emb=tgt_emb,
            add_estimator=model_config.add_estimator,
            hidden_size=model_config.decoder.hidden_size,
            image_token_id=model_config.encoder.image_token_id,
            patch_size=model_config.encoder.patch_size,
            spatial_merge_size=model_config.spatial_merge_size,
        )
        # from there, the base blocks exist, and the rest is done in the from_opt from base class

    def build_position_ids(self, src, image_locations, image_sizes):
        """
        src: [B, L]
        image_locations: bool mask of same shape as src
        image_sizes: [num_images, 2] with (height_px, width_px)

        This part is specific to HunyunOCR for several reasons:
        - it detects continuous sequences of image_token, so does not work with llava (IMG_BREAK)
        - it supposed one image_token as new line sep (W+1)
        - it adds 2 image_token at the end: H * (W+1) + 2

        """
        B, L = src.shape
        device = src.device
        position_ids = torch.zeros((B, L, 4), device=device, dtype=torch.long)
        ar = torch.arange(L, device=device)
        position_ids[:, :, 0] = ar
        position_ids[:, :, 1] = ar
        position_ids[:, :, 2] = ar
        position_ids[:, :, 3] = 0

        img_ptr = 0  # pointer into image_sizes list

        # --- For each batch item ---
        for b in range(B):
            mask = image_locations[b]  # [L]
            if not mask.any():
                continue
            # ---------- find contiguous blocks of True ----------
            idxs = mask.nonzero(as_tuple=False).squeeze(1)  # sorted positions
            # difference = 1 means contiguous
            breaks = (idxs[1:] != idxs[:-1] + 1).nonzero(as_tuple=False).squeeze(1)
            starts = torch.cat([idxs[:1], idxs[breaks + 1]])
            ends = torch.cat([idxs[breaks], idxs[-1:]])
            for start, end in zip(starts.tolist(), ends.tolist()):
                length = end - start + 1
                H, W = image_sizes[img_ptr] // (self.patch_size * self.spatial_merge_size)
                img_ptr += 1
                HW = H * (W + 1) + 2  # SPECIFIC to HunyuanOCR
                if HW != length:
                    raise ValueError(f"image tokens={length} but H*W={HW} at image {img_ptr-1}")
                r = torch.arange(length, device=device)
                w = r % (W + 1)  # SPECIFIC to HunyuanOCR
                h = r // (W + 1)  # SPECIFIC to HunyuanOCR
                position_ids[b, start : end + 1, 1] = w
                position_ids[b, start : end + 1, 2] = h
                position_ids[b, start : end + 1, 3] = 0
        return position_ids

    def embed_vision_language_features(self, src, **kwargs):
        batch_size, seq_len = src.size()
        images = kwargs.get("images", None)
        if images is None or len(images) == 0:
            return self.tgt_emb(src), None

        text_locations = src != self.image_token_id
        image_locations = src == self.image_token_id
        text_features = self.tgt_emb(src[text_locations])
        image_sizes = torch.tensor([[images[i].size(1), images[i].size(2)] for i in range(len(images))])

        if self.encoder.sam is not None:
            sam_patches, _ = self.encoder.sam(images)  # tensor B C H W
            encoded_images, _ = self.encoder(images, sam_patches=sam_patches)
            global_features = torch.cat((encoded_images[:, 1:], sam_patches.flatten(2).permute(0, 2, 1)), dim=-1)
            image_features = self.adapter(global_features, image_sizes=image_sizes)
            b, hw, n_dim = image_features.shape
            h = w = int(hw**0.5)
            image_features = image_features.view(b, h, w, n_dim)
            # Add newline token at the end of each row → shape (b, h, w+1, n_dim)
            newline = self.image_newline.view(1, 1, 1, n_dim).expand(b, h, 1, n_dim)
            image_features = torch.cat([image_features, newline], dim=2)
            # Flatten back to (b, h*(w+1), n_dim)
            image_features = image_features.view(b, -1, n_dim)
            separator = self.view_separator.view(1, 1, n_dim).expand(b, 1, n_dim)
            image_features = torch.cat([image_features, separator], dim=1)
        else:
            encoded_images, _ = self.encoder(images)
            image_features = self.adapter(encoded_images, image_sizes=image_sizes)

        N_img, tokperimg, D_img = image_features.shape
        N_txt = text_features.size(0)

        assert text_features.size(-1) == D_img
        assert batch_size * seq_len == N_txt + N_img * tokperimg, (
            f"seq_len {seq_len} should be equal to N_txt + N_img * tokperimg "
            f"{(N_txt, N_img, tokperimg, image_locations.sum().item())}"
        )

        combined_features = torch.empty(
            (batch_size, seq_len, D_img),
            dtype=text_features.dtype,
            device=text_features.device,
        )
        text_mask = text_locations.unsqueeze(-1).expand_as(combined_features)
        combined_features = combined_features.masked_scatter(text_mask, text_features)
        image_mask = image_locations.unsqueeze(-1).expand_as(combined_features)
        combined_features = combined_features.masked_scatter(image_mask, image_features.view(-1, D_img))

        # TODO: Revisit this when implementing real mRope for Qwen VL (3 sections). This is a temporary solution
        # and may not generalize to other vision-language models with different position encoding schemes.
        if self.adapter.__class__.__name__ == "HunYuanVisionPatchMerger":
            position_ids = self.build_position_ids(src, image_locations, image_sizes)
        else:
            position_ids = None

        return combined_features, position_ids

    def forward(self, src, tgt, src_len, with_align=False, **kwargs):
        """A DecoderModel forward the src side to the decoder along
        with the source lengths vector. It is a decoder only LM (cf GPT-2)"""

        self.decoder.init_state()
        emb, pos_ids = self.embed_vision_language_features(src, **kwargs)
        dec_in = tgt[:, :-1]
        pad_idx = self.tgt_emb.word_padding_idx
        pad_mask = src.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
        dec_out, attns = self.decoder(
            emb,
            enc_out=None,
            src_len=src_len,
            with_align=with_align,
            tgt_pad_mask=pad_mask,
            pos_ids=pos_ids,
        )

        if self.add_estimator:  # we take the average of dec_out using the pad mask
            pad_mask2 = ~dec_in.eq(pad_idx)
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


def get_model_class(model_config):
    # might have more cases later
    if model_config.decoder is None:
        return EncoderModel
    elif model_config.encoder is None:
        return DecoderModel
    elif model_config.encoder.encoder_type == "vision":
        return VisionEncoderDecoderModel
    else:
        return EncoderDecoderModel
