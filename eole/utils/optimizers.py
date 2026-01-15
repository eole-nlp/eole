"""Optimizers class"""

from __future__ import annotations

import functools
import os
from math import cos, pi, sqrt
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer as TorchOptimizer

try:
    import optimi

    OPTIMI_AVAILABLE = True
except ImportError:
    OPTIMI_AVAILABLE = False


def build_torch_optimizer(model: Module, config: Any) -> TorchOptimizer:
    """Builds the PyTorch optimizer.

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf

    Args:
        model: The model to optimize.
        config: The configuration object with optimizer settings.

    Returns:
        A torch.optim.Optimizer instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    betas = (config.adam_beta1, config.adam_beta2)  # Use tuple instead of list

    # Use optimi if available and not using AMP (optimi doesn't support AMP well)
    optim_module = torch.optim if (config.use_amp or not OPTIMI_AVAILABLE) else optimi

    # Dictionary mapping optimizer names to their configurations
    optimizer_configs = {
        "sgd": lambda: optim_module.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay),
        "adagrad": lambda: torch.optim.Adagrad(
            params,
            lr=config.learning_rate,
            initial_accumulator_value=config.adagrad_accumulator_init,
            weight_decay=config.weight_decay,
        ),
        "adadelta": lambda: torch.optim.Adadelta(params, lr=config.learning_rate, weight_decay=config.weight_decay),
        "adafactor": lambda: torch.optim.Adafactor(
            params,
            lr=config.learning_rate,
            beta2_decay=config.adafactor_beta2,
            eps=config.adafactor_eps,
            d=config.adafactor_d,
            weight_decay=config.weight_decay,
        ),
        "adam": lambda: optim_module.Adam(
            params,
            lr=config.learning_rate,
            betas=betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        ),
        "adamw": lambda: optim_module.AdamW(
            params,
            lr=config.learning_rate,
            betas=betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            foreach=False,  # Can be True for potential speedup
        ),
    }

    # Handle special optimizers
    if config.optim == "sparseadam":
        return _build_sparse_adam_optimizer(model, config, betas, optim_module)
    elif config.optim in ["adamw8bit", "pagedadamw8bit", "pagedadamw32bit"]:
        return _build_bnb_optimizer(params, config, betas)
    elif config.optim in optimizer_configs:
        return optimizer_configs[config.optim]()
    else:
        raise ValueError(f"Invalid optimizer type: {config.optim}")


def _build_sparse_adam_optimizer(
    model: Module, config: Any, betas: tuple[float, float], optim_module: Any
) -> MultipleOptimizer:
    """Build sparse Adam optimizer for embeddings."""
    dense_params = []
    sparse_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # TODO: Find a better way to check for sparse gradients
        if "embed" in name:
            sparse_params.append(param)
        else:
            dense_params.append(param)

    return MultipleOptimizer(
        [
            optim_module.Adam(dense_params, lr=config.learning_rate, betas=betas, eps=config.adam_eps),
            torch.optim.SparseAdam(sparse_params, lr=config.learning_rate, betas=betas, eps=config.adam_eps),
        ]
    )


def _build_bnb_optimizer(params: list, config: Any, betas: tuple[float, float]) -> Any:
    """Build bitsandbytes optimizer."""
    try:
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError("Install bitsandbytes to use bnb optimizers: " "pip install bitsandbytes") from e

    optimizer_map = {
        "adamw8bit": bnb.optim.AdamW8bit,
        "pagedadamw8bit": bnb.optim.PagedAdamW8bit,
        "pagedadamw32bit": bnb.optim.PagedAdamW32bit,
    }

    optimizer_class = optimizer_map.get(config.optim)
    if optimizer_class is None:
        raise ValueError(f"Invalid optimizer type: {config.optim}")

    # Common arguments for all bnb optimizers
    common_args = {
        "lr": config.learning_rate,
        "betas": betas,
        "eps": config.adam_eps,
        "weight_decay": config.weight_decay,
        "amsgrad": False,
        "percentile_clipping": 100,
        "block_wise": True,
    }

    # Specific arguments based on optimizer type
    if config.optim == "adamw8bit":
        common_args.update(
            {
                "optim_bits": 8,
                "args": None,
                "min_8bit_size": 1024,
                "is_paged": False,
            }
        )
    else:  # paged variants
        common_args.update(
            {
                "optim_bits": 8 if "8bit" in config.optim else 32,
                "args": None,
                "min_8bit_size": 4096,
            }
        )

    return optimizer_class(params, **common_args)


def make_learning_rate_decay_fn(config: Any, running_config: Optional[Any] = None) -> Optional[Callable[[int], float]]:
    """Returns the learning rate decay function from config.

    Args:
        config: The full configuration object.
        running_config: Optional override for training config (used when loading checkpoints).
    """
    model_config = config.model
    if running_config is None:
        running_config = config.training

    decay_functions = {
        "noam": functools.partial(
            noam_decay,
            warmup_steps=running_config.warmup_steps,
            model_size=model_config.hidden_size,
        ),
        "noamwd": functools.partial(
            noamwd_decay,
            warmup_steps=running_config.warmup_steps,
            model_size=model_config.hidden_size,
            rate=running_config.learning_rate_decay,
            decay_steps=running_config.decay_steps,
            start_step=running_config.start_decay_steps,
        ),
        "cosine": functools.partial(
            cosine_decay,
            warmup_steps=running_config.warmup_steps,
            train_steps=running_config.train_steps,
        ),
        "rsqrt": functools.partial(rsqrt_decay, warmup_steps=running_config.warmup_steps),
    }

    if running_config.decay_method in decay_functions:
        return decay_functions[running_config.decay_method]
    elif running_config.start_decay_steps is not None:
        return functools.partial(
            exponential_decay,
            rate=running_config.learning_rate_decay,
            decay_steps=running_config.decay_steps,
            start_step=running_config.start_decay_steps,
        )

    # Return None if no decay method is configured
    return None


def noam_decay(step: int, warmup_steps: int, model_size: int) -> float:
    """Learning rate schedule from 'Attention Is All You Need'.

    https://arxiv.org/pdf/1706.03762.pdf
    """
    return model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def noamwd_decay(
    step: int, warmup_steps: int, model_size: int, rate: float, decay_steps: int, start_step: int = 0
) -> float:
    """Learning rate schedule optimized for large batches."""
    base_rate = model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    decay_factor = rate ** (max(step - start_step + decay_steps, 0) // decay_steps)
    return base_rate * decay_factor


def cosine_decay(step: int, warmup_steps: int, train_steps: int) -> float:
    """Cosine annealing learning rate schedule."""
    if step < warmup_steps:
        return step / warmup_steps

    decay_ratio = (step - warmup_steps) / (train_steps - warmup_steps)
    return 0.5 * (1.0 + cos(pi * decay_ratio))


def exponential_decay(step: int, rate: float, decay_steps: int, start_step: int = 0) -> float:
    """Standard exponential decay.

    Scales the learning rate by `rate` every `decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step: int, warmup_steps: int) -> float:
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))


class MultipleOptimizer:
    """Wrapper for multiple optimizers (used for sparse Adam)."""

    def __init__(self, optimizers: list[TorchOptimizer]) -> None:
        """Initialize with list of optimizers."""
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list:
        """Get all parameter groups from all optimizers."""
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Perform optimization step for all optimizers."""
        for optimizer in self.optimizers:
            optimizer.step()

    @property
    def state(self) -> dict:
        """Get combined state from all optimizers."""
        return {k: v for optimizer in self.optimizers for k, v in optimizer.state.items()}

    def state_dict(self) -> list[dict]:
        """Get state dicts from all optimizers."""
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dicts: list[dict]) -> None:
        """Load state dicts into optimizers."""
        if len(state_dicts) != len(self.optimizers):
            raise ValueError(f"Expected {len(self.optimizers)} state_dicts, " f"got {len(state_dicts)}")
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)


class Optimizer:
    """Optimizer wrapper with learning rate scheduling and gradient scaling.

    Wraps a torch.optim.Optimizer with additional functionality:
    - Learning rate scheduling
    - Gradient clipping
    - Automatic mixed precision (AMP) support with gradient scaling

    Args:
        optimizer: A torch.optim.Optimizer instance.
        learning_rate: The initial learning rate.
        learning_rate_decay_fn: Optional callable for LR scheduling.
        max_grad_norm: Clip gradients to this global norm (0 = no clipping).
        use_amp: Whether to use automatic mixed precision.
    """

    def __init__(
        self,
        optimizer: TorchOptimizer,
        learning_rate: float,
        learning_rate_decay_fn: Optional[Callable[[int], float]] = None,
        max_grad_norm: Optional[float] = None,
        use_amp: bool = True,
    ) -> None:
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._training_step = 1
        self._decay_step = 1
        self.use_amp = use_amp
        self._scaler: Optional[GradScaler] = None

    @classmethod
    def from_config(cls, model: Module, config: Any, metadata: Optional[dict] = None) -> Optimizer:
        """Build optimizer from configuration.

        Args:
            model: The model to optimize.
            config: The configuration object.
            metadata: Optional checkpoint metadata to load states from.

        Returns:
            An Optimizer instance.
        """
        running_config = config.training
        optim_state_dict = None

        # Handle checkpoint loading and potentially update running_config
        if running_config.train_from and metadata is not None and "optim" in metadata:
            optim_state_dict, running_config = cls._process_checkpoint(metadata, running_config)

        # Determine if AMP should be used
        use_amp = running_config.use_amp and running_config.compute_dtype in [torch.float16, torch.bfloat16]

        optimizer = cls(
            build_torch_optimizer(model, running_config),
            running_config.learning_rate,
            learning_rate_decay_fn=make_learning_rate_decay_fn(config, running_config),
            max_grad_norm=running_config.max_grad_norm,
            use_amp=use_amp,
        )

        # Initialize GradScaler for AMP
        if use_amp:
            # Modern API: torch.amp.GradScaler with device specification
            optimizer._scaler = GradScaler("cuda")

        if optim_state_dict:
            optimizer.load_state_dict(optim_state_dict)

        return optimizer

    @staticmethod
    def _process_checkpoint(metadata: dict, running_config: Any) -> tuple[Optional[dict], Any]:
        """Process checkpoint metadata for optimizer state loading.

        Returns:
            Tuple of (optim_state_dict, config_to_use)
        """
        optim = metadata["optim"]
        ckpt_config = metadata["config"].training

        # Handle legacy format
        if isinstance(optim, Optimizer):
            ckpt_state_dict = {
                "training_step": optim._step + 1,
                "decay_step": optim._step + 1,
                "optimizer": optim.optimizer.state_dict(),
            }
        else:
            ckpt_state_dict = optim

        # Handle reset options
        reset_option = running_config.reset_optim

        if reset_option == "none":
            # Load everything from checkpoint including config
            return ckpt_state_dict, ckpt_config
        elif reset_option == "all":
            # Build from scratch with new config
            return None, running_config
        elif reset_option == "states":
            # Reset optimizer, but keep options from checkpoint
            result = ckpt_state_dict.copy()
            result.pop("optimizer", None)
            return result, ckpt_config
        elif reset_option == "keep_states":
            # Reset options, keep optimizer state
            return {"optimizer": ckpt_state_dict.get("optimizer")}, running_config

        return None, running_config

    @property
    def training_step(self) -> int:
        """The current training step."""
        return self._training_step

    @property
    def amp(self) -> bool:
        """Whether using automatic mixed precision."""
        return self.use_amp

    def learning_rate(self, step: Optional[int] = None) -> float:
        """Calculate current learning rate.

        Args:
            step: Step to calculate LR for (defaults to current decay_step).

        Returns:
            The learning rate value.
        """
        if step is None:
            step = self._decay_step

        if self._learning_rate_decay_fn is None:
            return self._learning_rate

        scale = self._learning_rate_decay_fn(step)
        return scale * self._learning_rate

    def state_dict(self) -> dict[str, Any]:
        """Get optimizer state for checkpointing."""
        return {
            "training_step": self._training_step,
            "decay_step": self._decay_step,
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state from checkpoint."""
        self._training_step = state_dict["training_step"]

        if "decay_step" in state_dict:
            self._decay_step = state_dict["decay_step"]

        if "optimizer" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer"])

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the gradients of optimized parameters.

        Args:
            set_to_none: Set gradients to None instead of zero for memory efficiency.
        """
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass with optional gradient scaling.

        Args:
            loss: The loss tensor to backpropagate.
        """
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self) -> None:
        """Update model parameters based on gradients.

        Handles learning rate updates, gradient clipping, and AMP scaling.
        """
        learning_rate = self.learning_rate()

        # Unscale gradients if using AMP
        if self._scaler is not None:
            self._scaler.unscale_(self._optimizer)

        # Update learning rate and apply gradient clipping
        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
            if self._max_grad_norm > 0:
                clip_grad_norm_(group["params"], self._max_grad_norm)

        # Perform optimizer step with optional AMP scaling
        if self._scaler is not None:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()

        self._decay_step += 1
        self._training_step += 1
