import torch.nn as nn


class DecoderBase(nn.Module):
    """Abstract class for decoders.

    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional
        # Decoder state
        self.state = {}

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor.

        Subclasses should override this method.
        """

        raise NotImplementedError

    def init_state(self, **kwargs):
        raise NotImplementedError

    def map_state(self, fn):
        raise NotImplementedError

    def detach_state(self):
        raise NotImplementedError
