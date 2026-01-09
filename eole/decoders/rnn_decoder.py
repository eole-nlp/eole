from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from eole.decoders.decoder import DecoderBase
from eole.modules.stacked_rnn import StackedLSTM, StackedGRU
from eole.modules.contextgate import context_gate_factory
from eole.modules.global_attention import GlobalAttention


class RNNDecoderBase(DecoderBase):
    """
    Base class for recurrent neural network decoders.

    Implements common logic for:
    - state initialization and mapping
    - incremental decoding
    - attention handling
    - output and attention normalization

    Subclasses must implement `_run_forward_pass`.
    """

    def __init__(
        self,
        decoder_config,
        running_config=None,
    ):
        super(RNNDecoderBase, self).__init__(
            attentional=decoder_config.global_attention != "none" and decoder_config.global_attention is not None
        )

        self.bidirectional_encoder = decoder_config.bidirectional_encoder
        self.num_layers = decoder_config.layers
        self.dropout = nn.Dropout(getattr(running_config, "dropout", [0.0])[0])

        # Build the RNN.
        self.rnn = self._build_rnn(
            decoder_config.rnn_type,
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=getattr(running_config, "dropout", [0.0])[0],
        )

        # Set up the context gate.
        self.context_gate = None
        if decoder_config.context_gate is not None:
            self.context_gate = context_gate_factory(
                decoder_config.context_gate,
                self._input_size,
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
            )

        # Set up the standard attention.
        self._coverage = decoder_config.coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.attn = GlobalAttention(
                self.hidden_size,
                coverage=decoder_config.coverage_attn,
                attn_type=decoder_config.global_attention,
                attn_func=decoder_config.global_attention_function,
            )

    def init_state(self, **kwargs):
        """Initialize decoder state with last state of the encoder."""
        enc_final_hs = kwargs.pop("enc_final_hs", None)

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0 : hidden.size(0) : 2], hidden[1 : hidden.size(0) : 2]], 2)
            return hidden

        if isinstance(enc_final_hs, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid) for enc_hid in enc_final_hs)
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(enc_final_hs),)

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)

        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)

        self.state["coverage"] = None

    def map_state(self, fn):
        # map_state applies on dim=0 (batch_size)
        # TODO refactor RNN states so that batch_size is natively on dim=0
        self.state["hidden"] = tuple(fn(h.transpose(0, 1)).transpose(0, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"].transpose(0, 1)).transpose(0, 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(self.state["coverage"].transpose(0, 1)).transpose(0, 1)

    def forward(
        self,
        emb: torch.Tensor,
        enc_out: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode a full sequence or a single incremental step.

        Args:
            emb (Tensor):
                Target embeddings of shape
                ``(batch_size, tgt_len, hidden_size)``.

            enc_out (Tensor):
                Encoder outputs of shape
                ``(batch_size, src_len, hidden_size)``.

            step (int, optional):
                Decoding step for incremental decoding.

            **kwargs:
                Additional decoder-specific arguments.

        Returns:
            (Tensor, Dict[str, Tensor]):

            * dec_outs:
                Decoder outputs of shape
                ``(batch_size, tgt_len, hidden_size)``.

            * attns:
                Dictionary of attention tensors.

                - ``attns["std"]``:
                  Attention weights of shape
                  ``(batch_size, tgt_len, src_len)``.
        """
        if self.attentional and enc_out is None:
            raise ValueError("enc_out required when attentional=True")

        src_pad_mask = kwargs.pop("src_pad_mask", None)

        dec_state, dec_outs, attns = self._run_forward_pass(emb, enc_out, src_pad_mask=src_pad_mask)

        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state

        if isinstance(dec_outs, list):
            dec_outs = torch.stack(dec_outs, dim=1)
            for k in attns:
                if isinstance(attns[k], list):
                    attns[k] = torch.stack(attns[k], dim=1)

        self.state["input_feed"] = dec_outs[:, -1].unsqueeze(0)

        if self._coverage:
            self.state["coverage"] = attns["coverage"][:, -1].unsqueeze(0) if "coverage" in attns else None

        return dec_outs, attns if self.attentional else {}

    def update_dropout(self, dropout, attention_dropout=None):
        self.dropout.p = dropout


class StdRNNDecoder(RNNDecoderBase):
    """Standard fully batched RNN decoder with attention.

    Faster implementation, uses CuDNN for implementation.
    See :class:`~eole.decoders.decoder.RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    """

    def __init__(
        self,
        decoder_config,
        running_config=None,
    ):
        self.hidden_size = decoder_config.hidden_size
        self._input_size = decoder_config.tgt_word_vec_size
        super(StdRNNDecoder, self).__init__(decoder_config, running_config)

    def _run_forward_pass(self, emb, enc_out, src_pad_mask=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.

        Args:
            emb (FloatTensor): input embeddings
                ``(batch, tgt_len, dim)``.
            enc_out (FloatTensor): output(tensor sequence) from the
                encoder RNN of size ``(batch, src_len, hidden_size)``.
            src_pad_mask (LongTensor): the source pad mask.

        Returns:
            dec_state: tuple[Tensor]
            dec_outs: Tensor (batch, tgt_len, hidden)
            attns["std"]: Tensor (batch, tgt_len, src_len)
        """

        assert not self._coverage  # TODO, no support yet.

        attns = {}

        if isinstance(self.rnn, nn.GRU):
            rnn_out, dec_state = self.rnn(emb, self.state["hidden"][0])
        else:
            rnn_out, dec_state = self.rnn(emb, self.state["hidden"])

        tgt_batch, tgt_len, _ = emb.size()

        # Calculate the attention.
        if not self.attentional:
            dec_outs = rnn_out
        else:
            dec_outs, p_attn = self.attn(rnn_out, enc_out, src_pad_mask=src_pad_mask)
            attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            dec_outs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_out.view(-1, rnn_out.size(2)),
                dec_outs.view(-1, dec_outs.size(2)),
            )
            dec_outs = dec_outs.view(tgt_batch, tgt_len, self.hidden_size)

        dec_outs = self.dropout(dec_outs)

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn = getattr(nn, rnn_type)(batch_first=True, **kwargs)
        return rnn


class InputFeedRNNDecoder(RNNDecoderBase):
    """Input feeding based decoder.

    See :class:`~eole.decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`

    """

    def __init__(
        self,
        decoder_config,
        running_config=None,
    ):
        self.hidden_size = decoder_config.hidden_size
        self._input_size = decoder_config.tgt_word_vec_size + self.hidden_size
        super(InputFeedRNNDecoder, self).__init__(decoder_config, running_config)

    def _run_forward_pass(self, emb, enc_out, src_pad_mask=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
        if self._coverage:
            attns["coverage"] = []

        assert emb.dim() == 3  # batch x len x embedding_dim

        dec_state = self.state["hidden"]

        coverage = self.state["coverage"].squeeze(0) if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1, dim=1):
            dec_in = torch.cat([emb_t.squeeze(1), input_feed], 1)
            rnn_out, dec_state = self.rnn(dec_in, dec_state)
            if self.attentional:
                dec_out, p_attn = self.attn(rnn_out, enc_out, src_pad_mask=src_pad_mask)
                attns["std"].append(p_attn)
            else:
                dec_out = rnn_out
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                dec_out = self.context_gate(dec_in, rnn_out, dec_out)
            dec_out = self.dropout(dec_out)
            input_feed = dec_out

            dec_outs += [dec_out]

            # Update the coverage attention.
            # attns["coverage"] is actually c^(t+1) of See et al(2017)
            # 1-index shifted
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns["coverage"] += [coverage]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    def update_dropout(self, dropout, attention_dropout=None):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
