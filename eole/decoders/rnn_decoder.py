import torch
import torch.nn as nn

from eole.decoders.decoder import DecoderBase
from eole.modules.stacked_rnn import StackedLSTM, StackedGRU
from eole.modules.gate import context_gate_factory
from eole.modules.global_attention import GlobalAttention


class RNNDecoderBase(DecoderBase):
    """Base recurrent attention-based decoder class.

    Specifies the interface used by different decoder types
    and required by :class:`~eole.models.BaseModel`.

    Args:
        model_config (eole.config.DecoderConfig): full decoder config
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        super(RNNDecoderBase, self).__init__(
            attentional=model_config.global_attention != "none"
            and model_config.global_attention is not None
        )

        self.bidirectional_encoder = model_config.bidirectional_encoder
        self.num_layers = model_config.layers
        self.dropout = nn.Dropout(getattr(running_config, "dropout", [0.0])[0])

        # Build the RNN.
        self.rnn = self._build_rnn(
            model_config.rnn_type,
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=getattr(running_config, "dropout", [0.0])[0],
        )

        # Set up the context gate.
        self.context_gate = None
        if model_config.context_gate is not None:
            self.context_gate = context_gate_factory(
                model_config.context_gate,
                self._input_size,
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
            )

        # Set up the standard attention.
        self._coverage = model_config.coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.attn = GlobalAttention(
                self.hidden_size,
                coverage=model_config.coverage_attn,
                attn_type=model_config.global_attention,
                attn_func=model_config.global_attention_function,
            )

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor."""
        # config = opt.model.decoder  # RnnDecoderConfig
        return cls(
            model_config,
            running_config=running_config,
        )

    def init_state(self, **kwargs):
        """Initialize decoder state with last state of the encoder."""
        enc_final_hs = kwargs.pop("enc_final_hs", None)

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat(
                    [hidden[0 : hidden.size(0) : 2], hidden[1 : hidden.size(0) : 2]], 2
                )
            return hidden

        if isinstance(enc_final_hs, tuple):  # LSTM
            self.state["hidden"] = tuple(
                _fix_enc_hidden(enc_hid) for enc_hid in enc_final_hs
            )
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(enc_final_hs),)

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)

        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = (
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        )

        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(
            fn(h.transpose(0, 1), 0).transpose(0, 1) for h in self.state["hidden"]
        )
        self.state["input_feed"] = fn(
            self.state["input_feed"].transpose(0, 1), 0
        ).transpose(0, 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(
                self.state["coverage"].transpose(0, 1), 0
            ).transpose(0, 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = self.state["coverage"].detach()

    def forward(self, emb, enc_out, src_len=None, step=None, **kwargs):
        """
        Args:
            emb (FloatTensor): input embeddings
                 ``(batch, tgt_len, dim)``.
            enc_out (FloatTensor): vectors from the encoder
                 ``(batch, src_len, hidden)``.
            src_len (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(batch, tgt_len, hidden)``.
            * attns: distribution over src at each tgt
              ``(batch, tgt_len, src_len)``.
        """
        dec_state, dec_outs, attns = self._run_forward_pass(
            emb, enc_out, src_len=src_len
        )

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       since stack(Variable) was allowed.
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs, dim=1)
            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])

        self.state["input_feed"] = dec_outs[:, -1, :].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1, :, :].unsqueeze(0)

        return dec_outs, attns

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
        model_config,
        running_config=None,
    ):
        self.hidden_size = model_config.hidden_size
        self._input_size = model_config.tgt_word_vec_size
        super(StdRNNDecoder, self).__init__(model_config, running_config)

    def _run_forward_pass(self, emb, enc_out, src_len=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.

        Args:
            emb (FloatTensor): input embeddings
                ``(batch, tgt_len, dim)``.
            enc_out (FloatTensor): output(tensor sequence) from the
                encoder RNN of size ``(batch, src_len, hidden_size)``.
            src_len (LongTensor): the source enc_out lengths.

        Returns:
            (Tensor, List[FloatTensor], Dict[str, List[FloatTensor]):

            * dec_state: final hidden state from the decoder.
            * dec_outs: an array of output of every time
              step from the decoder.
            * attns: a dictionary of different
              type of attention Tensor array of every time
              step from the decoder.
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
            dec_outs, p_attn = self.attn(rnn_out, enc_out, src_len=src_len)
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
        model_config,
        running_config=None,
    ):
        self.hidden_size = model_config.hidden_size
        self._input_size = model_config.tgt_word_vec_size + self.hidden_size
        super(InputFeedRNNDecoder, self).__init__(model_config, running_config)

    def _run_forward_pass(self, emb, enc_out, src_len=None):
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

        coverage = (
            self.state["coverage"].squeeze(0)
            if self.state["coverage"] is not None
            else None
        )

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1, dim=1):
            dec_in = torch.cat([emb_t.squeeze(1), input_feed], 1)
            rnn_out, dec_state = self.rnn(dec_in, dec_state)
            if self.attentional:
                dec_out, p_attn = self.attn(rnn_out, enc_out, src_len=src_len)
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
