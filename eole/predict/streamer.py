"""Text streaming utilities for token-by-token generation output."""

import queue
import threading
from typing import Optional


class GenerationStreamer:
    """Streamer for token-by-token generation output.

    Tokens are put into a thread-safe queue by the generation loop and
    can be consumed as a Python iterator. The streamer handles incremental
    detokenization so that consumers receive human-readable text chunks.

    This is primarily designed for use with ``GeneratorLM`` (decoder-only
    LLM models). For best results, use with ``batch_size=1``.

    Args:
        vocabs (dict): Vocabulary dictionaries from the model.
        transform_pipe (TransformPipe, optional): Transform pipeline for
            detokenization. When provided (typical for HuggingFace /
            id-tokenization models), full-sequence incremental decoding
            is used to yield clean text. When ``None``, tokens are looked
            up directly in the vocabulary.
        timeout (float): Maximum seconds to wait for the next token before
            the iterator stops. Default is 120.0.

    Example usage::

        import threading
        from eole.inference_engine import InferenceEnginePY
        from eole.predict.streamer import GenerationStreamer

        engine = InferenceEnginePY(config)
        streamer = GenerationStreamer(engine.predictor.vocabs,
                                      engine.transform_pipe)

        def run():
            engine.infer_list(["Hello, how are you?"], streamer=streamer)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        for chunk in streamer:
            print(chunk, end="", flush=True)

        thread.join()
    """

    # Sentinel value placed in the queue to signal end-of-generation
    _STOP = object()

    def __init__(self, vocabs, transform_pipe=None, timeout: float = 120.0):
        self.vocabs = vocabs
        self.transform_pipe = transform_pipe
        self.timeout = timeout
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        # Accumulated token IDs used for incremental decode (id_tokenization path)
        self._token_ids = []
        # Decoded text produced so far; used to compute the new suffix
        self._decoded_so_far: str = ""
        # Lock to allow put()/end() to be called from a background thread safely
        self._finished = threading.Event()

    # ------------------------------------------------------------------
    # Producer side (called from the inference thread)
    # ------------------------------------------------------------------

    def put(self, token_ids):
        """Add newly generated token IDs to the stream.

        Called by the generation loop after each decoding step.

        Args:
            token_ids: A 1-D tensor or list of shape ``(batch_size,)``
                containing the token IDs produced at the current step.
                Only the **first** element is used for streaming.
        """
        if hasattr(token_ids, "item"):
            # scalar tensor
            token_id = int(token_ids.item())
        elif hasattr(token_ids, "__getitem__"):
            item = token_ids[0]
            token_id = int(item.item()) if hasattr(item, "item") else int(item)
        else:
            token_id = int(token_ids)
        self._queue.put(token_id)

    def end(self):
        """Signal that generation is complete.

        Must be called once by the inference thread after the last token
        has been put, so that the consumer iterator can terminate cleanly.
        """
        self._queue.put(self._STOP)
        self._finished.set()

    # ------------------------------------------------------------------
    # Consumer side (called from the caller's thread / async task)
    # ------------------------------------------------------------------

    def __iter__(self):
        """Iterate over decoded text chunks as they are generated.

        Yields:
            str: Non-empty decoded text chunks, one per newly-generated
            token (or slightly larger when incremental detokenization
            defers output to avoid partial UTF-8 / BPE pieces).
        """
        while True:
            try:
                item = self._queue.get(timeout=self.timeout)
            except queue.Empty:
                # Timed out waiting for the next token – stop iteration.
                break

            if item is self._STOP:
                break

            token_id: int = item
            chunk = self._decode_token(token_id)
            if chunk:
                yield chunk

    def _decode_token(self, token_id: int) -> Optional[str]:
        """Convert a single new token ID to a text chunk.

        Two strategies are used depending on whether a transform pipeline
        is available:

        * **With transform_pipe** (id-tokenization / HF tokenizer path):
          Accumulate the token ID and decode the full sequence so far.
          Yield only the suffix that has not been yielded yet.  This
          avoids BPE/SentencePiece artefacts (e.g. leading "Ġ" or "▁").

        * **Without transform_pipe** (string-vocab path):
          Look up the token string directly in the vocabulary.

        Args:
            token_id: Integer token ID for the current step.

        Returns:
            Decoded text chunk, or ``None`` / empty string if nothing
            printable can be produced yet.
        """
        if self.transform_pipe is not None:
            # Incremental decode: decode everything and yield the new suffix
            self._token_ids.append(token_id)
            new_text: str = self.transform_pipe.apply_reverse(list(self._token_ids))
            new_chunk: str = new_text[len(self._decoded_so_far) :]
            if new_chunk:
                self._decoded_so_far = new_text
            return new_chunk
        else:
            # Fallback: direct vocabulary lookup
            voc = self.vocabs["tgt"].ids_to_tokens
            try:
                return voc[token_id]
            except (IndexError, KeyError):
                return None
