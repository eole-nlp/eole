import json
from eole.constants import CorpusTask, DefaultTokens, ModelType
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter
from eole.utils.distributed import ErrorHandler, spawned_infer
from eole.utils.logging import init_logger
from eole.transforms import get_transforms_cls, make_transforms, TransformPipe


class InferenceEngine(object):
    """Wrapper Class to run Inference.

    Args:
        opt: inference options
    """

    def __init__(self, config):
        self.config = config  # PredictConfig
        self.model_type = None

    def predict_batch(self, batch):
        pass

    def _predict(self, infer_iter):
        pass

    def infer_file(self):
        """File inference. Source file must be the opt.src argument"""
        if self.config.world_size <= 1:
            infer_iter = build_dynamic_dataset_iter(
                self.config,
                self.transforms,
                self.vocabs,
                task=CorpusTask.INFER,
                device_id=self.device_id,
                model_type=self.model_type,  # patch for CT2
            )
            scores, estims, preds = self._predict(infer_iter)
        else:
            scores, estims, preds = self.infer_file_parallel()
        return scores, estims, preds

    def infer_list(self, src):
        """List of strings inference `src`"""
        if self.config.world_size <= 1:
            infer_iter = build_dynamic_dataset_iter(
                self.config,
                self.transforms,
                self.vocabs,
                task=CorpusTask.INFER,
                src=src,
                device_id=self.device_id,
                model_type=self.model_type,
            )
            scores, estims, preds = self._predict(infer_iter)
        else:
            scores, estims, preds = self.infer_list_parallel(src)
        return scores, estims, preds

    def infer_file_parallel(self):
        """File inference in mulitprocessing with partitioned models."""
        raise NotImplementedError(
            "The inference in mulitprocessing with partitioned models is not implemented."
        )

    def infer_list_parallel(self, src):
        """The inference in mulitprocessing with partitioned models."""
        raise NotImplementedError(
            "The inference in mulitprocessing with partitioned models is not implemented."
        )

    def _score(self, infer_iter):
        pass

    def score_file(self):
        """File scoring. Source file must be the opt.src argument"""
        if self.config.world_size <= 1:
            infer_iter = build_dynamic_dataset_iter(
                self.config,
                self.transforms,
                self.vocabs,
                task=CorpusTask.INFER,
                device_id=self.device_id,
                tgt=self.config.src,
            )
            score_results = self._score(infer_iter)
        else:
            score_results = self.score_file_parallel()
        return score_results

    def score_list(self, src):
        """List of strings scoring tgt`"""
        if self.config.world_size <= 1:
            infer_iter = build_dynamic_dataset_iter(
                self.config,
                self.transforms,
                self.vocabs,
                task=CorpusTask.INFER,
                src=src,
                tgt=src,
                device_id=self.device_id,
            )
            score_results = self._score(infer_iter)
        else:
            score_results = self.score_list_parallel(src)
        return score_results

    def terminate(self):
        pass


class InferenceEnginePY(InferenceEngine):
    """Inference engine subclass to run inference with `predict.py`.

    Args:
        opt: inference options
    """

    def __init__(self, config):
        import torch
        from eole.predict import build_predictor

        super().__init__(config)
        self.logger = init_logger(config.log_file)

        if config.world_size > 1:
            mp = torch.multiprocessing.get_context("spawn")
            # Create a thread to listen for errors in the child processes.
            self.error_queue = mp.SimpleQueue()
            self.error_handler = ErrorHandler(self.error_queue)
            self.queue_instruct = []
            self.queue_result = []
            self.procs = []

            for device_id in range(config.world_size):
                self.queue_instruct.append(mp.Queue())
                self.queue_result.append(mp.Queue())
                self.procs.append(
                    mp.Process(
                        target=spawned_infer,
                        args=(
                            config,
                            device_id,
                            self.error_queue,
                            self.queue_instruct[device_id],
                            self.queue_result[device_id],
                        ),
                        daemon=False,
                    )
                )
                self.procs[device_id].start()
                self.error_handler.add_child(self.procs[device_id].pid)
        else:
            self.device_id = config.gpu
            self.predictor = build_predictor(
                config, self.device_id, logger=self.logger, report_score=True
            )
            self.transforms_cls = get_transforms_cls(config._all_transform)
            self.vocabs = self.predictor.vocabs
            self.transforms = make_transforms(config, self.transforms_cls, self.vocabs)
            self.transform_pipe = TransformPipe.build_from(self.transforms.values())

    def _predict(self, infer_iter):
        scores, estims, preds = self.predictor._predict(
            infer_iter,
            infer_iter.transforms,
            self.config.attn_debug,
            self.config.align_debug,
        )
        return scores, estims, preds

    def _score(self, infer_iter):
        self.predictor.with_scores = True
        self.predictor.return_gold_log_probs = True
        return self.predictor._score(infer_iter)

    def score_list_parallel(self, src):
        assert self.config.world_size > 1, "World size must be greater than 1."
        for device_id in range(self.config.world_size):
            self.queue_instruct[device_id].put(("score_list", src))
        score_results = []
        for device_id in range(self.config.world_size):
            score_results.append(self.queue_result[device_id].get())
        return score_results

    def score_file_parallel(self):
        assert self.config.world_size > 1, "World size must be greater than 1."
        for device_id in range(self.config.world_size):
            self.queue_instruct[device_id].put(("score_file", self.config))
        score_results = []
        for device_id in range(self.config.world_size):
            score_results.append(self.queue_result[device_id].get())
        return score_results[0]

    def infer_file_parallel(self):
        assert self.config.world_size > 1, "World size must be greater than 1."
        for device_id in range(self.config.world_size):
            self.queue_instruct[device_id].put(("infer_file", self.config))
        scores, preds = [], []
        for device_id in range(self.config.world_size):
            scores.append(self.queue_result[device_id].get())
            preds.append(self.queue_result[device_id].get())
        return scores[0], preds[0]

    def infer_list_parallel(self, src):
        assert self.config.world_size > 1, "World size must be greater than 1."
        for device_id in range(self.config.world_size):
            self.queue_instruct[device_id].put(("infer_list", src))
        scores, preds = [], []
        for device_id in range(self.config.world_size):
            scores.append(self.queue_result[device_id].get())
            preds.append(self.queue_result[device_id].get())
        return scores[0], preds[0]

    def terminate(self):
        if self.config.world_size > 1:
            for device_id in range(self.config.world_size):
                self.queue_instruct[device_id].put(("stop"))
                self.procs[device_id].terminate()


class InferenceEngineCT2(InferenceEngine):
    """Inference engine subclass to run inference with ctranslate2.

    Args:
        opt: inference options
    """

    def __init__(self, config, model_type=None):
        import ctranslate2
        import pyonmttok

        super().__init__(config)
        assert (
            model_type is not None
        ), "A model_type kwarg must be passed for CT2 models."
        self.logger = init_logger(config.log_file)
        assert self.config.world_size <= 1, "World size must be less than 1."
        self.device_id = config.gpu
        if config.world_size == 1:
            self.device_index = config.gpu_ranks
            self.device = "cuda"
        else:
            self.device_index = 0
            self.device = "cpu"
        self.transforms_cls = get_transforms_cls(self.config._all_transform)
        # Build translator
        self.model_type = model_type
        if self.model_type == ModelType.DECODER:
            self.predictor = ctranslate2.Generator(
                config.get_model_path(),
                device=self.device,
                device_index=self.device_index,
            )
        else:
            self.predictor = ctranslate2.Translator(
                config.get_model_path(),
                device=self.device,
                device_index=self.device_index,
            )
        # Build vocab
        vocab_path = config.src_subword_vocab  # this is not super clean
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        vocabs = {}
        src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
        vocabs["src"] = src_vocab
        vocabs["tgt"] = src_vocab
        vocabs["decoder_start_token"] = "<s>"
        self.vocabs = vocabs
        # Build transform pipe
        transforms = make_transforms(config, self.transforms_cls, self.vocabs)
        self.transforms = TransformPipe.build_from(transforms.values())

    def predict_batch(self, batch, config):
        input_tokens = []
        for i in range(batch["src"].size()[0]):
            start_ids = batch["src"][i, :].cpu().numpy().tolist()
            _input_tokens = [
                self.vocabs["src"].lookup_index(id)
                for id in start_ids
                if id != self.vocabs["src"].lookup_token(DefaultTokens.PAD)
            ]
            input_tokens.append(_input_tokens)
        if self.model_type == ModelType.DECODER:
            predicted_batch = self.predictor.generate_batch(
                start_tokens=input_tokens,
                batch_type=("examples" if config.batch_type == "sents" else "tokens"),
                max_batch_size=config.batch_size,
                beam_size=config.beam_size,
                num_hypotheses=config.n_best,
                max_length=config.max_length,
                return_scores=True,
                include_prompt_in_result=False,
                sampling_topk=config.random_sampling_topk,
                sampling_topp=config.random_sampling_topp,
                sampling_temperature=config.random_sampling_temp,
            )
            preds = [
                [self.transforms.apply_reverse(tokens) for tokens in out.sequences]
                for out in predicted_batch
            ]
            scores = [out.scores for out in predicted_batch]
        elif self.model_type == ModelType.ENCODER_DECODER:
            predicted_batch = self.predictor.translate_batch(
                input_tokens,
                batch_type=("examples" if config.batch_type == "sents" else "tokens"),
                max_batch_size=config.batch_size,
                beam_size=config.beam_size,
                num_hypotheses=config.n_best,
                max_decoding_length=config.max_length,
                return_scores=True,
                sampling_topk=config.random_sampling_topk,
                sampling_topp=config.random_sampling_topp,
                sampling_temperature=config.random_sampling_temp,
            )
            preds = [
                [self.transforms.apply_reverse(tokens) for tokens in out.hypotheses]
                for out in predicted_batch
            ]
            scores = [out.scores for out in predicted_batch]

        return scores, None, preds

    def _predict(self, infer_iter):
        scores = []
        preds = []
        for batch, bucket_idx in infer_iter:
            _scores, _, _preds = self.predict_batch(batch, self.config)
            scores += _scores
            preds += _preds
        return scores, None, preds

    def _score(self, infer_iter):
        raise NotImplementedError(
            "The scoring with InferenceEngineCT2 is not implemented."
        )
