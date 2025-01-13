import torch
import json
import os
import codecs
from eole.constants import CorpusTask, DefaultTokens, ModelType
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter
from eole.utils.distributed import ErrorHandler
from eole.utils.distributed_workers import spawned_infer
from eole.utils.logging import init_logger
from eole.utils.misc import get_device_type
from eole.transforms import get_transforms_cls, make_transforms, TransformPipe


class InferenceEngine(object):
    """Wrapper Class to run Inference.

    Args:
        opt: inference options
    """

    def __init__(self, config):
        self.config = config  # PredictConfig
        self.model_type = None
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)

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

        out_file = codecs.open(self.config.output, "w+", "utf-8")

        flatten_preds = [text for sublist in preds for text in sublist]
        flatten_scores = [score for sublist in scores for score in sublist]
        if estims is not None:
            flatten_estims = [estim for sublist in estims for estim in sublist]
        else:
            flatten_estims = [1.0 for sublist in scores for score in sublist]

        if self.config.with_score:
            out_all = [
                pred + "\t" + str(score) + "\t" + str(estim)
                for (pred, score, estim) in zip(flatten_preds, flatten_scores, flatten_estims)
            ]
            out_file.write("\n".join(out_all) + "\n")
        else:
            out_file.write("\n".join(flatten_preds) + "\n")

        return scores, estims, preds

    def infer_list(self, src, settings={}):
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
            scores, estims, preds = self._predict(infer_iter, settings=settings)
        else:
            scores, estims, preds = self.infer_list_parallel(src, settings=settings)
        return scores, estims, preds

    def infer_file_parallel(self):
        """File inference in mulitprocessing with partitioned models."""
        raise NotImplementedError("The inference in mulitprocessing with partitioned models is not implemented.")

    def infer_list_parallel(self, src, settings={}):
        """The inference in mulitprocessing with partitioned models."""
        raise NotImplementedError("The inference in mulitprocessing with partitioned models is not implemented.")

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
            self.queue_settings = []
            self.queue_result = []
            self.procs = []

            for device_id in range(config.world_size):
                self.queue_instruct.append(mp.Queue())
                self.queue_settings.append(mp.Queue())
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
            if len(config.gpu_ranks) > 0:
                self.device_id = config.gpu_ranks[0]
            else:
                self.device_id = -1  # cpu

            self.predictor = build_predictor(config, self.device_id, logger=self.logger, report_score=True)
            self.transforms_cls = get_transforms_cls(config._all_transform)
            self.vocabs = self.predictor.vocabs
            self.transforms = make_transforms(config, self.transforms_cls, self.vocabs)
            self.transform_pipe = TransformPipe.build_from(self.transforms.values())

    @torch.inference_mode()
    def _predict(self, infer_iter, settings={}):
        self.predictor.update_settings(**settings)
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

    def infer_file_parallel(self, settings={}):
        assert self.config.world_size > 1, "World size must be greater than 1."
        for device_id in range(self.config.world_size):
            self.queue_instruct[device_id].put(("infer_file", self.config))
            # not sure if we want a separate queue or additional info in queue_instruct
            self.queue_settings[device_id].put(settings)
        scores, estims, preds = [], [], []
        for device_id in range(self.config.world_size):
            scores.append(self.queue_result[device_id].get())
            estims.append(self.queue_result[device_id].get())
            preds.append(self.queue_result[device_id].get())
        return scores[0], estims[0], preds[0]

    def infer_list_parallel(self, src, settings={}):
        assert self.config.world_size > 1, "World size must be greater than 1."
        for device_id in range(self.config.world_size):
            self.queue_instruct[device_id].put(("infer_list", src))
            self.queue_settings[device_id].put(settings)
        scores, estims, preds = [], [], []
        for device_id in range(self.config.world_size):
            scores.append(self.queue_result[device_id].get())
            estims.append(self.queue_result[device_id].get())
            preds.append(self.queue_result[device_id].get())
        return scores[0], estims[0], preds[0]

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
        assert model_type is not None, "A model_type kwarg must be passed for CT2 models."
        self.logger = init_logger(config.log_file)
        assert self.config.world_size <= 1, "World size must be less than 1."
        if config.world_size == 1:
            self.device_id = config.gpu_ranks[0]
            self.device_index = config.gpu_ranks
            self.device = get_device_type()
        else:
            self.device_id = -1
            self.device_index = 0
            self.device = "cpu"
        self.transforms_cls = get_transforms_cls(self.config._all_transform)

        ct2_config = os.path.join(config.get_model_path() + "/ctranslate2", "config.json")
        ct2_json = json.load(open(ct2_config, "r"))
        vocabs = {}
        vocabs["specials"] = {}
        vocabs["specials"]["bos_token"] = ct2_json["bos_token"]
        vocabs["specials"]["eos_token"] = ct2_json["eos_token"]
        vocabs["specials"]["unk_token"] = ct2_json["unk_token"]
        if "pad_token" in ct2_json.keys():
            vocabs["specials"]["pad_token"] = ct2_json["pad_token"]
        else:
            vocabs["specials"]["pad_token"] = ct2_json["eos_token"]

        # Build generator or translator
        self.model_type = model_type
        if self.model_type == ModelType.DECODER:
            self.predictor = ctranslate2.Generator(
                config.get_model_path() + "/ctranslate2",
                device=self.device,
                device_index=self.device_index,
            )
            vocab_path = os.path.join(config.get_model_path() + "/ctranslate2", "vocabulary.json")
            vocab = json.load(open(vocab_path, "r"))
            src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
            config.share_vocab = True
            vocabs["src"] = src_vocab
            vocabs["tgt"] = src_vocab
            vocabs["decoder_start_token"] = ""
        else:
            self.predictor = ctranslate2.Translator(
                config.get_model_path(),
                device=self.device,
                device_index=self.device_index,
            )
            vocabs["decoder_start_token"] = ct2_json["decoder_start_token"]
            if os.path.exists(config.get_model_path() + "/ctranslate2", "shared_vocabulary.json"):
                vocab = json.load(
                    open(
                        config.get_model_path() + "/ctranslate2",
                        "shared_vocabulary.json",
                        "r",
                    )
                )
                src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
                config.share_vocab = True
                vocabs["src"] = src_vocab
                vocabs["tgt"] = src_vocab

            else:
                vocab_src = json.load(
                    open(
                        config.get_model_path() + "/ctranslate2",
                        "source_vocabulary.json",
                        "r",
                    )
                )
                src_vocab = pyonmttok.build_vocab_from_tokens(vocab_src)
                vocab_tgt = json.load(
                    open(
                        config.get_model_path() + "/ctranslate2",
                        "target_vocabulary.json",
                        "r",
                    )
                )
                tgt_vocab = pyonmttok.build_vocab_from_tokens(vocab_tgt)
                config.share_vocab = False
                vocabs["src"] = src_vocab
                vocabs["tgt"] = tgt_vocab

        self.vocabs = vocabs
        # Build transform pipe
        self.transforms = make_transforms(config, self.transforms_cls, self.vocabs)
        self.transforms_pipe = TransformPipe.build_from(self.transforms.values())

    def predict_batch(self, batch, config):
        input_tokens = []
        for i in range(batch["src"].size()[0]):
            start_ids = batch["src"][i, :].cpu().numpy().tolist()
            _input_tokens = [
                self.vocabs["src"].lookup_index(id)
                for id in start_ids
                if id != self.vocabs["src"].lookup_token(self.vocabs["specials"].get("pad_token", DefaultTokens.PAD))
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
                sampling_topk=config.top_k,
                sampling_topp=1 if config.top_p == 0 else config.top_p,
                sampling_temperature=config.temperature,
            )
            if self.transforms != {}:
                preds = [
                    [self.transforms_pipe.apply_reverse(nbest) for nbest in ex.sequences] for ex in predicted_batch
                ]
            else:
                preds = [[" ".join(nbest) for nbest in ex.sequences] for ex in predicted_batch]

        elif self.model_type == ModelType.ENCODER_DECODER:
            predicted_batch = self.predictor.translate_batch(
                input_tokens,
                batch_type=("examples" if config.batch_type == "sents" else "tokens"),
                max_batch_size=config.batch_size,
                beam_size=config.beam_size,
                num_hypotheses=config.n_best,
                max_decoding_length=config.max_length,
                return_scores=True,
                sampling_topk=config.top_k,
                sampling_topp=1 if config.top_p == 0 else config.top_p,
                sampling_temperature=config.temperature,
            )
            if self.transforms != {}:
                preds = [
                    [self.transforms_pipe.apply_reverse(nbest) for nbest in ex.hypothesis] for ex in predicted_batch
                ]
            else:
                preds = [[" ".join(nbest) for nbest in ex.sequences] for ex in predicted_batch]

        scores = [[nbest for nbest in ex.scores] for ex in predicted_batch]
        return scores, None, preds

    def _predict(self, infer_iter, settings={}):
        # TODO: convert settings to CT2 naming
        predictions = {}
        predictions["scores"] = []
        predictions["preds"] = []
        predictions["cid_line_number"] = []
        for batch, bucket_idx in infer_iter:
            _scores, _, _preds = self.predict_batch(batch, self.config)
            predictions["scores"] += _scores
            predictions["preds"] += _preds
            predictions["cid_line_number"] += batch["cid_line_number"]
        sorted_data = sorted(
            zip(
                predictions["cid_line_number"],
                predictions["preds"],
                predictions["scores"],
            )
        )
        sorted_predictions = {
            "cid_line_number": [item[0] for item in sorted_data],
            "preds": [item[1] for item in sorted_data],
            "scores": [item[2] for item in sorted_data],
        }
        return sorted_predictions["scores"], None, sorted_predictions["preds"]

    def _score(self, infer_iter):
        raise NotImplementedError("The scoring with InferenceEngineCT2 is not implemented.")
