import codecs
import os
from eole.predict import GNMTGlobalScorer, Translator, GeneratorLM
from eole.config.run import (
    PredictConfig,
)  # probably should be done differently, but might work for now
from eole.constants import CorpusTask
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter
from eole.transforms import get_transforms_cls, make_transforms


class ScoringPreparator:
    """Allow the calculation of metrics via the Trainer's
    training_eval_handler method.
    """

    def __init__(self, vocabs, config):
        self.vocabs = vocabs
        self.config = config
        if self.config.dump_preds is not None:
            if not os.path.exists(self.config.dump_preds):
                os.makedirs(self.config.dump_preds)
        self.transforms = None
        self.transforms_cls = None

    def warm_up(self, transforms):
        self.transforms_cls = get_transforms_cls(transforms)
        self.transforms = make_transforms(self.config, self.transforms_cls, self.vocabs)

    def translate(self, model, gpu_rank, step):
        """Compute and save the sentences predicted by the
        current model's state related to a batch.

        Args:
            model (:obj:`eole.models.XXXModel`): The current model's state.
            transformed_batches(list of lists): A list of transformed batches.
            gpu_rank (int): Ordinal rank of the gpu where the
                translation is to be done.
            step: The current training step.
            mode: (string): 'train' or 'valid'.
        Returns:
            preds (list): Detokenized predictions
            texts_ref (list): Detokenized target sentences
        """
        # ######### #
        # Predictor #
        # ######### #

        # Build predictor from options
        model_config = self.config.model
        model_config._validate_model_config()

        # This is somewhat broken and we shall remove or improve
        # (take 'inference' field of config if exists?)
        # Set "default" translation options on empty cfgfile
        self.config.training.num_workers = 0
        is_seq2seq = model.encoder is not None and model.decoder is not None
        if not is_seq2seq:
            if "insert_mask_before_placeholder" in self.config.transforms:
                self.response_patterns = self.config.transforms_configs.insert_mask_before_placeholder.response_patterns
        else:
            self.response_patterns = None

        predict_config = PredictConfig(
            model_path=["dummy"],
            src="dummy",
            compute_dtype=self.config.training.compute_dtype,
            beam_size=1,
            transforms=self.config.transforms,
            transforms_configs=self.config.transforms_configs,
            model=model_config,
            tgt_file_prefix=self.config.transforms_configs.prefix.tgt_prefix != "",
            gpu_ranks=[gpu_rank],
            batch_type=self.config.training.batch_type,
            batch_size=self.config.training.batch_size,
        )

        scorer = GNMTGlobalScorer.from_config(predict_config)

        if is_seq2seq:
            predictor = Translator.from_config(  # we need to review opt/config stuff in translator
                model,
                self.vocabs,
                predict_config,
                model_config,
                device_id=gpu_rank,
                global_scorer=scorer,
                report_align=predict_config.report_align,
                report_score=False,
                logger=None,
            )
        else:
            predictor = GeneratorLM.from_config(
                model,
                self.vocabs,
                predict_config,
                model_config,
                device_id=gpu_rank,
                global_scorer=scorer,
                report_align=predict_config.report_align,
                report_score=False,
                logger=None,
            )

        # ################### #
        # Validation iterator #
        # ################### #

        # Reinstantiate the validation iterator
        # Retrieve raw references and sources
        with codecs.open(self.config.data["valid"].path_src, "r", encoding="utf-8") as f:
            raw_srcs = [line.strip("\n") for line in f if line.strip("\n")]

        if not is_seq2seq and self.response_patterns is not None:
            prompts, answers = [], []
            for i, _raw_src in enumerate(raw_srcs):
                for _pattern in self.response_patterns:
                    if len(_raw_src.split(_pattern)) == 2:
                        prompt, answer = _raw_src.split(_pattern)
                        prompts.append(prompt + _pattern)
                        answers.append(answer)
            raw_srcs = prompts
            raw_refs = answers
        else:
            with codecs.open(self.config.data["valid"].path_tgt, "r", encoding="utf-8") as f:
                raw_refs = [line.strip("\n") for line in f if line.strip("\n")]

        infer_iter = build_dynamic_dataset_iter(
            predict_config,
            self.transforms,
            predictor.vocabs,
            src=raw_srcs,
            task=CorpusTask.INFER,
            tgt="",  # This force to clear the target side (needed when using tgt_file_prefix)
            device_id=gpu_rank,
        )

        # ########### #
        # Predictions #
        # ########### #
        _, _, preds = predictor._predict(
            infer_iter,
            transform=infer_iter.transforms,
            attn_debug=predict_config.attn_debug,
            align_debug=predict_config.align_debug,
        )

        # ####### #
        # Outputs #
        # ####### #

        # Flatten predictions
        preds = [x.lstrip() for sublist in preds for x in sublist]
        # Save results
        if len(preds) > 0 and self.config.scoring_debug and self.config.dump_preds is not None:
            path = os.path.join(self.config.dump_preds, f"preds.valid_step_{step}.txt")
            with open(path, "a") as file:
                for i in range(len(raw_srcs)):
                    file.write("SOURCE: {}\n".format(raw_srcs[i]))
                    file.write("REF: {}\n".format(raw_refs[i]))
                    file.write("PRED: {}\n\n".format(preds[i]))
        return preds, raw_refs
