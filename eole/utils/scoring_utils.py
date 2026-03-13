import codecs
import os
from eole.predict import GNMTGlobalScorer, Translator
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
            raw_srcs: raw (non-detokenized) source texts as they were read
        """
        # ########## #
        # Translator #
        # ########## #

        # Build translator from options
        model_config = self.config.model
        model_config._validate_model_config()

        data_type = getattr(self.config.training, "data_type", "text")

        # This is somewhat broken and we shall remove or improve
        # (take 'inference' field of config if exists?)
        # Set "default" translation options on empty cfgfile
        self.config.training.num_workers = 0

        model_path = self.config.training.get_model_path()

        predict_kwargs = dict(
            model_path=[model_path],
            src=self.config.data["valid"].path_src,
            compute_dtype=self.config.training.compute_dtype,
            share_vocab=self.config.share_vocab,
            beam_size=1,
            transforms=self.config.transforms,
            transforms_configs=self.config.transforms_configs,
            model=model_config,
            gpu_ranks=[gpu_rank],
            data_type=data_type,
        )

        if self.config.inference is not None:
            for field in ("language", "task", "max_length", "beam_size", "length_penalty", "seed"):
                value = getattr(self.config.inference, field, None)
                if value is not None:
                    predict_kwargs[field] = value

        prefix_config = getattr(self.config.transforms_configs, "prefix", None)
        if prefix_config is not None:
            predict_kwargs["tgt_file_prefix"] = prefix_config.tgt_prefix != ""
        else:
            predict_kwargs["tgt_file_prefix"] = False

        predict_config = PredictConfig(**predict_kwargs)

        scorer = GNMTGlobalScorer.from_config(predict_config)

        if data_type == "audio":
            from eole.predict.audio_predictor import AudioPredictor

            translator = AudioPredictor(
                model,
                self.vocabs,
                predict_config,
                model_config,
                device_id=gpu_rank,
                global_scorer=scorer,
                report_score=False,
                logger=None,
            )
        else:
            translator = Translator(
                model,
                self.vocabs,
                predict_config,
                model_config,
                device_id=gpu_rank,
                global_scorer=scorer,
                report_score=False,
                logger=None,
            )

        # ################### #
        # Validation iterator #
        # ################### #

        # Reinstantiate the validation iterator
        # Retrieve raw references and sources
        with codecs.open(self.config.data["valid"].path_tgt, "r", encoding="utf-8") as f:
            raw_refs = [line.strip("\n") for line in f if line.strip("\n")]
        with codecs.open(self.config.data["valid"].path_src, "r", encoding="utf-8") as f:
            raw_srcs = [line.strip("\n") for line in f if line.strip("\n")]

        infer_iter = build_dynamic_dataset_iter(
            predict_config,
            self.transforms,
            translator.vocabs,
            task=CorpusTask.INFER,
            tgt="",  # This force to clear the target side (needed when using tgt_file_prefix)
            device_id=gpu_rank,
        )

        # ########### #
        # Predictions #
        # ########### #
        _, _, preds = translator._predict(
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

        if hasattr(translator, "_tokenizer") and translator._tokenizer is not None:
            refs = [
                translator._tokenizer.decode(translator._tokenizer.encode(ref).ids, skip_special_tokens=True)
                for ref in raw_refs
            ]
        else:
            refs = raw_refs

        # Save results
        if len(preds) > 0 and self.config.scoring_debug and self.config.dump_preds is not None:
            path = os.path.join(self.config.dump_preds, f"preds.valid_step_{step}.txt")
            with open(path, "a", encoding="utf-8") as file:
                for i in range(len(raw_srcs)):
                    file.write("SOURCE: {}\n".format(raw_srcs[i]))
                    file.write("RAW REF: {}\n".format(raw_refs[i]))
                    file.write("REF: {}\n".format(refs[i]))
                    file.write("PRED: {}\n\n".format(preds[i]))
        return preds, refs, raw_srcs
