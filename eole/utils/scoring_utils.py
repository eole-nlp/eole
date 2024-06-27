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
        """
        # ########## #
        # Translator #
        # ########## #

        # This is somewhat broken and we shall remove or improve
        # (take 'inference' field of config if exists?)
        # Set "default" translation options on empty cfgfile
        predict_config = PredictConfig(model_path=["dummy"], src="dummy")
        predict_config.gpu = gpu_rank
        if predict_config.transforms_configs.prefix.tgt_prefix != "":
            predict_config.tgt_file_prefix = True
        predict_config.beam_size = 1  # prevent OOM when GPU is almost full at training
        predict_config._validate_predict_config()
        # Build translator from options
        scorer = GNMTGlobalScorer.from_config(predict_config)
        out_file = codecs.open(os.devnull, "w", "utf-8")
        model_config = self.config.model
        model_config._validate_model_config()
        translator = (
            Translator.from_config(  # we need to review opt/config stuff in translator
                model,
                self.vocabs,
                predict_config,
                model_config,
                global_scorer=scorer,
                out_file=out_file,
                report_align=predict_config.report_align,
                report_score=False,
                logger=None,
            )
        )

        # ################### #
        # Validation iterator #
        # ################### #

        # Reinstantiate the validation iterator
        self.config.training.num_workers = 0
        predict_config.src = self.config.data["valid"].path_src
        predict_config.transforms = self.config.transforms
        predict_config.transforms_configs = self.config.transforms_configs
        predict_config.model = model_config
        # Retrieve raw references and sources
        with codecs.open(
            self.config.data["valid"].path_tgt, "r", encoding="utf-8"
        ) as f:
            raw_refs = [line.strip("\n") for line in f if line.strip("\n")]
        with codecs.open(
            self.config.data["valid"].path_src, "r", encoding="utf-8"
        ) as f:
            raw_srcs = [line.strip("\n") for line in f if line.strip("\n")]

        infer_iter = build_dynamic_dataset_iter(
            predict_config,
            self.transforms,
            translator.vocabs,
            task=CorpusTask.INFER,
            tgt="",  # This force to clear the target side (needed when using tgt_file_prefix)
            device_id=predict_config.gpu,
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
        # Save results
        if (
            len(preds) > 0
            and self.config.scoring_debug
            and self.config.dump_preds is not None
        ):
            path = os.path.join(self.config.dump_preds, f"preds.valid_step_{step}.txt")
            with open(path, "a") as file:
                for i in range(len(raw_srcs)):
                    file.write("SOURCE: {}\n".format(raw_srcs[i]))
                    file.write("REF: {}\n".format(raw_refs[i]))
                    file.write("PRED: {}\n\n".format(preds[i]))
        return preds, raw_refs
