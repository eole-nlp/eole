"""Base Transform class and relate utils."""
import os
import json
import shutil
import copy
from eole.utils.logging import logger

from eole.config import Config, recursive_model_fields_set


class TransformConfig(Config):
    # really bad patch for now, which will allow us to store anything in the TransformConfig
    # model_config = ConfigDict(
    #     extra="allow"
    # )  # not sure we want to retain this, here for testing purposes for now
    pass


class Transform(object):
    """A Base class that every transform method should derived from."""

    name = None  # set in register_transform wrapper

    def __init__(self, config):
        """Initialize Transform by parsing `opts` and add them as attribute."""
        # better segregation/nesting of transform specific stuff in config
        self.config = getattr(config.transforms_configs, self.name)
        # set at transform level to save some artifacts to model_path directory
        self.artifacts = []
        # retain a copy of the full config for some specific cases (seed, share_vocab, etc.)
        self.full_config = config
        self._parse_config()

    def _set_seed(self, seed):
        """Reproducibility: Set seed for non-deterministic transform."""
        pass

    @classmethod
    def require_vocab(cls):
        """Override this method to inform it need vocab to start."""
        return False

    def warm_up(self, vocabs=None):
        """Procedure needed after initialize and before apply.

        This should be override if there exist any procedure necessary
        before `apply`, like setups based on parsed options or load models,
        etc.
        """
        if self.full_config.seed > 0:
            self._set_seed(self.full_config.seed)
        if self.require_vocab():
            if vocabs is None:
                raise ValueError(f"{type(self).__name__} requires vocabs!")
            self.vocabs = vocabs

    def _save_artifacts(self, model_path):
        save_config = copy.deepcopy(self.config)
        for artifact in self.artifacts:
            maybe_artifact = getattr(self, artifact, None)
            if maybe_artifact is not None and os.path.exists(maybe_artifact):
                # copy file to model_path directory
                try:
                    shutil.copy(maybe_artifact, model_path)
                except shutil.SameFileError:
                    pass
                setattr(
                    save_config,
                    artifact,
                    os.path.join("${MODEL_PATH}", os.path.basename(maybe_artifact)),
                )
        config_path = os.path.join(model_path, f"{self.name}.json")
        with open(config_path, "w") as f:
            json.dump(
                recursive_model_fields_set(save_config), f, indent=2, ensure_ascii=False
            )

    @classmethod
    def add_options(cls, parser):
        """Available options relate to this Transform."""
        pass

    @classmethod
    def _validate_options(cls, config):
        """Extra checks to validate options added from `add_options`."""
        pass

    @classmethod
    def get_specials(cls, config):
        return ([], [])

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply transform to `example`.

        Args:
            example (dict): a dict of field value, ex. src, tgt;
            is_train (bool): Indicate if src/tgt is training data;
            stats (TransformStatistics): a statistic object.
        """
        raise NotImplementedError

    def batch_apply(self, batch, is_train=False, **kwargs):
        """Apply transform to `batch`.
        Args:
            batch (list): a list of examples;
            is_train (bool): Indicate if src/tgt is training data;bject.
        """
        transformed_batch = []
        for example, _, cid in batch:
            example = self.apply(example, is_train=is_train, **kwargs)
            if example is not None:
                transformed_batch.append((example, self, cid))
        return transformed_batch

    def apply_reverse(self, predicted):
        return predicted

    def batch_apply_reverse(self, predicted_batch, **kwargs):
        transformed_batch = []
        for predicted in predicted_batch:
            reversed = self.apply_reverse(predicted, **kwargs)
            if isinstance(reversed, list):
                transformed_batch += reversed
            elif reversed is not None:
                transformed_batch.append(reversed)
        return transformed_batch

    def __getstate__(self):
        """Pickling following for rebuild."""
        state = {"config": self.config, "full_config": self.full_config}
        if hasattr(self, "vocabs"):
            state["vocabs"] = self.vocabs
        return state

    def _parse_config(self):
        """Parse opts to set/reset instance's attributes.

        This should be override if there are attributes other than self.opts.
        To make sure we recover from picked state.
        (This should only contain attribute assignment, other routine is
        suggest to define in `warm_up`.)
        """
        pass

    def __setstate__(self, state):
        """Reload when unpickling from save file."""
        self.config = state["config"]
        self.full_config = state["full_config"]
        self._parse_config()
        vocabs = state.get("vocabs", None)
        self.warm_up(vocabs=vocabs)

    def stats(self):
        """Return statistic message."""
        return ""

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return ""

    def __repr__(self):
        cls_name = type(self).__name__
        cls_args = self._repr_args()
        return "{}({})".format(cls_name, cls_args)


class ObservableStats:
    """A running observable statistics."""

    __slots__ = []

    def name(self) -> str:
        """Return class name as name for statistics."""
        return type(self).__name__

    def update(self, other: "ObservableStats"):
        """Update current statistics with others."""
        raise NotImplementedError

    def __str__(self) -> str:
        return "{}({})".format(
            self.name(),
            ", ".join(f"{name}={getattr(self, name)}" for name in self.__slots__),
        )


class TransformStatistics:
    """A observer containing runing statistics."""

    def __init__(self):
        self.observables = {}

    def update(self, observable: ObservableStats):
        """Adding observable to observe/updating existing observable."""
        name = observable.name()
        if name not in self.observables:
            self.observables[name] = observable
        else:
            self.observables[name].update(observable)

    def report(self):
        """Pop out all observing statistics and reporting them."""
        msgs = []
        report_ids = list(self.observables.keys())
        for name in report_ids:
            observable = self.observables.pop(name)
            msgs.append(f"\t\t\t* {str(observable)}")
        if len(msgs) != 0:
            return "\n".join(msgs)
        else:
            return ""


class TransformPipe(Transform):
    """Pipeline built by a list of Transform instance."""

    def __init__(self, config, transform_list):
        """Initialize pipeline by a list of transform instance."""
        self.config = None  # not required -> not sure why?
        self.transforms = transform_list
        self.statistics = TransformStatistics()

    @classmethod
    def build_from(cls, transform_list):
        """Return a `TransformPipe` instance build from `transform_list`."""
        for transform in transform_list:
            assert isinstance(
                transform, Transform
            ), "transform should be a instance of Transform."
        transform_pipe = cls(None, transform_list)
        return transform_pipe

    def warm_up(self, vocabs):
        """Warm up Pipeline by iterate over all transfroms."""
        for transform in self.transforms:
            transform.warm_up(vocabs)

    # not sure this is really neeeded as a classmethod, might be simplified
    @classmethod
    def get_specials(cls, config, transforms):
        """Return all specials introduced by `transforms`."""
        src_specials, tgt_specials = [], []
        for transform in transforms:
            _src_special, _tgt_special = transform.get_specials(transform.config)
            for _src_spec in _src_special:
                src_specials.append(_src_spec)
            for _tgt_spec in _tgt_special:
                tgt_specials.append(_tgt_spec)
        return (src_specials, tgt_specials)

    def apply(self, example, is_train=False, **kwargs):
        """Apply transform pipe to `example`.

        Args:
            example (dict): a dict of field value, ex. src, tgt.

        """
        for transform in self.transforms:
            example = transform.apply(
                example, is_train=is_train, stats=self.statistics, **kwargs
            )
            if example is None:
                break
        return example

    def batch_apply(self, batch, is_train=False, **kwargs):
        """Apply transform pipe to `example`.

        Args:
            example (dict): a dict of field value, ex. src, tgt.

        """
        for transform in self.transforms:
            batch = transform.batch_apply(
                batch, is_train=is_train, stats=self.statistics, **kwargs
            )
            if batch is None:
                break
        return batch

    def apply_reverse(self, predicted):
        for transform in self.transforms:
            predicted = transform.apply_reverse(predicted)
        return predicted

    def __getstate__(self):
        """Pickling following for rebuild."""
        return (self.config, self.transforms, self.statistics)

    def __setstate__(self, state):
        """Reload when unpickling from save file."""
        self.config, self.transforms, self.statistics = state

    def stats(self):
        """Return statistic message."""
        return self.statistics.report()

    def _repr_args(self):
        """Return str represent key arguments for class."""
        info_args = []
        for transform in self.transforms:
            info_args.append(repr(transform))
        return ", ".join(info_args)


def make_transforms(config, transforms_cls, vocabs):
    """Build transforms in `transforms_cls` with vocab of `fields`."""
    transforms = {}
    if transforms_cls:
        for name, transform_cls in transforms_cls.items():
            if transform_cls.require_vocab() and vocabs is None:
                logger.warning(
                    f"{transform_cls.__name__} require vocab to apply, skip it."
                )
                continue
            # let's look for saved transform config in model_path dir
            # model_path = config.get_model_path()
            # config_path = os.path.join(model_path, f"{name}.json")
            # if config.training.train_from is not None and os.path.exists(config_path):
            #     # TODO: add some mechanism to prevent override if transforms explicitly passed
            #     with open(config_path) as f:
            #         os.environ["MODEL_PATH"] = model_path
            #         transform_config = json.loads(os.path.expandvars(f.read()))
            #         transform_config = transform_cls.config_model(**transform_config)
            #         setattr(config.transforms_configs, name, transform_config)
            transform_obj = transform_cls(config)
            transform_obj.warm_up(vocabs)
            transforms[name] = transform_obj
    logger.info(f"Transforms applied: {list(transforms.keys())}")
    return transforms


def get_specials(config, transforms_cls_dict):
    """Get specials of transforms that should be registed in Vocab."""
    all_specials = {"src": [], "tgt": []}
    for name, transform_cls in transforms_cls_dict.items():
        src_specials, tgt_specials = transform_cls.get_specials(config)
        for src_spec in src_specials:
            all_specials["src"].append(src_spec)
        for tgt_spec in tgt_specials:
            all_specials["tgt"].append(tgt_spec)
    logger.info(f"Get special vocabs from Transforms: {all_specials}.")
    return all_specials
