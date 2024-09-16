
# How can I create custom on-the-fly data transforms?

The code is easily extendable with custom transforms inheriting from the `Transform` base class.

You can for instance have a look at the `FilterTooLongTransform` and the corresponding `FilterTooLongConfig` classes as a template:

```python
class FilterTooLongConfig(TransformConfig):
    src_seq_length: int | None = Field(
        default=192, description="Maximum source sequence length."
    )
    tgt_seq_length: int | None = Field(
        default=192, description="Maximum target sequence length."
    )

...

@register_transform(name="filtertoolong")
class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    config_model = FilterTooLongConfig

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        self.src_seq_length = self.config.src_seq_length
        self.tgt_seq_length = self.config.tgt_seq_length

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if too long else return as is."""
        if len(example["src"]) > self.src_seq_length or (
            example["tgt"] is not None and len(example["tgt"]) > self.tgt_seq_length - 2
        ):
            if stats is not None:
                stats.update(FilterTooLongStats())
            return None
        else:
            return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return "{}={}, {}={}".format(
            "src_seq_length", self.src_seq_length, "tgt_seq_length", self.tgt_seq_length
        )
```

Methods:
- `_parse_opts` allows to parse options from the `config_model`;
- `apply` is where the transform happens;
- `_repr_args` is for clean logging purposes.

As you can see, there is the `@register_transform` wrapper before the class definition. This will allow for the class to be automatically detected (if put in the proper `transforms` folder) and usable in your training configurations through its `name` argument.

You could also collect statistics for your custom transform by creating a class inheriting `ObservableStats`:

```python
class FilterTooLongStats(ObservableStats):
    """Runing statistics for FilterTooLongTransform."""
    __slots__ = ["filtered"]

    def __init__(self):
        self.filtered = 1

    def update(self, other: "FilterTooLongStats"):
        self.filtered += other.filtered
```

NOTE:
- Add elements to keep track in the `__init__` and also `__slot__` to make it lightweight;
- Supply update logic in `update` method;
- (Optional) override `__str__` to change default log message format;
- Instantiate and passing the statistic object in the `apply` method of the corresponding transform class;
- statistics will be gathered per corpus per worker, but only first worker will report for its shard by default.

The `example` argument of `apply` is a `dict` of the form:
```
{
	"src": <source string>,
	"tgt": <target string>,
	"align": <alignment pharaoh string> # optional
}
```

This is defined in `eole.inputters.text_corpus.ParallelCorpus.load`. This class is not easily extendable for now but it can be considered for future developments. For instance, we could create some `CustomParallelCorpus` class that would handle other kind of inputs.

