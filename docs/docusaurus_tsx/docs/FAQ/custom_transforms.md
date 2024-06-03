
# How can I create custom on-the-fly data transforms?

The code is easily extendable with custom transforms inheriting from the `Transform` base class.

You can for instance have a look at the `FilterTooLongTransform` class as a template:

```python
@register_transform(name='filtertoolong')
class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Filter")
        group.add("--src_seq_length", "-src_seq_length", type=int, default=200,
                  help="Maximum source sequence length.")
        group.add("--tgt_seq_length", "-tgt_seq_length", type=int, default=200,
                  help="Maximum target sequence length.")

    def _parse_opts(self):
        self.src_seq_length = self.opts.src_seq_length
        self.tgt_seq_length = self.opts.tgt_seq_length

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if too long else return as is."""
        if (len(example['src']) > self.src_seq_length or
                len(example['tgt']) > self.tgt_seq_length):
            if stats is not None:
                stats.update(FilterTooLongStats())
            return None
        else:
            return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format(
            'src_seq_length', self.src_seq_length,
            'tgt_seq_length', self.tgt_seq_length
        )
```

Methods:
- `add_options` allows to add custom options that would be necessary for the transform configuration;
- `_parse_opts` allows to parse options introduced in `add_options` when initialize;
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

This is defined in `onmt.inputters.corpus.ParallelCorpus.load`. This class is not easily extendable for now but it can be considered for future developments. For instance, we could create some `CustomParallelCorpus` class that would handle other kind of inputs.

