# How can I ensemble Models at inference?

You can specify several models in the `onmt_translate` command line: `-model model1_seed1 model2_seed2`
Bear in mind that your models must share the same target vocabulary.