# How can I update a checkpoint's vocabulary?

New vocabulary can be used to continue training from a checkpoint. Existing vocabulary embeddings will be mapped to the new vocabulary, and new vocabulary tokens will be initialized as usual.

Run `eole build_vocab` as usual with the new dataset. New vocabulary files will be created.

Training options to perform vocabulary update are:

* `-update_vocab`: set this option
* `-reset_optim`: set the value to "states"
* `-train_from`: checkpoint path