import torch
from eole.models.model import get_model_class
from eole.models.model_saver import load_checkpoint
from eole.inputters.inputter import dict_to_vocabs

from eole.utils.logging import logger
from eole.bin import BaseBin, register_bin


def write_embeddings(filename, vocab, embeddings):
    with open(filename, "wb") as file:
        for i in range(min(len(embeddings), len(vocab))):
            str = vocab.lookup_index(i).encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + (" %5f" % (embeddings[i][j])).encode("utf-8")
            file.write(str + b"\n")


@register_bin(name="extract_embeddings")
class ExtractEmbeddings(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("-model", required=True, help="Path to model .pt file")
        parser.add_argument("-output_dir", default=".", help="""Path to output the embeddings""")
        parser.add_argument("-gpu", type=int, default=-1, help="Device to run on")

    @classmethod
    def run(cls, args):
        args.cuda = args.gpu > -1
        if args.cuda and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)

        # Add in default model arguments, possibly added since training.
        checkpoint = load_checkpoint(args.model)

        vocabs = dict_to_vocabs(checkpoint["vocab"])
        src_vocab = vocabs["src"]  # assumes src is text
        tgt_vocab = vocabs["tgt"]

        model_config = checkpoint["config"].model

        model = get_model_class(model_config).build_base_model(model_config, vocabs, running_config=None)
        model.load_safe_state_dict(args.model)

        encoder_embeddings = model.src_emb.embeddings.weight.data.tolist()
        decoder_embeddings = model.tgt_emb.embeddings.weight.data.tolist()

        logger.info("Writing source embeddings")
        write_embeddings(args.output_dir + "/src_embeddings.txt", src_vocab, encoder_embeddings)

        logger.info("Writing target embeddings")
        write_embeddings(args.output_dir + "/tgt_embeddings.txt", tgt_vocab, decoder_embeddings)

        logger.info("... done.")
        logger.info("Converting model...")
