# converts a SentencePiece vocabulary to the format expected by dynamic data
# (essentially converts float expected counts to "fixed precision" int pseudo
# counts)
import sys
import math
from eole.constants import DefaultTokens

from eole.bin import BaseBin, register_bin


@register_bin(name="spm_to_vocab")
class SpmtoVocab(BaseBin):
    OMIT = (DefaultTokens.UNK, DefaultTokens.BOS, DefaultTokens.EOS)

    @classmethod
    def add_args(cls, parser):
        # legacy tool directly reads lines from stdin
        pass

    @classmethod
    def convert(cls, lines):
        for line in lines:
            w, c = line.rstrip("\n").split(None, 1)
            if w in cls.OMIT:
                continue
            c = math.exp(float(c)) * 1000000
            c = int(c) + 1
            yield w, c

    @classmethod
    def run(cls, args):
        for c, w in cls.convert(sys.stdin):
            print("{}\t{}".format(c, w))
