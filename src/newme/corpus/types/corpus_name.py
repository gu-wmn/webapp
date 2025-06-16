from enum import Enum

# list of corpora in enum format
class CorpusName(Enum):
    BNC = ("bnc", "British National Corpus")
    WINNING_ARGS = ("winning-args-corpus", "Winning Arguments (ChangeMyView) Corpus")
    SWITCHBOARD = ("switchboard-corpus", "Switchboard Dialog Act Corpus")

    def __new__(cls, codename, fullname):
        obj = object.__new__(cls)
        obj._value_ = codename
        obj.codename = codename
        obj.fullname = fullname
        return obj
