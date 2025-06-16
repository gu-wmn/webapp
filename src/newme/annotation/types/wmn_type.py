from enum import Enum
from dataclasses import dataclass

#from newme.annotation.types import WMNMeaning


# TODO: For later, perhaps
# @dataclass
# class __WMNTypeProperties:
#     value: str
#     meanings: list
#     hidden: bool


class WMNType(Enum):
    NON = "WMN: non-understanding" # TODO: add two more fields to all props
    DIN = "WMN: disagreement"
    OTHER = "WMN: other"
    SIMN = "SIMN"
    OTHER_CLAR_REQ = "Other kinds of clarification requests"
    NO_TRIGGER = "Without trigger"
    NON_PURSUED = "Non-pursued"
    IMPOSSIBLE = "Impossible to annotate"
    REFERENCE_NE = "reference/NE"
    #Nothing = "Nothing"
    #DUP_NON = "Already annotated # WMN: non-understanding"
    #DUP_DIN = "Already annotated # WMN: disagreement"
    #DUP_WMN_Other = "Already annotated # WMN: other"
    #DUP_SIMN = "Already annotated # SIMN"
    #DUP_Other_Clar_Req = "Already annotated # Other kinds of clarification requests"
    #DUP_No_Trigger = "Already annotated # Without trigger"
    #DUP_Non_Pursued = "Already annotated # Non-pursued"
