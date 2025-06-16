from dataclasses import dataclass

from newme.annotation.types import WMNType, WMNMeaning
from newme.corpus.types import CorpusName

@dataclass
class SummaryDialogue:
    dialogue_id: str
    corpus_fullname: CorpusName
    context: str
    sequence_ids: list[str]
    wmn_types: list[WMNType]
    wmn_meanings: list[WMNMeaning]
    triggers: dict
    indicators: dict
