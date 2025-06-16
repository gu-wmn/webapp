from dataclasses import dataclass

from newme.annotation.types import WMNType, WMNMeaning, LabelName
from newme.corpus.types import CorpusName

@dataclass
class SummaryLabel:
    label_type: LabelName
    excerpt: str
    count: int
    dialogue_ids: list[str]
    sequence_ids: dict
    corpora: list[CorpusName]
    contexts: list[str]
    wmn_types: list[WMNType]
    wmn_meanings: list[WMNMeaning]
