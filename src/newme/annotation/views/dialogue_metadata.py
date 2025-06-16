from dataclasses import dataclass

from newme.annotation.models import WMNData
from newme.corpus.models import MetaData

@dataclass
class DialogueMetaData:
    dialogue_id: str
    corpus_metadata: MetaData
    wmn_sequences: list[WMNData]
