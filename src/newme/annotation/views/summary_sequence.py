from dataclasses import dataclass

from newme.annotation.types import (
    Context,
    WMNType,
    WMNMeaning,
)
from newme.annotation.models import (
    LabelData,
    AnnotatedExcerpt
)


@dataclass
class SummarySequence:
    wmn_id: str
    dialogue_id: str
    corpus_codename: str
    context: Context
    wmn_type: WMNType
    wmn_meaning: WMNMeaning
    triggers: dict
    indicators: dict
    labeldata: LabelData = None #TODO: remove
    annotated_utterance: AnnotatedExcerpt = None #TODO: remove
    excerpt: str = None #TODO: remove
    label_count: int = None #TODO: remove
