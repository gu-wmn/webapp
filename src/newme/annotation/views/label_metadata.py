from newme.annotation.types import LabelName
from dataclasses import dataclass

@dataclass
class LabelMetaData:
    labelname: LabelName
    excerpt: str
    count: int
    dialogue_ids: list[str]
    sequence_ids: dict
    # wmn_sequences: list[dict]
