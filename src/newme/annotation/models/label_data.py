from dataclasses import dataclass
from newme.annotation.types import LabelName

@dataclass
class LabelData():
    name: LabelName
    start_index: int
    end_index: int
    start_offset: int
    end_offset: int
    excerpt: str
