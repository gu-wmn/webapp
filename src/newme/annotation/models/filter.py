from dataclasses import dataclass, field
from newme.annotation.types import Context, WMNType, LabelName

@dataclass
class Filter():
    wmn_id: int = None
    corpus_codename: str = None
    dialogue_id: str = None
    context: Context = None
    annotator: str = None
    wmn_types: list[WMNType] = field(default_factory=lambda: [])
    label_name: LabelName = None
    text_includes: str = None
    group_by: str = None
