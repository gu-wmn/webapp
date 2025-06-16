from dataclasses import dataclass

from newme.annotation.types import (
    WMNType,
    WMNMeaning,
    Context
)
from newme.annotation.models import LabelData

@dataclass
class WMNData:
    wmn_id: str
    wmn_type: WMNType
    wmn_meaning: WMNMeaning
    context: Context
    triggers: dict
    indicators: dict
    negotiations: dict
