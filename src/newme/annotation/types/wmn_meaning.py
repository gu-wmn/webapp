from enum import Enum

class WMNMeaning(Enum):
    BOTH = "both"
    SITUATED_MEANING = "situated meaning"
    POTENTIAL_MEANING = "potential meaning"
    NO_WMN = "no WMN"

    @property
    def value(self):
        if self == WMNMeaning.BOTH:
            return "Situated and potential meaning"
        elif self == WMNMeaning.SITUATED_MEANING:
            return "Situated meaning"
        elif self == WMNMeaning.POTENTIAL_MEANING:
            return "Potential meaning"
        elif self == WMNMeaning.NO_WMN:
            return "No WMN"
