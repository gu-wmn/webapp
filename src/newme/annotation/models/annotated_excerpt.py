from newme.corpus.models import Utterance
from newme.annotation.models import LabelData


class AnnotatedExcerpt:
    utterance: Utterance
    labeled_text: str = None
    text_before: str = None
    text_after: str = None

    def __init__(
        self,
        utterance: Utterance,
        labeldata: LabelData = None
    ):
        self.utterance = utterance

        if labeldata:
            self.labeled_text = utterance.text[labeldata.start_offset:labeldata.end_offset]

            if labeldata.start_offset > 0:
                if labeldata.start_offset > 56:
                    self.text_before = "..." + utterance.text[labeldata.start_offset - 55:labeldata.start_offset]
                else:
                    self.text_before = utterance.text[:labeldata.start_offset]
            else:
                self.text_before = ""

            if labeldata.end_offset < len(utterance.text):
                if labeldata.end_offset < (len(utterance.text) - 56):
                    self.text_after = utterance.text[labeldata.end_offset:labeldata.end_offset + 55] + "..."
                else:
                    self.text_after = utterance.text[labeldata.end_offset:]
            else:
                self.text_after = ""
