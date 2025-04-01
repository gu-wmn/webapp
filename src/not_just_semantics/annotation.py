import json
import os
import sys
from dataclasses import dataclass
from typing import Type
from enum import Enum
from .corpus import (Corpus, Utterance, MetaData)


'''
ENUMS
'''

class WMN(Enum):
    NON = "WMN: non-understanding"
    DIN = "WMN: disagreement"
    Other = "WMN: other"
    SIMN = "SIMN"
    Other_Clar_Req = "Other kinds of clarification requests"
    No_Trigger = "Without trigger"
    Non_Pursued = "Non-pursued"
    Impossible = "Impossible to annotate"
    Reference_NE = "reference/NE"
    #Nothing = "Nothing"
    DUP_NON = "Already annotated # WMN: non-understanding"
    DUP_DIN = "Already annotated # WMN: disagreement"
    DUP_WMN_Other = "Already annotated # WMN: other"
    DUP_SIMN = "Already annotated # SIMN"
    DUP_Other_Clar_Req = "Already annotated # Other kinds of clarification requests"
    DUP_No_Trigger = "Already annotated # Without trigger"
    DUP_Non_Pursued = "Already annotated # Non-pursued"

    # @property
    # def value(self):
    #     if self == WMN.NON:
    #         return "Non-understanding"
    #     elif self == WMN.DIN:
    #         return "Disagreement"
    #     elif self == WMN.Other:
    #         return "Other"
    #     else:
    #         return self._value_


class WMNMeaning(Enum):
    Both = "both"
    Situated_Meaning = "situated meaning"
    Potential_Meaning = "potential meaning"
    No_WMN = "no WMN"

    @property
    def value(self):
        if self == WMNMeaning.Both:
            return "Situated and potential meaning"
        elif self == WMNMeaning.Situated_Meaning:
            return "Situated meaning"
        elif self == WMNMeaning.Potential_Meaning:
            return "Potential meaning"
        elif self == WMNMeaning.No_WMN:
            return "No WMN"


class Context(Enum):
    Spoken = "Spoken interaction"
    Online = "Online interaction"


class LabelName(Enum):
    Trigger = "Trigger"
    Indicator = "Indicator"
    Negotiation = "Negotiation"
    Trigger_Reference = "Trigger reference"


'''
DATA CLASSES
'''
@dataclass
class SearchParameters():
    wmn_id: int = None
    corpus_codename: str = None
    dialogue_id: str = None
    context: Context = None
    annotator: str = None
    wmn: list[WMN] = None
    label_name: LabelName = None
    text_includes: str = None
    include_predictions: bool = False


@dataclass
class LabelData():
    labelname: LabelName
    start_index: int
    end_index: int
    start_offset: int
    end_offset: int
    excerpt: str


@dataclass
class Prediction:
    model_version: str
    label: LabelData
    start_index: int
    end_index: int
    start_offset: int
    end_offset: int
    excerpt: str


class AnnotatedUtterance:
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


@dataclass
class WMNSummary:
    wmn_id: str
    context: Context
    wmn: WMN
    wmn_meaning: WMNMeaning
    labeldata: LabelData
    label_count: int
    excerpt: str
    annotated_utterance: AnnotatedUtterance


class WMNSequence:
    wmn_id: str
    corpus_codename: str
    corpus_fullname: str
    corpus_url: str
    corpus_license_url: str
    dialogue_id: str
    context: Context
    annotator: str
    wmn: WMN
    wmn_meaning: WMNMeaning
    utterances: list[type[Utterance]]
    labels: list[type[dict]]
    prediction: Prediction
    comment: str


    def __init__(
        self,
        annotation_data: dict,
        utterances: list[type[Utterance]],
        meta_data: MetaData
    ):
        self.corpus_codename = annotation_data['corpus_codename']
        self.corpus_fullname= meta_data.fullname
        self.corpus_url = meta_data.url
        self.corpus_license_url = meta_data.license_url
        self.dialogue_id = annotation_data['dialogue_id']
        self.context = Context(annotation_data['context'])
        self.annotator = annotation_data['annotator']
        self.wmn = WMN(annotation_data['wmn'])
        self.wmn_meaning = WMNMeaning(annotation_data['wmn_meaning'])
        self.utterances = utterances
        self.prediction = Prediction(
            model_version = annotation_data['prediction']['model_version'],
            label = LabelName(annotation_data['prediction']['label']['name']),
            start_index = int(annotation_data['prediction']['label']['start_index']),
            end_index = int(annotation_data['prediction']['label']['end_index']),
            start_offset = int(annotation_data['prediction']['label']['start_offset']),
            end_offset = int(annotation_data['prediction']['label']['end_offset']),
            excerpt = annotation_data['prediction']['label']['excerpt']
        )
        self.index_first_label = False
        self.index_last_label = False
        # TODO turn labels into Label and adjust web view code as well
        self.labels = annotation_data['labels']


class Annotation:

    def __init__(self, data_path: str):
        _annotation_file_path = os.path.join(
            os.path.dirname(__file__),
            "wmn_annotations.json"
        )
        with open(_annotation_file_path) as annotations_file:
            self._wmn_sequences = json.load(annotations_file)

        self._corpus = Corpus(data_path=data_path)
        self._corpus.load_corpora() # TODO: change this behaviour


    def _create_summary(self, sequence: dict, label: dict) -> WMNSummary:
        labeldata = LabelData(
            labelname = LabelName(label['name']),
            start_index = int(label['start_index']),
            end_index = int(label['end_index']),
            start_offset = int(label['start_offset']),
            end_offset = int(label['end_offset']),
            excerpt = label['excerpt']
        )
        annotated_utterance = AnnotatedUtterance(
            utterance = self._corpus.get_utterance(
                sequence['corpus_codename'],
                sequence['dialogue_id'],
                label['start_index']
            ),
            labeldata = labeldata
        )
        summary = WMNSummary(
                wmn_id = sequence['wmn_id'],
                context = Context(sequence['context']),
                wmn = WMN(sequence['wmn']),
                wmn_meaning = WMNMeaning(sequence['wmn_meaning']),
                labeldata = labeldata,
                excerpt = label['excerpt'],
                annotated_utterance = annotated_utterance,
                label_count = 1
        )
        return summary


    def match_and_get_summaries(
        self,
        search_parameters: "Annotation.SearchParameters"
    ) -> list:

        summaries = []

        for sequence in self._wmn_sequences:
            # so we don't add duplicate triggers or indicators
            summaries_to_append = {
                LabelName.Trigger: {},
                LabelName.Indicator: {}
            }

            # we are only showing summaries that has labeled text
            if not sequence['labels']:
                continue

            # we only care about sequences that have one of these WMN types
            if not any(sequence['wmn'] == wmn.value for wmn in [
                WMN.NON,
                WMN.DIN,
                WMN.Other,
                WMN.SIMN,
                WMN.Other_Clar_Req,
                WMN.No_Trigger,
                WMN.Non_Pursued,
                WMN.Impossible,
                WMN.Reference_NE
            ]):
                continue

            if search_parameters.wmn_id \
            and search_parameters.wmn_id != sequence['wmn_id']:
                continue

            if search_parameters.corpus_codename \
            and search_parameters.corpus_codename != sequence['corpus_codename']:
                continue

            if search_parameters.context \
            and search_parameters.context != Context(sequence['context']):
                continue

            if search_parameters.annotator \
            and search_parameters.annotator != sequence['annotator']:
                continue

            if search_parameters.wmn:
                if not any(WMN(sequence['wmn']) == param for param in search_parameters.wmn):
                    continue

            for label in sequence['labels']:
                if search_parameters.text_includes:
                    if search_parameters.text_includes not in label['excerpt'] \
                    and search_parameters.text_includes not in sequence['wmn_id']:
                        continue
                if (
                    search_parameters.label_name
                    and LabelName(label['name']) != search_parameters.label_name
                ):
                    continue
                # we only list triggers and indicators
                if label['name'] not in [LabelName.Trigger.value, LabelName.Indicator.value]:
                    continue

                # matching is done, now create or update summary for this label

                labelname = LabelName(label['name'])
                nocase_excerpt = label['excerpt'].casefold()

                if (
                    nocase_excerpt in summaries_to_append[labelname].keys()
                    and nocase_excerpt == \
                    summaries_to_append[labelname][nocase_excerpt].labeldata.excerpt.casefold()
                ):
                    summaries_to_append[labelname][nocase_excerpt].label_count = \
                    summaries_to_append[labelname][nocase_excerpt].label_count + 1
                else:
                    summaries_to_append[labelname][nocase_excerpt] = self._create_summary(sequence, label)

            for summary in summaries_to_append[LabelName.Trigger].values():
                summaries.append(summary)
            for summary in summaries_to_append[LabelName.Indicator].values():
                summaries.append(summary)

        return summaries


    def get_wmn_sequence(self, wmn_id: str) -> WMNSequence:
        sequence = next((seq for seq in self._wmn_sequences if seq['wmn_id'] == wmn_id), None)

        return WMNSequence(
            annotation_data = sequence,
            utterances = self._corpus.get_dialogue(
                sequence['corpus_codename'],
                sequence['dialogue_id']
            ),
            meta_data = self._corpus.get_metadata(sequence['corpus_codename'])
        )
