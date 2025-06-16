import json
import os

import newme.corpus as corpus
from newme.corpus.types import CorpusName

from newme.annotation.types import (
    Context,
    GroupBy,
    LabelName,
    WMNMeaning,
    WMNType
)
from newme.annotation.models import (
    Filter,
    WMNData
)
from newme.annotation.views import (
    DialogueMetaData,
    LabelMetaData,
    SummaryDialogue,
    SummarySequence,
    SummaryLabel,
    WMNSequence
)


class Annotation:

    filter: Filter
    __corpus: corpus.Corpus

    def __init__(self, data_path: str):

        _annotation_file_path = os.path.join(
            os.path.dirname(__file__),
            "wmn_annotations.json"
        )
        with open(_annotation_file_path) as annotations_file:
            self.__wmn_sequences = json.load(annotations_file)

        self.__corpus = corpus.Corpus(data_path=data_path)
        self.__corpus.load_corpora() # TODO: change this behaviour


    def get_summaries(self):

        if self.filter.group_by == GroupBy.SEQUENCE:
            results = self.__get_summaries_by_sequence()
        elif self.filter.group_by == GroupBy.LABEL:
            results = self.__get_summaries_by_label()
        else:
            results = self.__get_summaries_by_dialogue()

        return results


    def get_dialogue_metadata(self, dialogue_id: str):

        selected_wmns = []
        corpus_codename = None

        for wmn in self.__wmn_sequences:
            if (
                wmn['dialogue_id'] == dialogue_id
                and wmn['wmn'] in [wmn_type.value for wmn_type in WMNType]
            ):
                corpus_codename = wmn['corpus_codename']
                selected_labels = {
                    LabelName.TRIGGER: {},
                    LabelName.INDICATOR: {},
                    LabelName.NEGOTIATION: {}
                }
                if wmn['labels']:
                    for label in wmn['labels']:
                        labelname = LabelName(label['name'])
                        if not label['excerpt'] in selected_labels[labelname].keys():
                            selected_labels[labelname][label['excerpt']] = 1
                        else:
                            selected_labels[labelname][label['excerpt']] += 1

                    selected_wmns.append(
                        WMNData(
                            wmn_id = wmn['wmn_id'],
                            wmn_type = WMNType(wmn['wmn']),
                            wmn_meaning = WMNMeaning(wmn['wmn_meaning']),
                            context = Context(wmn['context']),
                            triggers = selected_labels[LabelName.TRIGGER] or None,
                            indicators = selected_labels[LabelName.INDICATOR] or None,
                            negotiations = selected_labels[LabelName.NEGOTIATION] or None
                        )
                    )

        if selected_wmns:
            return DialogueMetaData(
                dialogue_id = dialogue_id,
                corpus_metadata = self.__corpus.get_metadata(corpus_codename),
                wmn_sequences = selected_wmns
            )
        else:
            print("Could not get dialogue metadata", flush=True)


    def get_label_metadata(
        self,
        excerpt: str
    ):
        label_metadata = LabelMetaData(
            labelname = None,
            excerpt = excerpt,
            count = 0,
            dialogue_ids = set(),
            sequence_ids = {},
            # wmn_sequences = list()
        )
        for wmn_sequence in self.__wmn_sequences:
            if not wmn_sequence['labels']:
                continue

            found_match = False
            wmn_count = 0
            for label in wmn_sequence['labels']:
                if label['excerpt'].casefold().strip() == excerpt:
                    found_match = True
                    wmn_count += 1
                    label_metadata.count += 1
                    label_metadata.labelname = LabelName(label['name'])

            if found_match:
                label_metadata.dialogue_ids.add(
                    wmn_sequence['dialogue_id']
                )
                label_metadata.sequence_ids[wmn_sequence['wmn_id']] = {
                    'dialogue_id': wmn_sequence['dialogue_id'],
                    'wmn_count': wmn_count
                }
                # label_metadata.wmn_sequences.append(
                #     wmn_sequence
                # )
        print(label_metadata, flush=True)
        return label_metadata


    def get_wmn_sequence(
        self,
        dialogue_id: str,
        wmn_id: str
    ) -> WMNSequence:

        wmn_dict = None
        sibling_wmn_ids = []
        for wmn in self.__wmn_sequences:
            if not wmn['wmn'] in [wmn_type.value for wmn_type in WMNType]:
                continue
            if wmn['wmn_id'] == wmn_id:
               wmn_dict = wmn
            if wmn['dialogue_id'] == dialogue_id:
                sibling_wmn_ids.append(wmn['wmn_id'])

        return WMNSequence(
            annotation_data = wmn_dict,
            utterances = self.__corpus.get_dialogue(
                wmn_dict['corpus_codename'],
                wmn_dict['dialogue_id']
            ),
            meta_data = self.__corpus.get_metadata(wmn_dict['corpus_codename']),
            sibling_wmn_ids = sibling_wmn_ids
        )


    def __get_summaries_by_sequence(self)-> list:

        results = []
        for wmn_sequence in self.__wmn_sequences:
            if not wmn_sequence['labels']:
                continue
            if not self.__is_matching_wmn(wmn_sequence):
                continue

            triggers = {}
            indicators = {}

            for label in wmn_sequence['labels']:
                if not self.__is_matching_label(
                    label,
                    wmn_sequence['dialogue_id'],
                    wmn_sequence['wmn_id']
                ):
                    continue

                excerpt = label['excerpt'].lower()
                if label['name'] == LabelName.TRIGGER.value:
                    if not excerpt in triggers.keys():
                        triggers[excerpt] = 1
                    else:
                        triggers[excerpt] += 1
                elif label['name'] == LabelName.INDICATOR.value:
                    if not excerpt in indicators.keys():
                        indicators[excerpt] = 1
                    else:
                        indicators[excerpt] += 1

            if triggers or indicators:
                results.append(
                    SummarySequence(
                        wmn_id = wmn_sequence['wmn_id'],
                        dialogue_id = wmn_sequence['dialogue_id'],
                        corpus_codename = wmn_sequence['corpus_codename'],
                        context = Context(wmn_sequence['context']),
                        wmn_type = WMNType(wmn_sequence['wmn']),
                        wmn_meaning = WMNMeaning(wmn_sequence['wmn_meaning']),
                        triggers = triggers,
                        indicators = indicators
                    )
                )

        return results


    def __get_summaries_by_dialogue(self):

        results = {}

        for wmn_sequence in self.__wmn_sequences:
            if not wmn_sequence['labels']:
                continue
            # if not wmn_sequence['wmn'] in WMNType.__members__:
            #     continue
            if not self.__is_matching_wmn(wmn_sequence):
                continue

            dialogue_id = wmn_sequence['dialogue_id']

            matching_triggers = {}
            matching_indicators = {}

            for label in wmn_sequence['labels']:
                if not self.__is_matching_label(
                    label,
                    dialogue_id,
                    wmn_sequence['wmn_id']
                ):
                    continue
                if LabelName(label['name']) == LabelName.TRIGGER:
                    if not label['excerpt'] in matching_triggers.keys():
                        matching_triggers[label['excerpt']] = 1
                    else:
                        matching_triggers[label['excerpt']] += 1
                elif LabelName(label['name']) == LabelName.INDICATOR:
                    if not label['excerpt'] in matching_indicators.keys():
                        matching_indicators[label['excerpt']] = 1
                    else:
                        matching_indicators[label['excerpt']] += 1
            if (
                not matching_triggers
                and not matching_indicators
            ):
                continue

            if not dialogue_id in results.keys():
                results[dialogue_id] = SummaryDialogue(
                    dialogue_id = dialogue_id,
                    corpus_fullname = CorpusName(wmn_sequence['corpus_codename']),
                    context = Context(wmn_sequence['context']),
                    sequence_ids = {wmn_sequence['wmn_id']},
                    wmn_types = {WMNType(wmn_sequence['wmn'])},
                    wmn_meanings = {WMNMeaning(wmn_sequence['wmn_meaning'])},
                    triggers = {},
                    indicators = {}
                )
            else:
                results[dialogue_id].sequence_ids.add(wmn_sequence['wmn_id'])
                results[dialogue_id].wmn_types.add(WMNType(wmn_sequence['wmn']))
                results[dialogue_id].wmn_meanings.add(WMNMeaning(wmn_sequence['wmn_meaning']))

            for excerpt, amount in matching_triggers.items():
                if not excerpt in results[dialogue_id].triggers.keys():
                    results[dialogue_id].triggers[excerpt] = amount
                else:
                    results[dialogue_id].triggers[excerpt] += amount
            for excerpt, amount in matching_indicators.items():
                if not excerpt in results[dialogue_id].indicators.keys():
                    results[dialogue_id].indicators[excerpt] = amount
                else:
                    results[dialogue_id].indicators[excerpt] += amount

        return list(results.values())


    def __get_summaries_by_label(self) -> list:

        results = {}

        for wmn_sequence in self.__wmn_sequences:
            # we are only showing summaries that has labeled text
            if not wmn_sequence['labels']:
                continue
            # if not wmn_sequence['wmn'] in [wmn_type for wmn_type in WMNType]:
            #     continue
            # continue only if there are matches
            if not self.__is_matching_wmn(wmn_sequence):
                continue

            for label in wmn_sequence['labels']:
                if not self.__is_matching_label(
                    label,
                    wmn_sequence['dialogue_id'],
                    wmn_sequence['wmn_id']
                ):
                    continue

                excerpt = label['excerpt'].casefold().strip()
                if not excerpt in results.keys():
                    results[excerpt] = SummaryLabel(
                        label_type = LabelName(label['name']),
                        excerpt = excerpt,
                        count = 1,
                        dialogue_ids = {wmn_sequence['dialogue_id']},
                        sequence_ids = {
                            wmn_sequence['wmn_id']: wmn_sequence['dialogue_id']
                        },
                        corpora = {CorpusName(wmn_sequence['corpus_codename'])},
                        contexts = {Context(wmn_sequence['context'])},
                        wmn_types = {WMNType(wmn_sequence['wmn'])},
                        wmn_meanings = {WMNMeaning(wmn_sequence['wmn_meaning'])},
                    )
                else:
                    results[excerpt].count += 1
                    results[excerpt].dialogue_ids.add(
                        wmn_sequence['dialogue_id']
                    )
                    if not wmn_sequence['wmn_id'] in results[excerpt].sequence_ids.keys():
                        results[excerpt].sequence_ids[wmn_sequence['wmn_id']] = wmn_sequence['dialogue_id']
                    results[excerpt].corpora.add(
                        CorpusName(wmn_sequence['corpus_codename'])
                    )
                    results[excerpt].contexts.add(
                        Context(wmn_sequence['context'])
                    )
                    results[excerpt].wmn_types.add(
                        WMNType(wmn_sequence['wmn'])
                    )
                    results[excerpt].wmn_meanings.add(
                        WMNMeaning(wmn_sequence['wmn_meaning'])
                    )

        results = list(results.values())
        results.sort(key=lambda x: x.excerpt)

        return results


    def __is_matching_wmn(
        self,
        wmn_sequence: dict
    ) -> bool:
        # we are only showing summaries that has labeled text
        if not wmn_sequence['labels']:
            return False

        if not any(
            wmn_sequence['wmn'] == wmn.value
            for wmn in WMNType
        ):
            return False

        if (
            self.filter.wmn_id
            and self.filter.wmn_id != wmn_sequence['wmn_id']
        ):
            return False

        if (
            self.filter.corpus_codename
            and self.filter.corpus_codename != CorpusName(wmn_sequence['corpus_codename'])
        ):
            return False

        if (
            self.filter.context
            and self.filter.context != Context(wmn_sequence['context'])
        ):
            return False

        if (
            self.filter.annotator
            and self.filter.annotator != wmn_sequence['annotator']
        ):
            return False

        if self.filter.wmn_types:
            if not any(WMNType(wmn_sequence['wmn']) == param for param in self.filter.wmn_types):
                return False

        return True


    def __is_matching_label(
        self,
        label: dict,
        dialogue_id,
        wmn_id: str,
    ) -> bool:

        # we only list triggers and indicators
        if label['name'] not in [
            LabelName.TRIGGER.value,
            LabelName.INDICATOR.value
        ]:
            return False

        if (
            self.filter.label_name
            and LabelName(label['name']) != self.filter.label_name
        ):
            return False

        if self.filter.text_includes:
            if (
                self.filter.text_includes not in label['excerpt'].casefold().strip()
                and self.filter.text_includes not in dialogue_id
                and self.filter.text_includes not in wmn_id
            ):
                return False

        return True
