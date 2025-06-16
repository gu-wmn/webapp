from newme.annotation.types import (
    Context,
    WMNType,
    WMNMeaning
)
import newme.corpus as corpus


class WMNSequence:
    wmn_id: str
    corpus_codename: str
    corpus_fullname: str
    corpus_url: str
    corpus_license_url: str
    dialogue_id: str
    context: Context
    annotator: str
    wmn: WMNType
    wmn_meaning: WMNMeaning
    utterances: list[type[corpus.models.Utterance]]
    labels: list[type[dict]]
    comment: str
    sibling_wmn_ids: list[str]

    def __init__(
        self,
        annotation_data: dict,
        utterances: list[type[corpus.models.Utterance]],
        meta_data: corpus.models.MetaData,
        sibling_wmn_ids: list[str]
    ):
        self.corpus_codename = annotation_data['corpus_codename']
        self.corpus_fullname= meta_data.fullname
        self.corpus_url = meta_data.url
        self.corpus_license_url = meta_data.license_url
        self.dialogue_id = annotation_data['dialogue_id']
        self.context = Context(annotation_data['context'])
        self.annotator = annotation_data['annotator']
        self.wmn_id = annotation_data['wmn_id']
        self.sibling_wmn_ids = sibling_wmn_ids
        self.wmn = WMNType(annotation_data['wmn'])
        self.wmn_meaning = WMNMeaning(annotation_data['wmn_meaning'])
        self.utterances = utterances
        self.index_first_label = False
        self.index_last_label = False
        # TODO turn labels into Label and adjust web view code as well
        self.labels = annotation_data['labels']
