import os
#import convokit
import shutil
import hashlib
import zipfile
import requests
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import json
import regex as re


@dataclass
class MetaData:
    fullname: str
    license_url: str
    url: str


@dataclass
class Utterance:
    author: str
    text: str
    author_plain: str = None
    reply_to: str = None
    utt_order_num: str = None


class Corpus:
    '''Handles all interaction with corpus data.
    Downloads, extracts and handles searches in corpus.'''

    corpora: dict = {
        'bnc': {
            'fullname': 'British National Corpus',
            'license_url': 'http://www.natcorp.ox.ac.uk/docs/licence.html',
            'url': 'http://www.natcorp.ox.ac.uk/',
            'download_url': 'https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/2554/2554.zip?sequence=4&isAllowed=y',
            'md5sum': '5fdea535beb437d232af2a0217e92bf1',
            'dialogues': []
        },
        'winning-args-corpus': {
            'fullname': 'Winning Arguments (ChangeMyView) Corpus',
            'license_url': '',
            'url': 'https://convokit.cornell.edu/documentation/winning.html',
            'download_url': 'https://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/winning-args-corpus.zip',
            'md5sum': 'd17056c46f4805a83cd192e56900863f',
            'dialogues': []
        },
        'switchboard-corpus': {
            'fullname': 'Switchboard Dialog Act Corpus',
            'license_url': 'https://creativecommons.org/licenses/by-nc-sa/3.0/',
            'url': 'http://compprag.christopherpotts.net/swda.html',
            'download_url': 'https://zissou.infosci.cornell.edu/convokit/datasets/switchboard-corpus/switchboard-corpus.zip',
            'md5sum': '714650ec360b91ba37452e4f7448c44a',
            'dialogues': []
        }

    }

    _extracted_corpora_path: str
    _original_corpora_dir: str

    _convokit_corpuses =  [
        "winning-args-corpus",
        "switchboard-corpus",
    ]


    def __init__(self, data_path="./"):

        self._original_corpora_dir = os.path.join(
            data_path,
            "corpora",
            "original_corpora"
        )

        self._extracted_corpora_path = os.path.join(
            data_path,
            "corpora",
            "extracted_corpora.json"
        )

        if not os.path.isdir(self._original_corpora_dir):
            os.makedirs(self._original_corpora_dir, exist_ok=True)

        self._annotated_conversation_ids = []

    def get_utterance(
        self,
        corpus_codename,
        dialogue_id,
        utterance_index
    ) -> Utterance:

        for dialogue in self.corpora[corpus_codename]['dialogues']:
            if dialogue['id'] == dialogue_id:
                utt = dialogue['utterances'][utterance_index]

        return Utterance(
            author = utt['author'],
            text = utt['text'],
            author_plain = utt['author_plain'] if 'author_plain' in utt.keys() else None,
            reply_to = utt['reply_to'] if 'reply_to' in utt.keys() else None,
            utt_order_num = utt['utt_order_num'] if 'utt_order_num' in utt.keys() else None
        )


    def get_dialogue(
        self,
        corpus_codename,
        dialogue_id
    ) -> list[Utterance]:
        utterances = []
        for dialogue in self.corpora[corpus_codename]['dialogues']:
            if dialogue['id'] == dialogue_id:
                for utt in dialogue['utterances']:
                    utterances.append(
                        Utterance(
                            author = utt['author'],
                            text = utt['text'],
                            # author_plain = utt['author_plain'] if 'author_plain' in utt.keys() else None,
                            # reply_to = utt['reply_to'] if 'reply_to' in utt.keys() else None,
                            # utt_order_num = utt['utt_order_num'] if 'utt_order_num' in utt.keys() else None
                        )
                    )
        return utterances


    def get_metadata(self, corpus_codename) -> MetaData:
        return MetaData(
            fullname = self.corpora[corpus_codename]['fullname'],
            license_url = self.corpora[corpus_codename]['license_url'],
            url = self.corpora[corpus_codename]['url']
        )


    def load_corpora(self) -> bool:

        if not os.path.isfile(self._extracted_corpora_path):
            self._get_annotated_conversation_ids()
            self._extract_corpora()

        if os.path.isfile(self._extracted_corpora_path):
            with open(self._extracted_corpora_path) as json_obj:
                self.corpora = json.load(json_obj)
        else:
            print("File not found:", self._extracted_corpora_path, flush=True)
            return False


    def _to_json(self)-> None:
        '''Save extracted corpus to json format'''

        print("Saving", self._extracted_corpora_path, flush="True")

        # json_object = json.dumps({'corpora': self.corpora }, indent=4)
        json_object = json.dumps(self.corpora)
        with open(self._extracted_corpora_path, "w") as json_file:
            json_file.write(json_object)


    def _download_and_unpack_corpora(self, corpusname) -> None:
        """Download and unpack corpus"""
        corpus = self.corpora[corpusname]

        # create dir, if needed
        corpus_dir = os.path.join(self._original_corpora_dir, corpusname)
        if not os.path.isdir(corpus_dir):
            os.makedirs(corpus_dir, exist_ok=True)

        corpus_zipfile = os.path.join(corpus_dir, corpusname + ".zip")

        # download zip file
        print("Downloading corpus: {}...".format(corpusname), end="", flush=True)
        try:
            with requests.get(corpus['download_url'], stream=True) as r:
                with open(corpus_zipfile, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            print("[done]", flush=True)
        except:
            print("[failed]", flush=True)
            print("Unable to download {} from {}. Skipping..".format(corpusname, corpus['download_url']), flush=True)

        # checksum
        print("Performing checksum on downloaded file... ", end="", flush=True)
        with open(corpus_zipfile, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        if corpus['md5sum'] != file_hash.hexdigest():
            print("[failed]", flush=True)
            print("Downloaded checksum did not match the expected one.", flush=True)
            print("Try running this application a second time to make sure we didn't have a flipped bit somewhere.", flush=True)
        else:
            print("[done]", flush=True)

        #unzip
        print("Unzipping './{}'... ".format(corpus_zipfile), end="", flush=True)
        try:
            with zipfile.ZipFile(corpus_zipfile, 'r') as zip_ref:
                zip_ref.extractall(corpus_dir)
            print("[done]", flush="true")
        except:
            print("[failed]", flush=True)


    def _extract_corpora(self) -> None:

        self._extract_swda()
        self._extract_winning_args()
        self._extract_bnc()

        self._to_json()


    def _get_annotated_conversation_ids(self) -> list:
        annotations_file = os.path.join(
            os.path.dirname(__file__),
            "wmn_annotations.json"
        )
        with open(annotations_file) as f:
            annotations = json.load(f)
            for wmn in annotations:
                if wmn['dialogue_id'] not in self._annotated_conversation_ids:
                    self._annotated_conversation_ids.append(wmn['dialogue_id'])


    def _extract_swda(self):

        utterance_file = os.path.join(
            self._original_corpora_dir,
            'switchboard-corpus',
            'switchboard-corpus',
            'utterances.jsonl'
        )
        if not os.path.isfile(utterance_file):
            self._download_and_unpack_corpora('switchboard-corpus')

        print("Exctracting switchboard-corpus...", end="", flush=True)

        with open(utterance_file) as f:
            utterances = list(f)

            conversations = {}

            for utt in utterances:
                utterance = json.loads(utt)
                # skip the ones that have no wmn
                if not utterance['conversation_id'] in self._annotated_conversation_ids:
                    continue
                # add the rest and keep only the data we care about
                if not utterance['conversation_id'] in conversations.keys():
                    conversations[utterance['conversation_id']] = []
                id = utterance['id']
                text = utterance['text'] # do processing first
                author = utterance['speaker']

                conversations[utterance['conversation_id']].append(
                    {
                        'id': id,
                        'text': text,
                        'author': author
                    }
                )
            for id, utterances in conversations.items():
                self.corpora['switchboard-corpus']['dialogues'].append(
                    {
                        'id': id,
                        'utterances': utterances
                    }
                )
        print("[Done]", flush=True)


    def _extract_winning_args(self):

        utterance_file = os.path.join(
            self._original_corpora_dir,
            'winning-args-corpus',
            'winning-args-corpus',
            'utterances.jsonl'
        )
        if not os.path.isfile(utterance_file):
            self._download_and_unpack_corpora('winning-args-corpus')

        print("Exctracting winning-args-corpus...", end="", flush=True)

        conversations = {}

        with open(utterance_file) as f:
            utterances = list(f)

        for str in utterances:
            utterance = json.loads(str)

            # stop processing if the utterance is not part of an annotated wmn
            if not utterance['root'] in self._annotated_conversation_ids:
                continue

            # Do some reformatting of the text, if needed
            if utterance['text']:
                citation_matches = re.finditer("&gt;.*?\n\n", utterance['text'])
                if citation_matches:
                    filtered_text = ''
                    citation_spans = [(m.span()[0], m.span()[1]) for m in citation_matches]
                    #if citation_spans:
                    decalage = 0
                    next_first_index = 0
                    for cit_span in citation_spans:
                        filtered_text += utterance['text'][next_first_index:cit_span[0]] + '[STA-CITE]' + utterance['text'][cit_span[0]:cit_span[1]] + '[END-CITE]'
                        decalage += len('[STA-CITE]') + len('[END-CITE]')
                        next_first_index = len(filtered_text) - decalage
                    filtered_text += utterance['text'][next_first_index:]
                    utterance['text'] = filtered_text

            # sometimes there is no reply-to
            if 'reply-to' in utterance.keys():
                reply_to = utterance['reply-to']
            else:
                reply_to = None

            if not utterance['root'] in conversations.keys():
                conversations[utterance['root']] = []
            # use only keys we need
            conversations[utterance['root']].append({
                'id': utterance['id'],
                'author': utterance['user'],
                'reply_to': reply_to,
                'text': utterance['text']
            })

        print("[Done]", flush=True)

        # re-order utterances
        print("Re-ordering utterances in dialogues for winning-args-corpus...", end="", flush=True)
        for conv_id, utterances in conversations.items():
            # REORDERING based primarily on response hierarchy (but keep track of original order)
            reply_chain_almost = dict()

            ### first check
            for utt in utterances:
                reply_chain_almost[utt['id']] = utt['reply_to']

            # extend the chain
            reply_chain = dict() # from utterance id to the tree of message ids that it replies to
            for k in reply_chain_almost:
                reply_chain[k] = [reply_chain_almost[k]]
                finished = False
                while not finished:
                    if None in reply_chain[k]:
                        finished = True
                    for other_k in reply_chain_almost:
                        if other_k != k and other_k in reply_chain[k] and reply_chain_almost[other_k] not in reply_chain[k]:
                            if reply_chain_almost[other_k] == None:
                                finished = True
                                break
                            else:
                                reply_chain[k].append(reply_chain_almost[other_k])

            ordered_utterances = []
            for utt in utterances:
                if utt['reply_to'] == None:
                    ordered_utterances.append(utt)
                else:
                    last_other_possible_reply_idx = None
                    for utt_idx, possible_parent_utt in enumerate(ordered_utterances):
                        if possible_parent_utt['id'] in reply_chain[utt['id']]:
                            last_other_possible_reply_idx = utt_idx

                    if last_other_possible_reply_idx is not None:
                        ordered_utterances = ordered_utterances[:last_other_possible_reply_idx+1] + [utt] + ordered_utterances[last_other_possible_reply_idx+1:]

            utterances = ordered_utterances
            # add title to the conversation
            conv_file = os.path.join(
                self._original_corpora_dir,
                'winning-args-corpus',
                'winning-args-corpus',
                'conversations.json'
            )
            with open(conv_file) as cf:
                convs = json.load(cf)
            utterances = [{'author': 'TITLE', 'text': convs[conv_id]['op-title'], 'id': conv_id}] + utterances

            self.corpora['winning-args-corpus']['dialogues'].append(
                {
                    'id': conv_id,
                    'utterances': utterances
                }
            )
        print("[Done]", flush=True)


    def _download_and_unpack_bnc(self) -> bool:
        '''Downloads and unpacks the British National Corpus'''

        bnc_url = "https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/2554/2554.zip?sequence=4&isAllowed=y"
        bnc_md5sum = "5fdea535beb437d232af2a0217e92bf1"
        bnc_corpora_dir = os.path.join(self._original_corpora_dir, "bnc")
        bnc_zip_file = os.path.join(bnc_corpora_dir, "bnc.zip")

        if not os.path.isdir(bnc_corpora_dir):
            os.makedirs(bnc_corpora_dir, exist_ok=True)

        # download
        print("Downloading BNC ... ", end="", flush=True)
        try:
            with requests.get(bnc_url, stream=True) as r:
                with open(bnc_zip_file, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            print("[done]", flush=True)
        except:
            print("[failed]", flush=True)
            print("Unable to download BNC. Exiting program", flush=True)

            return False

        # checksum
        print("Performing checksum on downloaded file... ", end="", flush=True)
        with open(bnc_zip_file, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)

        if bnc_md5sum != file_hash.hexdigest():
            print("[failed]", flush=True)
            print("Downloaded checksum did not match the expected one.", flush=True)
            print("Try running this script a second time to make sure we didn't have a flipped bit somewhere.", flush=True)

            return False

        print("[done]", flush=True)

        # unzip
        print("Unzipping './{}'... ".format(bnc_zip_file), end="", flush=True)

        try:
            with zipfile.ZipFile(bnc_zip_file, 'r') as zip_ref:
                zip_ref.extractall(bnc_corpora_dir)
            print("[done]", flush="true")
        except:
            print("[failed]", flush=True)
            return False

        return True


    def _extract_bnc(self) -> bool:
        '''Downloads (if needed) and extracts text from BNC and saves to json'''

        bnc_texts_dir = os.path.join(self._original_corpora_dir, "bnc", "download", "Texts")
        if not os.path.isdir(bnc_texts_dir):
            if not self._download_and_unpack_bnc():
                print("Warning: Failed to download 'British National Corpus'. Skipping..")
                return False

        conversations = {}

        skipped_utts = []

        print("Extracting text from bnc...", end="", flush=True)
        for letter in os.listdir(bnc_texts_dir):

            for secondletter in os.listdir(os.path.join(bnc_texts_dir, letter)):
                #print("Scanning files in '{}'".format(os.path.join("bnc_texts_dir", "letter", "secondletter")))

                for fn in os.listdir(os.path.join(bnc_texts_dir, letter, secondletter)):

                    # skip if no annotations are applied to this file
                    if not fn[:-4] in self._annotated_conversation_ids:
                        continue

                    tree = ET.parse(os.path.join(bnc_texts_dir, letter, secondletter, fn))
                    root = tree.getroot()
                    neighbors = [x for x in root.iter()]
                    found_stext = [x for x in neighbors if x.tag == "stext"]

                    if found_stext:
                        assert len(found_stext) == 1
                        stext = found_stext[0]
                        clean_utterances = [] # (author, text)
                        uttnum = -1

                        for utterance in stext:

                            if "who" not in utterance.attrib:
                                skipped_utts.append((fn, utterance))
                                continue

                            uttnum +=1
                            author = utterance.attrib['who']
                            words = []

                            for sentence in utterance:
                                if sentence.tag != "s":
                                    if sentence.tag == "unclear":
                                        words.append("[UNCLEAR]")

                                else:
                                    for word in sentence:
                                        if word.tag in ["w","c"]:
                                            words.append(word.text.strip())

                                        elif word.tag in ["align","pause","shift","event","vocal"]:
                                            continue

                                        elif word.tag == "unclear":
                                            words.append("[UNCLEAR]")

                                        elif word.tag == "gap":
                                            if "reason" in word.attrib and word.attrib['reason'] == "anonymization":
                                                words.append("[ANONYMIZATION]")

                                        elif word.tag in ["trunc","mw","corr"]:

                                            for truncword in word:
                                                if truncword.tag == "mw":
                                                    for otherword in truncword:
                                                        words.append(otherword.text.strip())

                                                elif truncword.tag == "w":
                                                    words.append(truncword.text.strip())

                                                else:
                                                    continue

                            clean_utterances.append((author, uttnum, words))

                        conversations[fn] = clean_utterances

        for fn in conversations.keys():

            dialogue = {
                'id': fn[:-4],
                'utterances': []
            }

            if len(conversations[fn]) != 0:
                for author, uttnum, utt in conversations[fn]:
                    #if not bnc_id in conv_dic:
                    #    conv_dic[bnc_id] = []
                    dialogue['utterances'].append(
                        {
                            'author': author,
                            'text': " ".join(utt).strip()
                        }
                    )
            self.corpora['bnc']['dialogues'].append(dialogue)

        print("[Done]", flush=True)

        return True
