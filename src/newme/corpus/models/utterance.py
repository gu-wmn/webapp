from dataclasses import dataclass

@dataclass
class Utterance:
    author: str
    text: str
    author_plain: str = None
    reply_to: str = None
    utt_order_num: str = None
