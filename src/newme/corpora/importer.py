from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from ..extensions import db
from ..models.corpus_data import Corpus, Dialogue, Utterance

LoggerFn = Callable[[str], None]


class DialogueImporter:
    """Load dialogues from a corpora JSON file into the database.

    The file is shaped like the extracted_corpora.json the CorpusInstaller
    writes: an object keyed by corpus codename, each holding the corpus
    metadata and a list of dialogues.  Unlike the installer this downloads
    nothing - the dialogues are taken as they stand in the file, which is how
    material that was prepared elsewhere, such as the Swedish forums export,
    gets in.

    Only the corpora named in the file are touched, so importing one set leaves
    the others in the database alone.
    """

    def __init__(self, dialogues_path: str, logger: LoggerFn | None = None) -> None:
        self._path = Path(dialogues_path).expanduser().resolve()
        self._log = logger or print

    def import_dialogues(self) -> dict[str, Any]:
        if not self._path.is_file():
            raise FileNotFoundError(f"Dialogue file not found: {self._path}")

        with self._path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)

        if not isinstance(payload, Mapping):
            raise ValueError(
                "Dialogue file must contain a JSON object keyed by corpus codename."
            )

        dialogue_counts: dict[str, int] = {}
        utterance_total = 0

        for codename, corpus_data in payload.items():
            if not isinstance(corpus_data, Mapping):
                raise ValueError(f"Invalid payload for corpus '{codename}'")

            corpus = self._store_corpus_row(str(codename), corpus_data)
            dialogues = corpus_data.get("dialogues") or []
            if not isinstance(dialogues, list):
                raise ValueError(f"Corpus '{codename}' has invalid dialogues")

            stored, utterances = self._store_dialogues(corpus, str(codename), dialogues)
            dialogue_counts[str(codename)] = stored
            utterance_total += utterances
            self._log(f"Stored {stored} dialogues ({utterances} utterances) for {codename}.")

        db.session.commit()
        return {
            "dialogues_path": str(self._path),
            "dialogue_counts": dialogue_counts,
            "utterance_count": utterance_total,
        }

    def _store_corpus_row(self, codename: str, corpus_data: Mapping[str, Any]) -> Corpus:
        corpus = Corpus.query.filter_by(codename=codename).one_or_none()
        if corpus is None:
            corpus = Corpus(codename=codename, fullname=str(corpus_data.get("fullname") or codename))
            db.session.add(corpus)

        corpus.fullname = str(corpus_data.get("fullname") or codename)
        corpus.license_url = corpus_data.get("license_url")
        corpus.url = corpus_data.get("url")
        corpus.download_url = corpus_data.get("download_url")
        corpus.md5sum = corpus_data.get("md5sum")
        db.session.flush()
        return corpus

    def _store_dialogues(
        self, corpus: Corpus, codename: str, dialogues: list[Any]
    ) -> tuple[int, int]:
        self._delete_existing(corpus.id)

        stored = 0
        utterance_total = 0
        for dialogue_data in dialogues:
            if not isinstance(dialogue_data, Mapping):
                self._log(f"Skipping malformed dialogue in corpus {codename}: {dialogue_data!r}")
                continue

            external_id = dialogue_data.get("id")
            if external_id is None:
                self._log(f"Skipping dialogue without id in corpus {codename}.")
                continue

            utterances = dialogue_data.get("utterances") or []
            if not isinstance(utterances, list):
                self._log(f"Skipping dialogue {external_id} in {codename}: invalid utterances.")
                continue

            dialogue = Dialogue(
                corpus_id=corpus.id,
                external_id=str(external_id),
                url=dialogue_data.get("url"),
            )
            db.session.add(dialogue)
            db.session.flush()

            rows = []
            for position, utterance in enumerate(utterances):
                if not isinstance(utterance, Mapping):
                    self._log(
                        f"Skipping utterance {position} of dialogue {external_id} in {codename}."
                    )
                    continue
                utterance_external_id = utterance.get("id")
                reply_to = utterance.get("reply_to")
                rows.append(
                    Utterance(
                        dialogue_id=dialogue.id,
                        # The position is what a label's start_index counts
                        # against, so the file's order is kept as it is.
                        position=position,
                        external_id=str(utterance_external_id)
                        if utterance_external_id is not None
                        else None,
                        author=str(utterance.get("author") or ""),
                        text=str(utterance.get("text") or ""),
                        reply_to_external_id=str(reply_to) if reply_to is not None else None,
                    )
                )

            if rows:
                db.session.add_all(rows)
            stored += 1
            utterance_total += len(rows)

        return stored, utterance_total

    def _delete_existing(self, corpus_id: int) -> None:
        dialogue_ids = select(Dialogue.id).where(Dialogue.corpus_id == corpus_id)
        db.session.execute(delete(Utterance).where(Utterance.dialogue_id.in_(dialogue_ids)))
        db.session.execute(delete(Dialogue).where(Dialogue.corpus_id == corpus_id))
        db.session.flush()
