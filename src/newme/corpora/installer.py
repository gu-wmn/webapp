from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET

import requests
from sqlalchemy import delete, select

from ..extensions import db
from ..models.corpus_data import Corpus, Dialogue, Utterance

LoggerFn = Callable[[str], None]

DEFAULT_CORPORA: dict[str, dict[str, str]] = {
    "bnc": {
        "fullname": "British National Corpus",
        "license_url": "http://www.natcorp.ox.ac.uk/docs/licence.html",
        "url": "http://www.natcorp.ox.ac.uk/",
        "download_url": "https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/2554/2554.zip?sequence=4&isAllowed=y",
        "md5sum": "5fdea535beb437d232af2a0217e92bf1",
    },
    "winning-args-corpus": {
        "fullname": "Winning Arguments (ChangeMyView) Corpus",
        "license_url": "",
        "url": "https://convokit.cornell.edu/documentation/winning.html",
        "download_url": "https://zissou.infosci.cornell.edu/convokit/datasets/winning-args-corpus/winning-args-corpus.zip",
        "md5sum": "d17056c46f4805a83cd192e56900863f",
    },
    "switchboard-corpus": {
        "fullname": "Switchboard Dialog Act Corpus",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "url": "http://compprag.christopherpotts.net/swda.html",
        "download_url": "https://zissou.infosci.cornell.edu/convokit/datasets/switchboard-corpus/switchboard-corpus.zip",
        "md5sum": "714650ec360b91ba37452e4f7448c44a",
    },
}


class CorpusInstaller:
    def __init__(self, config: Mapping[str, Any], logger: LoggerFn | None = None) -> None:
        self._config = config
        self._log = logger or print
        self._timeout = int(config.get("CORPORA_TIMEOUT_SECONDS", 120))
        self._force_redownload = bool(config.get("CORPORA_FORCE_REDOWNLOAD", False))

        self._corpora_dir = Path(config["CORPORA_PATH"])
        self._corpora_dir.mkdir(parents=True, exist_ok=True)
        self._source_dir = Path(config.get("CORPORA_SOURCE_DIR", self._corpora_dir / "original_corpora"))
        self._source_dir.mkdir(parents=True, exist_ok=True)
        self._extracted_corpora_path = self._corpora_dir / "extracted_corpora.json"

        self._corpora = self._load_corpora_definitions(config.get("CORPORA_CONFIG_PATH"))
        self._enabled_corpora = self._resolve_enabled_corpora(config.get("CORPORA_ENABLED"))
        self._dialogue_ids = self._load_dialogue_ids()

    def install(self) -> dict[str, Any]:
        if self._dialogue_ids:
            self._log(f"Applying dialogue filter with {len(self._dialogue_ids)} ids.")
        else:
            self._log("No dialogue filter configured; full corpora extraction will run.")

        dialogue_counts: dict[str, int] = {}
        failed_corpora: dict[str, str] = {}
        extracted_corpora = self._build_extracted_corpora_skeleton()
        for codename in self._enabled_corpora:
            self._log(f"Processing corpus: {codename}")
            try:
                if codename == "switchboard-corpus":
                    dialogues = self._extract_switchboard()
                elif codename == "winning-args-corpus":
                    dialogues = self._extract_winning_args()
                elif codename == "bnc":
                    dialogues = self._extract_bnc()
                else:
                    raise ValueError(f"Unsupported corpus extractor for '{codename}'")
                count = self._store_corpus(codename, dialogues)
                dialogue_counts[codename] = count
                extracted_corpora[codename]["dialogues"] = dialogues
                self._log(f"Stored {count} dialogues for {codename}.")
            except Exception as exc:
                db.session.rollback()
                failed_corpora[codename] = str(exc)
                self._log(f"Failed corpus '{codename}', continuing with next: {exc}")

        self._write_extracted_corpora_json(extracted_corpora)

        summary = {
            "enabled_corpora": self._enabled_corpora,
            "dialogue_counts": dialogue_counts,
            "failed_corpora": failed_corpora,
            "extracted_corpora_path": str(self._extracted_corpora_path),
        }
        self._log(f"Corpora extraction summary: {summary['dialogue_counts']}")
        if failed_corpora:
            self._log(f"Corpora failed: {failed_corpora}")
        return summary

    def _build_extracted_corpora_skeleton(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for codename, corpus_def in self._corpora.items():
            payload[codename] = {
                "fullname": str(corpus_def.get("fullname") or codename),
                "license_url": str(corpus_def.get("license_url") or ""),
                "url": str(corpus_def.get("url") or ""),
                "download_url": str(corpus_def.get("download_url") or ""),
                "md5sum": str(corpus_def.get("md5sum") or ""),
                "dialogues": [],
            }
        return payload

    def _write_extracted_corpora_json(self, payload: dict[str, dict[str, Any]]) -> None:
        try:
            self._extracted_corpora_path.write_text(json.dumps(payload), encoding="utf-8")
            self._log(f"Saved extracted corpora JSON: {self._extracted_corpora_path}")
        except OSError as exc:
            self._log(f"Failed to save extracted corpora JSON: {exc}")

    def _store_corpus(self, codename: str, dialogues: list[dict[str, Any]]) -> int:
        corpus_def = self._corpora[codename]
        corpus = Corpus.query.filter_by(codename=codename).one_or_none()
        if corpus is None:
            corpus = Corpus(codename=codename, fullname=corpus_def["fullname"])
            db.session.add(corpus)

        corpus.fullname = corpus_def["fullname"]
        corpus.license_url = corpus_def.get("license_url")
        corpus.url = corpus_def.get("url")
        corpus.download_url = corpus_def.get("download_url")
        corpus.md5sum = corpus_def.get("md5sum")
        db.session.flush()

        self._delete_existing_corpus_data(corpus.id)

        for dialogue_data in dialogues:
            dialogue_external_id = dialogue_data.get("id")
            if dialogue_external_id is None:
                continue
            dialogue = Dialogue(
                corpus_id=corpus.id,
                external_id=str(dialogue_external_id),
            )
            db.session.add(dialogue)
            db.session.flush()

            utterance_rows = []
            for position, utterance in enumerate(dialogue_data.get("utterances", [])):
                external_id = utterance.get("id")
                reply_to = utterance.get("reply_to")
                utterance_rows.append(
                    Utterance(
                        dialogue_id=dialogue.id,
                        position=position,
                        external_id=str(external_id) if external_id is not None else None,
                        author=str(utterance.get("author") or ""),
                        text=str(utterance.get("text") or ""),
                        reply_to_external_id=str(reply_to) if reply_to is not None else None,
                    )
                )
            if utterance_rows:
                db.session.add_all(utterance_rows)

        db.session.commit()
        return len(dialogues)

    def _delete_existing_corpus_data(self, corpus_id: int) -> None:
        dialogue_ids = select(Dialogue.id).where(Dialogue.corpus_id == corpus_id)
        db.session.execute(delete(Utterance).where(Utterance.dialogue_id.in_(dialogue_ids)))
        db.session.execute(delete(Dialogue).where(Dialogue.corpus_id == corpus_id))
        db.session.flush()

    def _load_corpora_definitions(self, config_path: str | None) -> dict[str, dict[str, str]]:
        corpora: dict[str, dict[str, str]] = {k: dict(v) for k, v in DEFAULT_CORPORA.items()}
        if not config_path:
            return corpora

        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CORPORA_CONFIG_PATH does not exist: {path}")

        with path.open(encoding="utf-8") as config_file:
            overrides = json.load(config_file)

        if not isinstance(overrides, dict):
            raise ValueError("CORPORA_CONFIG_PATH must contain a JSON object keyed by corpus codename.")

        for codename, values in overrides.items():
            if codename not in corpora:
                raise ValueError(f"Unknown corpus codename in CORPORA_CONFIG_PATH: {codename}")
            if not isinstance(values, dict):
                raise ValueError(f"Invalid override payload for '{codename}'")
            corpora[codename].update(values)

        return corpora

    def _resolve_enabled_corpora(self, configured: list[str] | None) -> list[str]:
        enabled = configured or list(DEFAULT_CORPORA.keys())
        unknown = [name for name in enabled if name not in self._corpora]
        if unknown:
            raise ValueError(f"Unknown corpus names in CORPORA_ENABLED: {', '.join(unknown)}")
        return enabled

    def _load_dialogue_ids(self) -> set[str]:
        configured_ids = self._config.get("CORPORA_DIALOGUE_IDS") or []
        dialogue_ids = {str(value) for value in configured_ids if value}

        annotations_path = self._config.get("CORPORA_ANNOTATIONS_PATH")
        if annotations_path:
            path = Path(annotations_path).expanduser().resolve()
            if path.is_file():
                with path.open(encoding="utf-8") as annotation_file:
                    annotations = json.load(annotation_file)
                for item in annotations:
                    dialogue_id = item.get("dialogue_id")
                    if dialogue_id:
                        dialogue_ids.add(str(dialogue_id))
            else:
                self._log(f"Configured CORPORA_ANNOTATIONS_PATH does not exist: {path}")

        return dialogue_ids

    def _ensure_downloaded_and_unpacked(self, codename: str, required_relative_path: str) -> Path:
        corpus = self._corpora[codename]
        corpus_dir = self._source_dir / codename
        corpus_dir.mkdir(parents=True, exist_ok=True)

        required_path = corpus_dir / required_relative_path
        if required_path.exists() and not self._force_redownload:
            return required_path

        archive_path = corpus_dir / f"{codename}.zip"
        if self._force_redownload or not archive_path.exists():
            self._download_archive(corpus["download_url"], archive_path)
            self._verify_md5(archive_path, corpus.get("md5sum"))

        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(corpus_dir)

        if not required_path.exists():
            raise FileNotFoundError(
                f"Expected extracted file not found for '{codename}': {required_path}"
            )
        return required_path

    def _download_archive(self, url: str, archive_path: Path) -> None:
        self._log(f"Downloading {url}")
        with requests.get(url, stream=True, timeout=self._timeout) as response:
            response.raise_for_status()
            with archive_path.open("wb") as archive_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive_file.write(chunk)

    def _verify_md5(self, archive_path: Path, expected_md5: str | None) -> None:
        if not expected_md5:
            return
        digest = hashlib.md5()
        with archive_path.open("rb") as archive_file:
            while True:
                chunk = archive_file.read(8192)
                if not chunk:
                    break
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_md5:
            raise ValueError(
                f"Checksum mismatch for {archive_path}. expected={expected_md5}, actual={actual}"
            )

    def _is_in_filter(self, dialogue_id: str) -> bool:
        if not self._dialogue_ids:
            return True
        return str(dialogue_id) in self._dialogue_ids

    def _extract_switchboard(self) -> list[dict[str, Any]]:
        utterance_file = self._ensure_downloaded_and_unpacked(
            "switchboard-corpus", "switchboard-corpus/utterances.jsonl"
        )

        conversations: dict[str, list[dict[str, Any]]] = {}
        with utterance_file.open(encoding="utf-8") as file_obj:
            for line in file_obj:
                utterance = json.loads(line)
                conversation_id = utterance.get("conversation_id")
                if conversation_id is not None:
                    conversation_id = str(conversation_id)
                if not conversation_id or not self._is_in_filter(conversation_id):
                    continue
                conversations.setdefault(conversation_id, []).append(
                    {
                        "id": utterance.get("id"),
                        "text": utterance.get("text", ""),
                        "author": utterance.get("speaker", ""),
                    }
                )

        return [
            {"id": conversation_id, "utterances": utterances}
            for conversation_id, utterances in conversations.items()
        ]

    def _extract_winning_args(self) -> list[dict[str, Any]]:
        utterance_file = self._ensure_downloaded_and_unpacked(
            "winning-args-corpus", "winning-args-corpus/utterances.jsonl"
        )
        conversations_file = self._ensure_downloaded_and_unpacked(
            "winning-args-corpus", "winning-args-corpus/conversations.json"
        )

        with conversations_file.open(encoding="utf-8") as conv_file:
            conversation_metadata = json.load(conv_file)

        conversations: dict[str, list[dict[str, Any]]] = {}
        with utterance_file.open(encoding="utf-8") as file_obj:
            for line in file_obj:
                utterance = json.loads(line)
                root_id = utterance.get("root")
                if root_id is not None:
                    root_id = str(root_id)
                if not root_id or not self._is_in_filter(root_id):
                    continue
                conversations.setdefault(root_id, []).append(
                    {
                        "id": utterance.get("id"),
                        "author": utterance.get("user", ""),
                        "reply_to": utterance.get("reply-to"),
                        "text": self._mark_citations(utterance.get("text")),
                    }
                )

        dialogue_list: list[dict[str, Any]] = []
        for conversation_id, utterances in conversations.items():
            ordered = self._order_winning_args_utterances(utterances)
            title = conversation_metadata.get(conversation_id, {}).get("op-title", "")
            ordered = [{"author": "TITLE", "text": title, "id": conversation_id}] + ordered
            dialogue_list.append({"id": conversation_id, "utterances": ordered})

        return dialogue_list

    def _mark_citations(self, text: str | None) -> str:
        if not text:
            return ""
        pattern = re.compile(r"&gt;.*?\n\n", flags=re.DOTALL)
        return pattern.sub(lambda match: f"[STA-CITE]{match.group(0)}[END-CITE]", text)

    def _order_winning_args_utterances(self, utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reply_parent = {utt.get("id"): utt.get("reply_to") for utt in utterances}
        ancestor_map: dict[Any, set[Any]] = {}

        for utt_id in reply_parent:
            ancestors: set[Any] = set()
            current = reply_parent[utt_id]
            while current is not None and current not in ancestors:
                ancestors.add(current)
                current = reply_parent.get(current)
            ancestor_map[utt_id] = ancestors

        ordered: list[dict[str, Any]] = []
        for utterance in utterances:
            reply_to = utterance.get("reply_to")
            if reply_to is None:
                ordered.append(utterance)
                continue

            insert_after = None
            for idx, maybe_parent in enumerate(ordered):
                maybe_parent_id = maybe_parent.get("id")
                if maybe_parent_id in ancestor_map.get(utterance.get("id"), set()):
                    insert_after = idx

            if insert_after is None:
                ordered.append(utterance)
            else:
                ordered = ordered[: insert_after + 1] + [utterance] + ordered[insert_after + 1 :]

        return ordered

    def _extract_bnc(self) -> list[dict[str, Any]]:
        bnc_text_dir = self._ensure_downloaded_and_unpacked("bnc", "download/Texts")
        if not bnc_text_dir.is_dir():
            raise NotADirectoryError(f"BNC text directory is missing: {bnc_text_dir}")

        dialogues: list[dict[str, Any]] = []
        for xml_path in bnc_text_dir.rglob("*.xml"):
            dialogue_id = str(xml_path.stem)
            if not self._is_in_filter(dialogue_id):
                continue
            utterances = self._parse_bnc_file(xml_path)
            dialogues.append({"id": dialogue_id, "utterances": utterances})

        return dialogues

    def _parse_bnc_file(self, xml_path: Path) -> list[dict[str, str]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        stext_nodes = [node for node in root.iter() if node.tag == "stext"]
        if not stext_nodes:
            return []

        clean_utterances: list[dict[str, str]] = []
        stext = stext_nodes[0]
        for utterance in stext:
            author = utterance.attrib.get("who")
            if not author:
                continue
            words = self._extract_bnc_words(utterance)
            clean_utterances.append({"author": author, "text": " ".join(words).strip()})

        return clean_utterances

    def _extract_bnc_words(self, utterance: Any) -> list[str]:
        words: list[str] = []
        for sentence in utterance:
            if sentence.tag != "s":
                if sentence.tag == "unclear":
                    words.append("[UNCLEAR]")
                continue

            for word in sentence:
                if word.tag in {"w", "c"} and word.text:
                    words.append(word.text.strip())
                elif word.tag in {"align", "pause", "shift", "event", "vocal"}:
                    continue
                elif word.tag == "unclear":
                    words.append("[UNCLEAR]")
                elif word.tag == "gap":
                    if word.attrib.get("reason") == "anonymization":
                        words.append("[ANONYMIZATION]")
                elif word.tag in {"trunc", "mw", "corr"}:
                    words.extend(self._extract_bnc_nested_words(word))
        return words

    def _extract_bnc_nested_words(self, parent: Any) -> list[str]:
        nested_words: list[str] = []
        for node in parent:
            if node.tag == "mw":
                for child in node:
                    if child.text:
                        nested_words.append(child.text.strip())
            elif node.tag == "w" and node.text:
                nested_words.append(node.text.strip())
        return nested_words
