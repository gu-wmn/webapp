from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = REPO_ROOT / "src/newme/corpora/installer.py"


class FakeQuery:
    def __init__(self, result: object | None = None) -> None:
        self._result = result

    def filter_by(self, **_: object) -> "FakeQuery":
        return self

    def one_or_none(self) -> object | None:
        return self._result


class FakeCorpus:
    query = FakeQuery()

    def __init__(self, codename: str, fullname: str) -> None:
        self.id: int | None = None
        self.codename = codename
        self.fullname = fullname
        self.license_url: str | None = None
        self.url: str | None = None
        self.download_url: str | None = None
        self.md5sum: str | None = None


class FakeDialogue:
    def __init__(self, corpus_id: int, external_id: str) -> None:
        self.id: int | None = None
        self.corpus_id = corpus_id
        self.external_id = external_id


class FakeUtterance:
    def __init__(
        self,
        *,
        dialogue_id: int,
        position: int,
        external_id: str | None,
        author: str,
        text: str,
        reply_to_external_id: str | None,
    ) -> None:
        self.dialogue_id = dialogue_id
        self.position = position
        self.external_id = external_id
        self.author = author
        self.text = text
        self.reply_to_external_id = reply_to_external_id


class FakeNestedTransaction:
    def __init__(self, session: "FakeSession") -> None:
        self._session = session

    def __enter__(self) -> "FakeNestedTransaction":
        self._corpus_len = len(self._session.corpora)
        self._dialogue_len = len(self._session.dialogues)
        self._utterance_len = len(self._session.utterances)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            del self._session.corpora[self._corpus_len :]
            del self._session.dialogues[self._dialogue_len :]
            del self._session.utterances[self._utterance_len :]
        return False


class FakeSession:
    def __init__(self, *, fail_dialogue_id: str | None = None) -> None:
        self.fail_dialogue_id = fail_dialogue_id
        self.corpora: list[FakeCorpus] = []
        self.dialogues: list[FakeDialogue] = []
        self.utterances: list[FakeUtterance] = []
        self.committed = False
        self.rollback_count = 0
        self._next_id = 1

    def add(self, obj: object) -> None:
        if isinstance(obj, FakeCorpus):
            self.corpora.append(obj)
        elif isinstance(obj, FakeDialogue):
            self.dialogues.append(obj)

    def add_all(self, objs: list[FakeUtterance]) -> None:
        self.utterances.extend(objs)

    def flush(self) -> None:
        for corpus in self.corpora:
            if corpus.id is None:
                corpus.id = self._allocate_id()
        for dialogue in self.dialogues:
            if dialogue.id is None:
                if dialogue.external_id == self.fail_dialogue_id:
                    raise ValueError(f"cannot store dialogue {dialogue.external_id}")
                dialogue.id = self._allocate_id()

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rollback_count += 1

    def begin_nested(self) -> FakeNestedTransaction:
        return FakeNestedTransaction(self)

    def execute(self, *_: object, **__: object) -> None:
        return None

    def _allocate_id(self) -> int:
        next_id = self._next_id
        self._next_id += 1
        return next_id


def load_installer_module():
    module_name = "newme.corpora.installer"
    for name in [
        module_name,
        "newme",
        "newme.corpora",
        "newme.extensions",
        "newme.models",
        "newme.models.corpus_data",
        "sqlalchemy",
        "requests",
    ]:
        sys.modules.pop(name, None)

    newme_pkg = types.ModuleType("newme")
    newme_pkg.__path__ = []
    corpora_pkg = types.ModuleType("newme.corpora")
    corpora_pkg.__path__ = []
    extensions_module = types.ModuleType("newme.extensions")
    models_pkg = types.ModuleType("newme.models")
    models_pkg.__path__ = []
    corpus_data_module = types.ModuleType("newme.models.corpus_data")
    sqlalchemy_module = types.ModuleType("sqlalchemy")
    requests_module = types.ModuleType("requests")

    extensions_module.db = types.SimpleNamespace(session=FakeSession())
    corpus_data_module.Corpus = FakeCorpus
    corpus_data_module.Dialogue = FakeDialogue
    corpus_data_module.Utterance = FakeUtterance
    sqlalchemy_module.delete = lambda *args, **kwargs: None
    sqlalchemy_module.select = lambda *args, **kwargs: None
    requests_module.get = None

    sys.modules["newme"] = newme_pkg
    sys.modules["newme.corpora"] = corpora_pkg
    sys.modules["newme.extensions"] = extensions_module
    sys.modules["newme.models"] = models_pkg
    sys.modules["newme.models.corpus_data"] = corpus_data_module
    sys.modules["sqlalchemy"] = sqlalchemy_module
    sys.modules["requests"] = requests_module

    spec = importlib.util.spec_from_file_location(module_name, INSTALLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class CorpusInstallerTests(unittest.TestCase):
    def test_install_continues_after_one_corpus_fails(self) -> None:
        module = load_installer_module()
        session = FakeSession()
        module.db.session = session

        with tempfile.TemporaryDirectory() as temp_dir:
            logs: list[str] = []
            installer = module.CorpusInstaller(
                {
                    "CORPORA_PATH": str(Path(temp_dir) / "corpora"),
                    "CORPORA_ENABLED": ["switchboard-corpus", "bnc"],
                },
                logger=logs.append,
            )

            def explode() -> list[dict[str, object]]:
                raise ValueError("switchboard exploded")

            installer._extract_switchboard = explode
            installer._extract_bnc = lambda: [{"id": "bnc-1", "utterances": []}]
            installer._store_corpus = lambda codename, dialogues: len(dialogues)

            result = installer.install()

        self.assertEqual(result["dialogue_counts"], {"bnc": 1})
        self.assertEqual(
            result["failed_corpora"],
            {"switchboard-corpus": "ValueError: switchboard exploded"},
        )
        self.assertEqual(session.rollback_count, 1)
        self.assertTrue(any("continuing with next" in line for line in logs))

    def test_extract_bnc_skips_malformed_dialogue(self) -> None:
        module = load_installer_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            logs: list[str] = []
            installer = module.CorpusInstaller(
                {"CORPORA_PATH": str(Path(temp_dir) / "corpora")},
                logger=logs.append,
            )

            texts_dir = Path(temp_dir) / "Texts"
            texts_dir.mkdir()
            (texts_dir / "good.xml").write_text(
                "<root><stext><u who='A'><s><w>Hello</w></s></u></stext></root>",
                encoding="utf-8",
            )
            (texts_dir / "bad.xml").write_text("<root><stext>", encoding="utf-8")

            installer._ensure_downloaded_and_unpacked = lambda codename, required: texts_dir

            dialogues = installer._extract_bnc()

        self.assertEqual(len(dialogues), 1)
        self.assertEqual(dialogues[0]["id"], "good")
        self.assertEqual(dialogues[0]["utterances"], [{"author": "A", "text": "Hello"}])
        self.assertTrue(any("Skipping BNC dialogue bad" in line for line in logs))

    def test_extract_switchboard_skips_entire_dialogue_when_one_row_is_bad(self) -> None:
        module = load_installer_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            logs: list[str] = []
            installer = module.CorpusInstaller(
                {"CORPORA_PATH": str(Path(temp_dir) / "corpora")},
                logger=logs.append,
            )

            utterance_file = Path(temp_dir) / "utterances.jsonl"
            utterance_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "conversation_id": "conv-1",
                                "id": "u1",
                                "speaker": "A",
                                "text": "Hello",
                            }
                        ),
                        json.dumps(
                            {
                                "conversation_id": "conv-1",
                                "id": "u2",
                                "speaker": "B",
                                "text": ["not", "text"],
                            }
                        ),
                        json.dumps(
                            {
                                "conversation_id": "conv-2",
                                "id": "u3",
                                "speaker": "C",
                                "text": "Still good",
                            }
                        ),
                        json.dumps(
                            {
                                "conversation_id": "conv-1",
                                "id": "u4",
                                "speaker": "D",
                                "text": "Should stay skipped",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            installer._ensure_downloaded_and_unpacked = lambda codename, required: utterance_file

            dialogues = installer._extract_switchboard()

        self.assertEqual(dialogues, [{"id": "conv-2", "utterances": [{"id": "u3", "text": "Still good", "author": "C"}]}])
        self.assertTrue(any("Skipping Switchboard dialogue conv-1" in line for line in logs))

    def test_store_corpus_skips_dialogue_that_fails_to_store(self) -> None:
        module = load_installer_module()
        session = FakeSession(fail_dialogue_id="bad")
        module.db.session = session
        FakeCorpus.query = FakeQuery()

        with tempfile.TemporaryDirectory() as temp_dir:
            logs: list[str] = []
            installer = module.CorpusInstaller(
                {"CORPORA_PATH": str(Path(temp_dir) / "corpora")},
                logger=logs.append,
            )
            installer._delete_existing_corpus_data = lambda corpus_id: None

            count = installer._store_corpus(
                "switchboard-corpus",
                [
                    {
                        "id": "good",
                        "utterances": [{"id": "u1", "author": "A", "text": "Hello"}],
                    },
                    {
                        "id": "bad",
                        "utterances": [{"id": "u2", "author": "B", "text": "Goodbye"}],
                    },
                ],
            )

        self.assertEqual(count, 1)
        self.assertTrue(session.committed)
        self.assertEqual([dialogue.external_id for dialogue in session.dialogues], ["good"])
        self.assertTrue(any("Skipping dialogue bad in corpus switchboard-corpus" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
