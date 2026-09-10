from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import delete, select

from ..extensions import db
from ..models.annotation_data import AnnotationLabel, AnnotationSequence

LoggerFn = Callable[[str], None]


class AnnotationImporter:
    def __init__(self, annotations_path: str, logger: LoggerFn | None = None) -> None:
        self._path = Path(annotations_path).expanduser().resolve()
        self._log = logger or print

    def import_annotations(self) -> dict[str, Any]:
        if not self._path.is_file():
            raise FileNotFoundError(f"Annotations file not found: {self._path}")

        with self._path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)

        if not isinstance(payload, list):
            raise ValueError("Annotations file must contain a JSON list.")

        # Only the corpora this file covers are replaced, so importing one set
        # of annotations leaves any other set in the database alone.
        codenames = {
            str(item.get("corpus_codename"))
            for item in payload
            if isinstance(item, dict) and item.get("corpus_codename")
        }
        self._delete_existing(codenames)

        label_count = 0
        sequence_count = 0
        for item in payload:
            wmn_id = item.get("wmn_id")
            corpus_codename = item.get("corpus_codename")
            dialogue_id = item.get("dialogue_id")
            if not wmn_id or not corpus_codename or not dialogue_id:
                continue

            sequence = AnnotationSequence(
                wmn_id=str(wmn_id),
                corpus_codename=str(corpus_codename),
                dialogue_external_id=str(dialogue_id),
                context=str(item.get("context") or ""),
                wmn_type=str(item.get("wmn") or ""),
                wmn_meaning=str(item.get("wmn_meaning") or ""),
                annotator=str(item.get("annotator") or ""),
                comment=item.get("comment"),
                prediction=item.get("prediction"),
            )
            db.session.add(sequence)
            db.session.flush()
            sequence_count += 1

            labels = item.get("labels") or []
            for position, label in enumerate(labels):
                excerpt = str(label.get("excerpt") or "")
                excerpt_hash = hashlib.md5(excerpt.casefold().strip().encode("utf-8")).hexdigest()
                db.session.add(
                    AnnotationLabel(
                        annotation_id=sequence.id,
                        position=position,
                        name=str(label.get("name") or ""),
                        start_index=self._parse_int(label.get("start_index")),
                        end_index=self._parse_int(label.get("end_index")),
                        start_offset=self._parse_int(label.get("start_offset")),
                        end_offset=self._parse_int(label.get("end_offset")),
                        excerpt=excerpt,
                        excerpt_hash=excerpt_hash,
                        type=label.get("type"),
                    )
                )
                label_count += 1

        db.session.commit()
        self._log(f"Imported {sequence_count} annotation sequences and {label_count} labels.")
        return {
            "annotations_path": str(self._path),
            "sequence_count": sequence_count,
            "label_count": label_count,
            "corpora": sorted(codenames),
        }

    def _delete_existing(self, codenames: set[str]) -> None:
        if not codenames:
            return

        sequence_ids = select(AnnotationSequence.id).where(
            AnnotationSequence.corpus_codename.in_(codenames)
        )
        db.session.execute(
            delete(AnnotationLabel).where(AnnotationLabel.annotation_id.in_(sequence_ids))
        )
        db.session.execute(
            delete(AnnotationSequence).where(AnnotationSequence.corpus_codename.in_(codenames))
        )
        db.session.flush()

    @staticmethod
    def _parse_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
