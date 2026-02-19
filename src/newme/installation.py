from __future__ import annotations

from typing import Any, Callable

from flask import current_app

from .extensions import db
from .models import (
    AnnotationLabel,
    AnnotationSequence,
    AppState,
    Corpus,
    Dialogue,
    Utterance,
)

LoggerFn = Callable[[str], None]


def perform_installation(
    *,
    install_corpora: bool | None = None,
    install_annotations: bool | None = None,
    logger: LoggerFn | None = None,
) -> dict[str, Any]:
    # Ensure SQLAlchemy metadata includes corpus tables before create_all().
    _ = (Corpus, Dialogue, Utterance, AnnotationSequence, AnnotationLabel)

    if install_corpora is None:
        install_corpora = bool(current_app.config.get("INSTALL_CORPORA_ON_SETUP", True))
    if install_annotations is None:
        install_annotations = bool(current_app.config.get("INSTALL_ANNOTATIONS_ON_SETUP", True))

    db.create_all()

    result: dict[str, Any] = {"corpora": None, "annotations": None}
    if install_corpora:
        try:
            from .corpora import CorpusInstaller
        except ModuleNotFoundError as exc:
            if exc.name == "requests":
                raise RuntimeError(
                    "Corpus installation requires 'requests'. Reinstall the package: pip install ."
                ) from exc
            raise

        emit = logger or (lambda message: current_app.logger.info(message))
        installer = CorpusInstaller(current_app.config, logger=emit)
        result["corpora"] = installer.install()

    if install_annotations:
        from .annotations import AnnotationImporter

        annotations_path = current_app.config.get("ANNOTATIONS_PATH")
        if not annotations_path:
            raise RuntimeError(
                "No annotations path configured. Set NEWME_ANNOTATIONS_PATH or provide wmn_annotations.json."
            )
        emit = logger or (lambda message: current_app.logger.info(message))
        importer = AnnotationImporter(annotations_path=annotations_path, logger=emit)
        result["annotations"] = importer.import_annotations()

    AppState.mark_initialized()
    db.session.commit()

    return result
