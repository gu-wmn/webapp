from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import OrderedDict

from flask import Blueprint, Response, abort, render_template, request, url_for
from markupsafe import Markup, escape
from sqlalchemy.orm import selectinload

from .extensions import db
from .models import AnnotationLabel, AnnotationSequence, Corpus, Dialogue, Utterance

bp = Blueprint("main", __name__)

SITE_TITLE = "NeWMe"

WMN_TYPE_OPTIONS = [
    ("NON", "WMN: non-understanding"),
    ("DIN", "WMN: disagreement"),
    ("OTHER", "WMN: other"),
    ("SIMN", "SIMN"),
    ("OTHER_CLAR_REQ", "Other kinds of clarification requests"),
    ("NO_TRIGGER", "Without trigger"),
    ("NON_PURSUED", "Non-pursued"),
    ("IMPOSSIBLE", "Impossible to annotate"),
    ("REFERENCE_NE", "reference/NE"),
]
DEFAULT_WMN_NAMES = {"NON", "DIN", "OTHER"}
WMN_TYPE_BY_NAME = {name: value for name, value in WMN_TYPE_OPTIONS}
WMN_TYPE_NAME_BY_VALUE = {value: name for name, value in WMN_TYPE_OPTIONS}
VALID_WMN_TYPES = set(WMN_TYPE_BY_NAME.values())

LABEL_OPTIONS = [
    ("TRIGGER", "Trigger"),
    ("INDICATOR", "Indicator"),
    ("NEGOTIATION", "Negotiation"),
]
LABEL_VALUE_BY_NAME = {name: value for name, value in LABEL_OPTIONS}
VALID_LABELS = set(LABEL_VALUE_BY_NAME.values())

CONTEXT_OPTIONS = [
    ("SPOKEN", "Spoken interaction"),
    ("ONLINE", "Online interaction"),
]
CONTEXT_VALUE_BY_NAME = {name: value for name, value in CONTEXT_OPTIONS}

GROUP_BY_OPTIONS = [
    ("DIALOGUE", "Dialogue"),
    ("SEQUENCE", "WMN Sequence"),
    ("LABEL", "Label"),
]

CSV_COLUMN_OPTIONS = [
    ("wmn_id", "wmn_id", "WMN ID"),
    ("corpus", "corpus", "Corpus"),
    ("wmn_type", "wmn_type", "WMN Type"),
    ("trigger", "trigger", "Trigger"),
    ("indicator", "indicator", "Indicator"),
    ("negotiation", "negotiation", "Negotiation"),
    ("wmn_link", "wmn_link", "Link to WMN"),
]
CSV_COLUMN_HEADERS = {key: header for key, header, _ in CSV_COLUMN_OPTIONS}
DEFAULT_CSV_COLUMNS = [
    key
    for key, _, _ in CSV_COLUMN_OPTIONS
    if key not in {"indicator", "negotiation"}
]

DEFAULT_CORPUS_OPTIONS = OrderedDict(
    [
        ("bnc", "British National Corpus"),
        ("winning-args-corpus", "Winning Arguments (ChangeMyView) Corpus"),
        ("switchboard-corpus", "Switchboard Dialog Act Corpus"),
    ]
)


@bp.app_template_filter("highlight_search")
def highlight_search(text: object, search_term: str | None) -> Markup:
    raw_text = "" if text is None else str(text)
    if not search_term:
        return Markup(escape(raw_text))

    normalized_term = search_term.strip()
    if not normalized_term:
        return Markup(escape(raw_text))

    pattern = re.compile(re.escape(normalized_term), re.IGNORECASE)
    chunks: list[str] = []
    cursor = 0

    for match in pattern.finditer(raw_text):
        if match.start() > cursor:
            chunks.append(str(escape(raw_text[cursor:match.start()])))
        chunks.append('<mark class="search-hit">')
        chunks.append(str(escape(match.group(0))))
        chunks.append("</mark>")
        cursor = match.end()

    if cursor == 0:
        return Markup(escape(raw_text))

    chunks.append(str(escape(raw_text[cursor:])))
    return Markup("".join(chunks))


@bp.get("/")
def main_page():
    corpus_options = _corpus_options()
    filters = _parse_filters(request.args, corpus_options)

    sequences = (
        AnnotationSequence.query.options(selectinload(AnnotationSequence.labels))
        .order_by(AnnotationSequence.id.asc())
        .all()
    )

    if filters["group_by"] == "DIALOGUE":
        summaries = _summaries_by_dialogue(sequences, filters, corpus_options)
    elif filters["group_by"] == "LABEL":
        summaries = _summaries_by_label(sequences, filters, corpus_options)
    else:
        summaries = _summaries_by_sequence(sequences, filters, corpus_options)

    return render_template(
        "index.html",
        site_title=SITE_TITLE,
        summaries=summaries,
        num_results=len(summaries),
        filters=filters,
        group_by_options=GROUP_BY_OPTIONS,
    )


@bp.get("/download.csv")
def download_csv():
    corpus_options = _corpus_options()
    filters = _parse_filters(request.args, corpus_options)

    sequences = (
        AnnotationSequence.query.options(selectinload(AnnotationSequence.labels))
        .order_by(AnnotationSequence.id.asc())
        .all()
    )
    rows = _csv_rows_by_sequence(sequences, filters)
    selected_columns = filters["selected_csv_columns"]

    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow([CSV_COLUMN_HEADERS[column] for column in selected_columns])
    for row in rows:
        writer.writerow([_csv_value_for_column(column, row) for column in selected_columns])

    filename = "wmn_results.csv"
    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@bp.get("/dialogue/<string:dialogue_id>")
def dialogue_page(dialogue_id: str):
    sequences = (
        AnnotationSequence.query.options(selectinload(AnnotationSequence.labels))
        .filter_by(dialogue_external_id=dialogue_id)
        .order_by(AnnotationSequence.wmn_id.asc())
        .all()
    )

    sequences = [
        sequence
        for sequence in sequences
        if sequence.wmn_type in VALID_WMN_TYPES and sequence.labels
    ]
    if not sequences:
        abort(404)

    corpus = Corpus.query.filter_by(codename=sequences[0].corpus_codename).one_or_none()

    sequence_summaries: list[dict] = []
    for sequence in sequences:
        grouped = {"Trigger": {}, "Indicator": {}, "Negotiation": {}}
        for label in _sorted_labels(sequence.labels):
            if label.name not in VALID_LABELS:
                continue
            excerpt = (label.excerpt or "").strip()
            if excerpt not in grouped[label.name]:
                grouped[label.name][excerpt] = 1
            else:
                grouped[label.name][excerpt] += 1

        sequence_summaries.append(
            {
                "wmn_id": sequence.wmn_id,
                "context": sequence.context,
                "wmn_type": sequence.wmn_type,
                "wmn_meaning": sequence.wmn_meaning,
                "triggers": grouped["Trigger"],
                "indicators": grouped["Indicator"],
                "negotiations": grouped["Negotiation"],
            }
        )

    return render_template(
        "dialogue.html",
        site_title=SITE_TITLE,
        dialogue_id=dialogue_id,
        corpus=corpus,
        sequence_summaries=sequence_summaries,
    )


@bp.get("/wmn/<string:dialogue_id>/<string:wmn_id>/")
def sequence_page(dialogue_id: str, wmn_id: str):
    sequence = (
        AnnotationSequence.query.options(selectinload(AnnotationSequence.labels))
        .filter_by(wmn_id=wmn_id, dialogue_external_id=dialogue_id)
        .one_or_none()
    )
    if sequence is None or sequence.wmn_type not in VALID_WMN_TYPES:
        abort(404)

    corpus = Corpus.query.filter_by(codename=sequence.corpus_codename).one_or_none()

    sibling_wmn_ids = [
        row.wmn_id
        for row in AnnotationSequence.query.filter_by(dialogue_external_id=dialogue_id)
        .order_by(AnnotationSequence.wmn_id.asc())
        .all()
        if row.wmn_type in VALID_WMN_TYPES
    ]

    dialogue = (
        db.session.query(Dialogue)
        .join(Corpus, Dialogue.corpus_id == Corpus.id)
        .filter(Corpus.codename == sequence.corpus_codename, Dialogue.external_id == dialogue_id)
        .one_or_none()
    )

    utterances: list[dict[str, str]] = []
    if dialogue is not None:
        utterance_rows = (
            Utterance.query.filter_by(dialogue_id=dialogue.id)
            .order_by(Utterance.position.asc())
            .all()
        )
        utterances = [{"author": row.author, "text": row.text} for row in utterance_rows]
    dialogue_missing = dialogue is None or not utterances

    labels = [
        {
            "name": label.name,
            "start_index": label.start_index,
            "end_index": label.end_index,
            "start_offset": label.start_offset,
            "end_offset": label.end_offset,
            "excerpt": label.excerpt,
        }
        for label in _sorted_labels(sequence.labels)
        if label.name in VALID_LABELS
    ]

    truncation = _label_truncation_bounds(labels, len(utterances))

    annotated_utterances, label_links = _annotate_utterances(utterances, labels)
    if dialogue_missing and labels:
        label_links = [
            {
                "name": label["name"],
                "excerpt": str(label.get("excerpt") or ""),
                "link": "",
            }
            for label in labels
        ]

    return render_template(
        "wmn.html",
        site_title=SITE_TITLE,
        sequence=sequence,
        corpus=corpus,
        sibling_wmn_ids=sibling_wmn_ids,
        utterances=annotated_utterances,
        label_links=label_links,
        truncation=truncation,
        dialogue_missing=dialogue_missing,
    )


@bp.get("/label/<string:excerpt_hash>")
def label_page(excerpt_hash: str):
    rows = (
        db.session.query(AnnotationLabel, AnnotationSequence)
        .join(AnnotationSequence, AnnotationLabel.annotation_id == AnnotationSequence.id)
        .filter(AnnotationLabel.excerpt_hash == excerpt_hash)
        .order_by(AnnotationSequence.wmn_id.asc(), AnnotationLabel.position.asc())
        .all()
    )
    if not rows:
        abort(404)

    first_label, _ = rows[0]
    label_meta = {
        "label_name": first_label.name,
        "excerpt": (first_label.excerpt or "").casefold().strip(),
        "count": len(rows),
        "dialogue_ids": set(),
        "sequence_ids": {},
    }

    for label, sequence in rows:
        dialogue_id = sequence.dialogue_external_id
        label_meta["dialogue_ids"].add(dialogue_id)

        sequence_entry = label_meta["sequence_ids"].setdefault(
            sequence.wmn_id,
            {"dialogue_id": dialogue_id, "wmn_count": 0},
        )
        sequence_entry["wmn_count"] += 1

    label_meta["dialogue_ids"] = sorted(label_meta["dialogue_ids"])

    return render_template("label.html", site_title=SITE_TITLE, label=label_meta)


@bp.get("/about")
def about_page():
    return render_template("about.html", site_title=SITE_TITLE)


def _parse_filters(args, corpus_options: list[dict]) -> dict:
    selected_wmn_names = {
        name
        for name, _ in WMN_TYPE_OPTIONS
        if args.get(name) == name
    }
    if len(args) == 0:
        selected_wmn_names = set(DEFAULT_WMN_NAMES)

    wmn_options = [
        {
            "name": name,
            "value": value,
            "checked": name in selected_wmn_names,
        }
        for name, value in WMN_TYPE_OPTIONS
    ]

    valid_label_names = {name for name, _ in LABEL_OPTIONS}
    selected_label_names = set(args.getlist("label-name")) & valid_label_names
    label_options = [
        {
            "name": name,
            "value": value,
            "checked": name in selected_label_names,
        }
        for name, value in LABEL_OPTIONS
    ]

    valid_context_names = {name for name, _ in CONTEXT_OPTIONS}
    selected_context_names = set(args.getlist("context")) & valid_context_names
    context_options = [
        {
            "name": name,
            "value": value,
            "checked": name in selected_context_names,
        }
        for name, value in CONTEXT_OPTIONS
    ]

    group_by = (args.get("group-by") or "SEQUENCE").upper()
    allowed_groups = {name for name, _ in GROUP_BY_OPTIONS}
    if group_by not in allowed_groups:
        group_by = "SEQUENCE"

    valid_corpora = {item["codename"] for item in corpus_options}
    selected_corpus_values = set(args.getlist("corpus")) & valid_corpora
    corpus_filter_options = [
        {
            "codename": item["codename"],
            "fullname": item["fullname"],
            "checked": item["codename"] in selected_corpus_values,
        }
        for item in corpus_options
    ]

    selected_csv_columns = [
        key
        for key, _, _ in CSV_COLUMN_OPTIONS
        if args.get(f"csv-col-{key}") == key
    ]
    if not selected_csv_columns:
        selected_csv_columns = list(DEFAULT_CSV_COLUMNS)

    csv_column_options = [
        {
            "key": key,
            "label": label,
            "param_name": f"csv-col-{key}",
            "checked": key in selected_csv_columns,
        }
        for key, _, label in CSV_COLUMN_OPTIONS
    ]

    return {
        "search": (args.get("search") or "").strip(),
        "search_cf": (args.get("search") or "").strip().casefold(),
        "label_names": {LABEL_VALUE_BY_NAME[name] for name in selected_label_names},
        "label_options": label_options,
        "context_values": {CONTEXT_VALUE_BY_NAME[name] for name in selected_context_names},
        "context_options": context_options,
        "corpus_values": selected_corpus_values,
        "corpus_options": corpus_filter_options,
        "group_by": group_by,
        "compact": args.get("mode") == "compact",
        "wmn_options": wmn_options,
        "csv_column_options": csv_column_options,
        "selected_csv_columns": selected_csv_columns,
        "selected_wmn_values": {
            WMN_TYPE_BY_NAME[name]
            for name in selected_wmn_names
            if name in WMN_TYPE_BY_NAME
        },
    }


def _corpus_options() -> list[dict]:
    corpus_map = OrderedDict(DEFAULT_CORPUS_OPTIONS)
    for row in Corpus.query.order_by(Corpus.fullname.asc()).all():
        corpus_map[row.codename] = row.fullname

    return [
        {"codename": codename, "fullname": fullname}
        for codename, fullname in corpus_map.items()
    ]


def _summaries_by_sequence(
    sequences: list[AnnotationSequence],
    filters: dict,
    corpus_options: list[dict],
) -> list[dict]:
    corpus_by_codename = {item["codename"]: item["fullname"] for item in corpus_options}

    results = []
    for sequence in sequences:
        if not _is_matching_wmn(sequence, filters):
            continue

        if not _sequence_satisfies_label_filter(sequence, filters):
            continue

        triggers: dict[str, dict] = {}
        indicators: dict[str, dict] = {}
        negotiations: list[dict[str, str]] = []

        for label in _sorted_labels(sequence.labels):
            if not _is_matching_label(label, sequence, filters):
                continue

            excerpt_key = (label.excerpt or "").casefold().strip()
            excerpt_hash = label.excerpt_hash or hashlib.md5(excerpt_key.encode("utf-8")).hexdigest()

            if label.name == "Trigger":
                _add_label_count(triggers, excerpt_key, excerpt_hash)
            elif label.name == "Indicator":
                _add_label_count(indicators, excerpt_key, excerpt_hash)
            elif label.name == "Negotiation":
                negotiations.append(
                    {
                        "excerpt": excerpt_key,
                        "hash": excerpt_hash,
                    }
                )

        if triggers or indicators or negotiations:
            results.append(
                {
                    "wmn_id": sequence.wmn_id,
                    "dialogue_id": sequence.dialogue_external_id,
                    "corpus_fullname": corpus_by_codename.get(sequence.corpus_codename, sequence.corpus_codename),
                    "context": sequence.context,
                    "wmn_type": sequence.wmn_type,
                    "wmn_type_short": _wmn_type_short(sequence.wmn_type),
                    "wmn_meaning": sequence.wmn_meaning,
                    "triggers": triggers,
                    "indicators": indicators,
                    "negotiations": negotiations,
                }
            )

    return results


def _summaries_by_dialogue(
    sequences: list[AnnotationSequence],
    filters: dict,
    corpus_options: list[dict],
) -> list[dict]:
    corpus_by_codename = {item["codename"]: item["fullname"] for item in corpus_options}

    results: OrderedDict[str, dict] = OrderedDict()

    for sequence in sequences:
        if not _is_matching_wmn(sequence, filters):
            continue

        if not _sequence_satisfies_label_filter(sequence, filters):
            continue

        matching_triggers: dict[str, int] = {}
        matching_indicators: dict[str, int] = {}
        matching_negotiations: list[dict[str, str]] = []

        for label in _sorted_labels(sequence.labels):
            if not _is_matching_label(label, sequence, filters):
                continue

            excerpt = (label.excerpt or "").casefold().strip()
            if label.name == "Trigger":
                matching_triggers[excerpt] = matching_triggers.get(excerpt, 0) + 1
            elif label.name == "Indicator":
                matching_indicators[excerpt] = matching_indicators.get(excerpt, 0) + 1
            elif label.name == "Negotiation":
                matching_negotiations.append(
                    {
                        "excerpt": excerpt,
                        "hash": label.excerpt_hash or hashlib.md5(excerpt.encode("utf-8")).hexdigest(),
                    }
                )

        if not matching_triggers and not matching_indicators and not matching_negotiations:
            continue

        dialogue_id = sequence.dialogue_external_id
        if dialogue_id not in results:
            results[dialogue_id] = {
                "dialogue_id": dialogue_id,
                "corpus_fullname": corpus_by_codename.get(sequence.corpus_codename, sequence.corpus_codename),
                "context": sequence.context,
                "sequence_ids": {},
                "wmn_meanings": set(),
                "triggers": {},
                "indicators": {},
                "negotiations": [],
            }

        results[dialogue_id]["sequence_ids"][sequence.wmn_id] = sequence.wmn_type
        results[dialogue_id]["wmn_meanings"].add(sequence.wmn_meaning)

        for excerpt, count in matching_triggers.items():
            _add_label_count(
                results[dialogue_id]["triggers"],
                excerpt,
                hashlib.md5(excerpt.encode("utf-8")).hexdigest(),
                count,
            )
        for excerpt, count in matching_indicators.items():
            _add_label_count(
                results[dialogue_id]["indicators"],
                excerpt,
                hashlib.md5(excerpt.encode("utf-8")).hexdigest(),
                count,
            )
        results[dialogue_id]["negotiations"].extend(matching_negotiations)

    final = []
    for summary in results.values():
        final.append(
            {
                **summary,
                "sequence_ids": {
                    wmn_id: _wmn_type_short(wmn_type)
                    for wmn_id, wmn_type in sorted(summary["sequence_ids"].items())
                },
                "wmn_meanings": sorted(summary["wmn_meanings"]),
            }
        )

    return final


def _summaries_by_label(
    sequences: list[AnnotationSequence],
    filters: dict,
    corpus_options: list[dict],
) -> list[dict]:
    corpus_by_codename = {item["codename"]: item["fullname"] for item in corpus_options}

    results: dict[str, dict] = {}

    for sequence in sequences:
        if not _is_matching_wmn(sequence, filters):
            continue

        if not _sequence_satisfies_label_filter(sequence, filters):
            continue

        for label in _sorted_labels(sequence.labels):
            if not _is_matching_label(label, sequence, filters):
                continue

            excerpt = (label.excerpt or "").casefold().strip()
            if excerpt not in results:
                results[excerpt] = {
                    "label_type": label.name,
                    "excerpt": excerpt,
                    "excerpt_hash": label.excerpt_hash or hashlib.md5(excerpt.encode("utf-8")).hexdigest(),
                    "count": 1,
                    "dialogue_ids": {sequence.dialogue_external_id},
                    "sequence_ids": {
                        sequence.wmn_id: {
                            "dialogue_id": sequence.dialogue_external_id,
                            "wmn_count": 1,
                            "wmn_type_short": _wmn_type_short(sequence.wmn_type),
                        }
                    },
                    "corpora": {corpus_by_codename.get(sequence.corpus_codename, sequence.corpus_codename)},
                    "contexts": {sequence.context},
                    "wmn_meanings": {sequence.wmn_meaning},
                }
            else:
                results[excerpt]["count"] += 1
                results[excerpt]["dialogue_ids"].add(sequence.dialogue_external_id)
                sequence_entry = results[excerpt]["sequence_ids"].setdefault(
                    sequence.wmn_id,
                    {
                        "dialogue_id": sequence.dialogue_external_id,
                        "wmn_count": 0,
                        "wmn_type_short": _wmn_type_short(sequence.wmn_type),
                    },
                )
                sequence_entry["wmn_count"] += 1
                results[excerpt]["corpora"].add(
                    corpus_by_codename.get(sequence.corpus_codename, sequence.corpus_codename)
                )
                results[excerpt]["contexts"].add(sequence.context)
                results[excerpt]["wmn_meanings"].add(sequence.wmn_meaning)

    final = []
    for excerpt in sorted(results.keys()):
        summary = results[excerpt]
        final.append(
            {
                **summary,
                "dialogue_ids": sorted(summary["dialogue_ids"]),
                "corpora": sorted(summary["corpora"]),
                "contexts": sorted(summary["contexts"]),
                "wmn_meanings": sorted(summary["wmn_meanings"]),
            }
        )

    return final


def _csv_rows_by_sequence(sequences: list[AnnotationSequence], filters: dict) -> list[dict]:
    rows = []

    for sequence in sequences:
        if not _is_matching_wmn(sequence, filters):
            continue

        sorted_labels = _sorted_labels(sequence.labels)
        has_matching_label = any(
            _is_matching_label(label, sequence, filters)
            for label in sorted_labels
        )
        if not has_matching_label:
            continue

        triggers: list[str] = []
        indicators: list[str] = []
        negotiations: list[str] = []
        seen_triggers: set[str] = set()
        seen_indicators: set[str] = set()
        seen_negotiations: set[str] = set()

        for label in sorted_labels:
            if label.name not in VALID_LABELS:
                continue

            excerpt = (label.excerpt or "").strip()
            if label.name == "Trigger":
                key = excerpt.casefold()
                if key not in seen_triggers:
                    seen_triggers.add(key)
                    triggers.append(excerpt)
            elif label.name == "Indicator":
                key = excerpt.casefold()
                if key not in seen_indicators:
                    seen_indicators.add(key)
                    indicators.append(excerpt)
            elif label.name == "Negotiation":
                key = excerpt.casefold()
                if key not in seen_negotiations:
                    seen_negotiations.add(key)
                    negotiations.append(excerpt)

        rows.append(
            {
                "wmn_id": sequence.wmn_id,
                "dialogue_id": sequence.dialogue_external_id,
                "corpus": _corpus_shortname(sequence.corpus_codename),
                "wmn_type": sequence.wmn_type,
                "triggers": triggers,
                "indicators": indicators,
                "negotiations": negotiations,
            }
        )

    return rows


def _csv_join_label_values(values: list[str]) -> str:
    if not values:
        return ""
    # Use pipe separator to avoid ambiguity with CSV commas.
    quoted_values: list[str] = []
    for value in values:
        escaped_value = value.replace('"', '""')
        quoted_values.append(f'"{escaped_value}"')
    return " | ".join(quoted_values)


def _csv_value_for_column(column: str, row: dict) -> str:
    if column == "wmn_id":
        return row["wmn_id"]
    if column == "corpus":
        return row["corpus"]
    if column == "wmn_type":
        return row["wmn_type"]
    if column == "trigger":
        return _csv_join_label_values(row["triggers"])
    if column == "indicator":
        return _csv_join_label_values(row["indicators"])
    if column == "negotiation":
        return _csv_join_label_values(row["negotiations"])
    if column == "wmn_link":
        return url_for(
            "main.sequence_page",
            dialogue_id=row["dialogue_id"],
            wmn_id=row["wmn_id"],
            _external=True,
        )
    return ""


def _corpus_shortname(codename: str) -> str:
    corpus_key = (codename or "").casefold()
    if "winning-args" in corpus_key or "reddit" in corpus_key:
        return "reddit"
    if "switchboard" in corpus_key:
        return "switchboard"
    if "bnc" in corpus_key:
        return "bnc"
    return codename


def _is_matching_wmn(sequence: AnnotationSequence, filters: dict) -> bool:
    if not sequence.labels:
        return False

    if sequence.wmn_type not in VALID_WMN_TYPES:
        return False

    if filters["corpus_values"] and sequence.corpus_codename not in filters["corpus_values"]:
        return False

    if filters["context_values"] and sequence.context not in filters["context_values"]:
        return False

    if filters["selected_wmn_values"] and sequence.wmn_type not in filters["selected_wmn_values"]:
        return False

    return True


def _label_matches_search(label: AnnotationLabel, sequence: AnnotationSequence, search_cf: str) -> bool:
    if not search_cf:
        return True

    return (
        search_cf in (label.excerpt or "").casefold().strip()
        or search_cf in (sequence.dialogue_external_id or "").casefold()
        or search_cf in (sequence.wmn_id or "").casefold()
    )


def _is_matching_label(label: AnnotationLabel, sequence: AnnotationSequence, filters: dict) -> bool:
    if label.name not in VALID_LABELS:
        return False

    if filters["label_names"] and label.name not in filters["label_names"]:
        return False

    return _label_matches_search(label, sequence, filters["search_cf"])


def _sequence_satisfies_label_filter(sequence: AnnotationSequence, filters: dict) -> bool:
    present_types = {
        label.name
        for label in sequence.labels
        if label.name in VALID_LABELS and _label_matches_search(label, sequence, filters["search_cf"])
    }

    if filters["label_names"]:
        return filters["label_names"] <= present_types

    return bool(present_types)


def _add_label_count(bucket: dict[str, dict], excerpt: str, excerpt_hash: str, count: int = 1) -> None:
    if excerpt not in bucket:
        bucket[excerpt] = {"hash": excerpt_hash, "count": count}
    else:
        bucket[excerpt]["count"] += count


def _sorted_labels(labels: list[AnnotationLabel]) -> list[AnnotationLabel]:
    def key(label: AnnotationLabel):
        start_index = label.start_index if label.start_index is not None else 10_000_000
        start_offset = label.start_offset if label.start_offset is not None else 10_000_000
        return (start_index, start_offset, label.position)

    return sorted(labels, key=key)


def _annotate_utterances(
    utterances: list[dict[str, str]], labels: list[dict]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    label_links: list[dict[str, str]] = []
    inserts: list[tuple[int, int, int]] = []

    label_id = 0

    for label in labels:
        start_index = _safe_int(label.get("start_index"))
        end_index = _safe_int(label.get("end_index"))
        start_offset_raw = _safe_int(label.get("start_offset"))
        end_offset_raw = _safe_int(label.get("end_offset"))
        name = label.get("name")

        if (
            start_index is None
            or end_index is None
            or start_offset_raw is None
            or end_offset_raw is None
            or name not in VALID_LABELS
        ):
            continue
        if start_index < 0 or end_index < 0:
            continue
        if start_index >= len(utterances) or end_index >= len(utterances):
            continue

        start_span = f'<span class="{name}" id="label-{label_id}">'
        end_span = "</span>"

        add_to_start = 0
        for insert in inserts:
            if insert[0] == start_index and insert[1] < start_offset_raw:
                add_to_start += insert[2]
        start_offset = start_offset_raw + add_to_start

        utterances[start_index]["text"] = (
            utterances[start_index]["text"][:start_offset]
            + start_span
            + utterances[start_index]["text"][start_offset:]
        )
        inserts.append((start_index, start_offset_raw, len(start_span)))

        label_links.append(
            {
                "name": name,
                "excerpt": label.get("excerpt", ""),
                "link": f"label-{label_id}",
            }
        )
        label_id += 1

        if start_index < end_index:
            utterances[start_index]["text"] += end_span

        if end_index - start_index == 1:
            utterances[start_index + 1]["text"] = start_span + utterances[start_index + 1]["text"]
            inserts.append((start_index + 1, 0, len(start_span)))
        elif end_index - start_index > 1:
            for idx in range(start_index + 1, end_index):
                utterances[idx]["text"] = start_span + utterances[idx]["text"] + end_span
                inserts.append((idx, 0, len(start_span)))
                inserts.append((idx, len(utterances[idx]["text"]), len(end_span)))
            utterances[end_index]["text"] = start_span + utterances[end_index]["text"]
            inserts.append((end_index, 0, len(start_span)))

        add_to_end = 0
        for insert in inserts:
            if insert[0] == end_index and insert[1] < end_offset_raw:
                add_to_end += insert[2]
        end_offset = end_offset_raw + add_to_end

        utterances[end_index]["text"] = (
            utterances[end_index]["text"][:end_offset]
            + end_span
            + utterances[end_index]["text"][end_offset:]
        )
        inserts.append((end_index, end_offset_raw, len(end_span)))

    return utterances, label_links


def _label_truncation_bounds(
    labels: list[dict], utterance_count: int
) -> dict[str, int | bool | None]:
    first_label_index: int | None = None
    last_label_index: int | None = None

    for label in labels:
        start_index = _safe_int(label.get("start_index"))
        end_index = _safe_int(label.get("end_index"))
        index_candidates = [
            idx
            for idx in (start_index, end_index)
            if idx is not None and 0 <= idx < utterance_count
        ]
        if not index_candidates:
            continue

        current_first = min(index_candidates)
        current_last = max(index_candidates)
        if first_label_index is None or current_first < first_label_index:
            first_label_index = current_first
        if last_label_index is None or current_last > last_label_index:
            last_label_index = current_last

    if first_label_index is None or last_label_index is None:
        return {
            "first_label_index": None,
            "last_label_index": None,
            "omitted_before": 0,
            "omitted_after": 0,
            "is_truncated": False,
        }

    omitted_before = first_label_index
    omitted_after = max(0, utterance_count - last_label_index - 1)

    return {
        "first_label_index": first_label_index,
        "last_label_index": last_label_index,
        "omitted_before": omitted_before,
        "omitted_after": omitted_after,
        "is_truncated": omitted_before > 0 or omitted_after > 0,
    }


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _wmn_type_short(wmn_type: str | None) -> str:
    if wmn_type is None:
        return ""
    normalized = wmn_type.strip()
    if not normalized:
        return ""
    return WMN_TYPE_NAME_BY_VALUE.get(normalized, normalized)
