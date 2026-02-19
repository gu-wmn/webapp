from ..extensions import db


class AnnotationSequence(db.Model):
    __tablename__ = "annotation_sequences"

    id = db.Column(db.Integer, primary_key=True)
    wmn_id = db.Column(db.String(255), unique=True, nullable=False, index=True)
    corpus_codename = db.Column(db.String(64), nullable=False, index=True)
    dialogue_external_id = db.Column(db.String(255), nullable=False, index=True)
    context = db.Column(db.String(128), nullable=False)
    wmn_type = db.Column(db.String(128), nullable=False)
    wmn_meaning = db.Column(db.String(128), nullable=False)
    annotator = db.Column(db.String(255), nullable=False)
    comment = db.Column(db.Text)
    prediction = db.Column(db.JSON)

    labels = db.relationship(
        "AnnotationLabel",
        backref="annotation",
        cascade="all, delete-orphan",
        lazy=True,
    )


class AnnotationLabel(db.Model):
    __tablename__ = "annotation_labels"

    id = db.Column(db.Integer, primary_key=True)
    annotation_id = db.Column(
        db.Integer,
        db.ForeignKey("annotation_sequences.id"),
        nullable=False,
        index=True,
    )
    position = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(64), nullable=False, index=True)
    start_index = db.Column(db.Integer)
    end_index = db.Column(db.Integer)
    start_offset = db.Column(db.Integer)
    end_offset = db.Column(db.Integer)
    excerpt = db.Column(db.Text, nullable=False)
    excerpt_hash = db.Column(db.String(32), nullable=False, index=True)

    __table_args__ = (
        db.UniqueConstraint("annotation_id", "position", name="uq_annotation_label_position"),
    )
