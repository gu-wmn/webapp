from ..extensions import db


class Corpus(db.Model):
    __tablename__ = "corpora"

    id = db.Column(db.Integer, primary_key=True)
    codename = db.Column(db.String(64), unique=True, nullable=False, index=True)
    fullname = db.Column(db.String(255), nullable=False)
    license_url = db.Column(db.String(512))
    url = db.Column(db.String(512))
    download_url = db.Column(db.String(1024))
    md5sum = db.Column(db.String(64))

    dialogues = db.relationship(
        "Dialogue",
        backref="corpus",
        cascade="all, delete-orphan",
        lazy=True,
    )


class Dialogue(db.Model):
    __tablename__ = "dialogues"

    id = db.Column(db.Integer, primary_key=True)
    corpus_id = db.Column(db.Integer, db.ForeignKey("corpora.id"), nullable=False, index=True)
    external_id = db.Column(db.String(255), nullable=False)

    utterances = db.relationship(
        "Utterance",
        backref="dialogue",
        cascade="all, delete-orphan",
        lazy=True,
    )

    __table_args__ = (
        db.UniqueConstraint("corpus_id", "external_id", name="uq_dialogue_corpus_external"),
    )


class Utterance(db.Model):
    __tablename__ = "utterances"

    id = db.Column(db.Integer, primary_key=True)
    dialogue_id = db.Column(db.Integer, db.ForeignKey("dialogues.id"), nullable=False, index=True)
    position = db.Column(db.Integer, nullable=False)
    external_id = db.Column(db.String(255))
    author = db.Column(db.String(255), nullable=False)
    text = db.Column(db.Text, nullable=False)
    reply_to_external_id = db.Column(db.String(255))

    __table_args__ = (
        db.UniqueConstraint("dialogue_id", "position", name="uq_utterance_dialogue_position"),
    )
