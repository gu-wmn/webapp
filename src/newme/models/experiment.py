from __future__ import annotations

from datetime import datetime, timezone

from ..extensions import db


class Experiment(db.Model):
    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String, nullable=False, index=True)
    name = db.Column(db.String, nullable=False)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
