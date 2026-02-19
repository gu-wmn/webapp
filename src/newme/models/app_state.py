from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from ..extensions import db


class AppState(db.Model):
    __tablename__ = "app_state"

    key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.String(255), nullable=False)

    @classmethod
    def mark_initialized(cls) -> None:
        state = db.session.get(cls, "initialized")
        if state is None:
            state = cls(key="initialized", value="true")
            db.session.add(state)
        else:
            state.value = "true"

    @classmethod
    def is_initialized(cls) -> bool:
        state = db.session.get(cls, "initialized")
        return bool(state and state.value == "true")

    @classmethod
    def is_initialized_safe(cls) -> bool:
        try:
            if not inspect(db.engine).has_table(cls.__tablename__):
                return False
            return cls.is_initialized()
        except SQLAlchemyError:
            return False
