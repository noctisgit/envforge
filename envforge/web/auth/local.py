"""Local auth: argon2 password hashes + HS256 session tokens in an HttpOnly cookie."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy.orm import Session

from envforge.web.models import User
from envforge.web.settings import get_settings

_ph = PasswordHasher()


def hash_password(pw: str) -> str:
    return _ph.hash(pw)


def verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return _ph.verify(pw_hash, pw)
    except VerifyMismatchError:
        return False


def issue_session(user: User) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": user.id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=s.SESSION_TTL_HOURS)).timestamp()),
        "typ": "session",
    }
    return jwt.encode(payload, s.APP_SECRET_KEY, algorithm="HS256")


def user_from_session(db: Session, token: str) -> User | None:
    s = get_settings()
    try:
        payload = jwt.decode(token, s.APP_SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "session":
        return None
    return db.get(User, payload.get("sub"))
