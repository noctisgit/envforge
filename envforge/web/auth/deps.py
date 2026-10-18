"""FastAPI dependencies: resolve the current user whichever provider is configured."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from envforge.web.db import get_db
from envforge.web.models import User
from envforge.web.settings import get_settings


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    s = get_settings()
    if s.AUTH_PROVIDER == "clerk":
        from envforge.web.auth import clerk

        token = _bearer(request) or request.cookies.get("__session")
        if not token:
            return None
        claims = clerk.verify_token(token)
        return clerk.user_from_claims(db, claims) if claims else None

    from envforge.web.auth import local

    token = request.cookies.get(s.COOKIE_NAME) or _bearer(request)
    if not token:
        return None
    return local.user_from_session(db, token)


def current_user(user: User | None = Depends(optional_user)) -> User:
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user
