"""/api/auth — local sign-up/login/logout; /me works for both providers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from envforge import __version__
from envforge.web.auth import current_user
from envforge.web.auth import local as local_auth
from envforge.web.db import get_db
from envforge.web.models import User
from envforge.web.schemas import LoginIn, ServerInfo, SignupIn, UserOut
from envforge.web.services import PROVIDERS, ensure_personal_workspace
from envforge.web.settings import get_settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user_out(u: User) -> UserOut:
    return UserOut(
        id=u.id,
        email=u.email,
        display_name=u.display_name,
        is_admin=u.is_admin,
        auth_provider=get_settings().AUTH_PROVIDER,
    )


def _set_cookie(resp: Response, token: str) -> None:
    s = get_settings()
    resp.set_cookie(
        s.COOKIE_NAME,
        token,
        httponly=True,
        secure=s.COOKIE_SECURE,
        samesite="lax",
        max_age=s.SESSION_TTL_HOURS * 3600,
        path="/",
    )


@router.get("/info", response_model=ServerInfo)
def info() -> ServerInfo:
    s = get_settings()
    return ServerInfo(
        auth_provider=s.AUTH_PROVIDER,
        allow_signup=s.ALLOW_SIGNUP,
        clerk_publishable_key=s.CLERK_PUBLISHABLE_KEY,
        allow_server_keys=s.ALLOW_SERVER_KEYS,
        providers=list(PROVIDERS),
        version=__version__,
    )


@router.post("/signup", response_model=UserOut, status_code=201)
def signup(body: SignupIn, resp: Response, db: Session = Depends(get_db)) -> UserOut:
    s = get_settings()
    if s.AUTH_PROVIDER != "local":
        raise HTTPException(400, "sign-up is handled by Clerk on this server")
    first_user = db.scalar(select(User.id).limit(1)) is None
    if not s.ALLOW_SIGNUP and not first_user:
        raise HTTPException(403, "sign-up is disabled; ask an admin")
    if db.scalar(select(User).where(User.email == body.email.lower())):
        raise HTTPException(409, "email already registered")
    user = User(
        email=body.email.lower(),
        display_name=body.display_name,
        password_hash=local_auth.hash_password(body.password),
        is_admin=first_user,  # the first account administers the server
    )
    db.add(user)
    db.flush()
    ensure_personal_workspace(db, user)
    db.commit()
    _set_cookie(resp, local_auth.issue_session(user))
    return _user_out(user)


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, resp: Response, db: Session = Depends(get_db)) -> UserOut:
    if get_settings().AUTH_PROVIDER != "local":
        raise HTTPException(400, "login is handled by Clerk on this server")
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if (
        not user
        or not user.password_hash
        or not local_auth.verify_password(body.password, user.password_hash)
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    _set_cookie(resp, local_auth.issue_session(user))
    return _user_out(user)


@router.post("/logout", status_code=204)
def logout(resp: Response) -> None:
    resp.delete_cookie(get_settings().COOKIE_NAME, path="/")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> UserOut:
    return _user_out(user)
