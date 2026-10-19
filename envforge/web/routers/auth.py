"""/api/auth — local sign-up/login/logout; /me and /me/usage work for both providers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from envforge import __version__
from envforge.web.auth import current_user
from envforge.web.auth import local as local_auth
from envforge.web.db import get_db
from envforge.web.models import User
from envforge.web.schemas import LoginIn, ServerInfo, SignupIn, UserOut
from envforge.web.security import client_ip, enforce
from envforge.web.services import PROVIDERS, ensure_personal_workspace, user_usage
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
def signup(
    body: SignupIn, request: Request, resp: Response, db: Session = Depends(get_db)
) -> UserOut:
    s = get_settings()
    if s.AUTH_PROVIDER != "local":
        raise HTTPException(400, "sign-up is handled by Clerk on this server")
    enforce(f"signup:{client_ip(request)}", s.SIGNUPS_PER_HOUR, 3600)
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
def login(
    body: LoginIn, request: Request, resp: Response, db: Session = Depends(get_db)
) -> UserOut:
    s = get_settings()
    if s.AUTH_PROVIDER != "local":
        raise HTTPException(400, "login is handled by Clerk on this server")
    email = body.email.lower()
    enforce(f"login-ip:{client_ip(request)}", s.LOGIN_ATTEMPTS_PER_MINUTE, 60)
    enforce(f"login-email:{email}", s.LOGIN_ATTEMPTS_PER_MINUTE, 60)
    user = db.scalar(select(User).where(User.email == email))
    if not user or not user.password_hash:
        local_auth.burn_verify(body.password)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    if not local_auth.verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    _set_cookie(resp, local_auth.issue_session(user))
    return _user_out(user)


@router.post("/logout", status_code=204)
def logout(resp: Response) -> None:
    resp.delete_cookie(get_settings().COOKIE_NAME, path="/")


@router.post("/logout-all", status_code=204)
def logout_all(
    resp: Response, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    """Invalidate every session this user has, on every device."""
    user.session_epoch = (user.session_epoch or 0) + 1
    db.add(user)
    db.commit()
    resp.delete_cookie(get_settings().COOKIE_NAME, path="/")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> UserOut:
    return _user_out(user)


@router.get("/me/usage")
def my_usage(
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Your own usage across every workspace you belong to."""
    return user_usage(db, user, days)
