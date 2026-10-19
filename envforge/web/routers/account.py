"""/api/me — the signed-in user's own profile, password and provider keys.

Provider keys belong to the user: one set per account, used by every workspace
they run in. Secrets are write-only from here; responses carry the masked form.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from envforge.web import services as svc
from envforge.web.auth import current_user
from envforge.web.auth import local as local_auth
from envforge.web.db import get_db
from envforge.web.models import ProviderKey, User
from envforge.web.routers.auth import _set_cookie, _user_out
from envforge.web.schemas import KeyIn, KeyOut, PasswordIn, ProfileIn, UserOut
from envforge.web.security import client_ip, enforce
from envforge.web.settings import get_settings

router = APIRouter(prefix="/api/me", tags=["account"])


def _key_out(r: ProviderKey) -> KeyOut:
    return KeyOut(
        provider=r.provider,
        masked=r.last4,
        base_url=r.base_url,
        updated_at=r.updated_at,
        verified_at=r.verified_at,
        n_models=len(r.models) if r.models is not None else None,
        verify_note=r.verify_note,
    )


@router.patch("", response_model=UserOut)
def update_profile(
    body: ProfileIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> UserOut:
    user.display_name = body.display_name.strip()
    db.add(user)
    db.commit()
    return _user_out(user)


@router.post("/password", status_code=204)
def change_password(
    body: PasswordIn,
    request: Request,
    resp: Response,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> None:
    """Local accounts only. Signs out every other session; this one gets a new cookie."""
    s = get_settings()
    if s.AUTH_PROVIDER != "local" or not user.password_hash:
        raise HTTPException(400, "passwords are managed by your sign-in provider")
    enforce(f"password:{client_ip(request)}", s.LOGIN_ATTEMPTS_PER_MINUTE, 60)
    if not local_auth.verify_password(body.current_password, user.password_hash):
        raise HTTPException(400, "current password is wrong")
    user.password_hash = local_auth.hash_password(body.new_password)
    user.session_epoch = (user.session_epoch or 0) + 1
    db.add(user)
    db.commit()
    _set_cookie(resp, local_auth.issue_session(user))


@router.get("/keys", response_model=list[KeyOut])
def list_keys(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [_key_out(r) for r in svc.user_keys(db, user.id)]


@router.put("/keys", response_model=list[KeyOut])
def put_key(body: KeyIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    svc.set_key(db, user, body.provider, body.secret, body.base_url)
    return [_key_out(r) for r in svc.user_keys(db, user.id)]


@router.delete("/keys/{provider}", response_model=list[KeyOut])
def delete_key(provider: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    svc.delete_key(db, user, provider)
    return [_key_out(r) for r in svc.user_keys(db, user.id)]
