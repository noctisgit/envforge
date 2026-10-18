"""Clerk auth: verify Clerk session JWTs against the instance's JWKS.

The Next.js frontend obtains a token with Clerk's `getToken()` and sends it as
`Authorization: Bearer <jwt>`. We verify signature (RS256), issuer and expiry,
then find-or-create the local user row keyed by Clerk's user id. We never store
passwords for Clerk users.

Reference: Clerk session tokens are standard JWTs; `sub` is the Clerk user id.
Email is not in the default token — we read it from the custom claim `email` if
you add it to Clerk's session token template, else synthesise a placeholder.
"""

from __future__ import annotations

from functools import lru_cache

import jwt
from jwt import PyJWKClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from envforge.web.models import User
from envforge.web.settings import get_settings


@lru_cache
def _jwk_client() -> PyJWKClient:
    return PyJWKClient(get_settings().CLERK_JWKS_URL, cache_keys=True)


def verify_token(token: str) -> dict | None:
    s = get_settings()
    try:
        key = _jwk_client().get_signing_key_from_jwt(token).key
        return jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=s.CLERK_ISSUER,
            options={"require": ["exp", "iat", "sub"]},
            leeway=10,
        )
    except jwt.PyJWTError:
        return None


def user_from_claims(db: Session, claims: dict) -> User:
    clerk_id = claims["sub"]
    user = db.scalar(select(User).where(User.clerk_user_id == clerk_id))
    if user:
        return user
    email = claims.get("email") or f"{clerk_id}@clerk.local"
    user = User(
        email=email,
        display_name=claims.get("name") or claims.get("first_name") or "",
        clerk_user_id=clerk_id,
    )
    db.add(user)
    db.flush()
    from envforge.web.services import ensure_personal_workspace

    ensure_personal_workspace(db, user)
    db.commit()
    return user
