"""Secrets at rest.

Provider API keys are encrypted with Fernet (AES-128-CBC + HMAC) using a key
derived from APP_SECRET_KEY. A database dump therefore does not leak keys; the
server's .env does. Rotate APP_SECRET_KEY and all stored keys become unreadable —
keep that in mind before changing it.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from envforge.web.settings import get_settings


def _fernet() -> Fernet:
    secret = get_settings().APP_SECRET_KEY.encode()
    # Fernet wants a 32-byte url-safe base64 key; derive it from the app secret.
    key = base64.urlsafe_b64encode(hashlib.sha256(b"envforge-kek:" + secret).digest())
    return Fernet(key)


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as e:
        raise RuntimeError("stored key cannot be decrypted (APP_SECRET_KEY changed?)") from e


def mask(plaintext: str) -> str:
    """'sk-ant-api03-…xyz1' style display string; never the full key."""
    tail = plaintext[-4:] if len(plaintext) >= 8 else "****"
    return f"…{tail}"
