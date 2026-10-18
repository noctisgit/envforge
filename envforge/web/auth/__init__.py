"""Authentication. Two providers behind one dependency:

    AUTH_PROVIDER=local   email + password (argon2), signed session cookie
    AUTH_PROVIDER=clerk   Clerk-issued JWT in `Authorization: Bearer` (or Clerk's cookie),
                          verified against Clerk's JWKS; users auto-provisioned on first sight

Everything else in the app only ever calls `current_user`.
"""

from envforge.web.auth.deps import current_user, optional_user

__all__ = ["current_user", "optional_user"]
