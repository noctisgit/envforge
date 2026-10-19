"""`envforge admin …` — server administration from a shell on the box.

    uv run envforge admin users
    uv run envforge admin reset-password me@example.com
    docker compose exec api envforge admin reset-password me@example.com   # the Docker stack

Talks straight to DATABASE_URL, so it works even when nobody can log in.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

admin_app = typer.Typer(help="Server administration (web layer).", no_args_is_help=True)
console = Console()


def _session():
    from envforge.web.db import session_factory

    return session_factory()()


def _user(db, email: str):
    from envforge.web.models import User

    user = db.scalar(select(User).where(User.email == email.lower()))
    if user is None:
        raise typer.BadParameter(f"no user with email {email}")
    return user


@admin_app.command("users")
def list_users() -> None:
    """List accounts with their workspace and run counts."""
    from envforge.web.models import Run, User, WorkspaceMember

    with _session() as db:
        t = Table("email", "name", "admin", "workspaces", "runs", "created")
        for u in db.scalars(select(User).order_by(User.created_at)):
            n_ws = db.scalar(
                select(func.count())
                .select_from(WorkspaceMember)
                .where(WorkspaceMember.user_id == u.id)
            )
            n_runs = db.scalar(select(func.count()).select_from(Run).where(Run.created_by == u.id))
            t.add_row(
                u.email,
                u.display_name,
                "yes" if u.is_admin else "",
                str(n_ws),
                str(n_runs),
                u.created_at.strftime("%Y-%m-%d"),
            )
        console.print(t)


@admin_app.command("reset-password")
def reset_password(
    email: str,
    password: str = typer.Option(
        ..., prompt=True, hide_input=True, confirmation_prompt=True, help="new password (≥10 chars)"
    ),
) -> None:
    """Set a new password and sign the user out everywhere."""
    from envforge.web.auth.local import hash_password

    if len(password) < 10:
        raise typer.BadParameter("password must be at least 10 characters")
    with _session() as db:
        user = _user(db, email)
        user.password_hash = hash_password(password)
        user.session_epoch = (user.session_epoch or 0) + 1
        db.commit()
    console.print(f"password reset for {email}; existing sessions signed out")


@admin_app.command("set-admin")
def set_admin(email: str, off: bool = typer.Option(False, "--off", help="revoke instead")) -> None:
    """Grant (or with --off revoke) the admin flag."""
    with _session() as db:
        user = _user(db, email)
        user.is_admin = not off
        db.commit()
    console.print(f"{email}: admin={'no' if off else 'yes'}")


@admin_app.command("sign-out")
def sign_out(email: str) -> None:
    """Invalidate every session a user has."""
    with _session() as db:
        user = _user(db, email)
        user.session_epoch = (user.session_epoch or 0) + 1
        db.commit()
    console.print(f"{email} signed out everywhere")
