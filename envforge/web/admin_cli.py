"""`envforge admin …` — server administration from a shell on the box.

    uv run envforge admin users
    uv run envforge admin create-user me@example.com --admin
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
def list_users(
    count: bool = typer.Option(False, "--count", help="print only the number of accounts"),
) -> None:
    """List accounts with their workspace and run counts."""
    from envforge.web.models import Run, User, WorkspaceMember

    with _session() as db:
        if count:
            print(db.scalar(select(func.count()).select_from(User)))
            return
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


def _read_password(password_stdin: bool) -> str:
    """The new password from stdin (scripts: never in argv or the environment) or a prompt."""
    import sys

    if password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = typer.prompt("password", hide_input=True, confirmation_prompt=True)
    if len(password) < 10:
        raise typer.BadParameter("password must be at least 10 characters")
    return password


@admin_app.command("create-user")
def create_user(
    email: str,
    name: str = typer.Option("", "--name", help="display name"),
    admin: bool = typer.Option(False, "--admin", help="make this account a server admin"),
    password_stdin: bool = typer.Option(
        False, "--password-stdin", help="read the password from the first line of stdin"
    ),
) -> None:
    """Create a local (email + password) account with its own private workspace.

    Used by install.sh to create the first admin, so nobody can claim the server by
    signing up first. `--password-stdin` keeps the password out of argv and `ps`.
    """
    from pydantic import TypeAdapter, ValidationError
    from pydantic.networks import EmailStr

    from envforge.web.auth.local import hash_password
    from envforge.web.models import User
    from envforge.web.services import ensure_personal_workspace

    try:
        email = TypeAdapter(EmailStr).validate_python(email.strip()).lower()
    except ValidationError as e:
        raise typer.BadParameter(f"not a valid email address: {email}") from e
    password = _read_password(password_stdin)
    with _session() as db:
        if db.scalar(select(User).where(User.email == email)):
            raise typer.BadParameter(f"{email} already has an account")
        user = User(
            email=email,
            display_name=name.strip()[:120],
            password_hash=hash_password(password),
            is_admin=admin,
        )
        db.add(user)
        db.flush()
        ws = ensure_personal_workspace(db, user)
        db.commit()
        console.print(f"created {email}{' (admin)' if admin else ''} with workspace '{ws.name}'")


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
