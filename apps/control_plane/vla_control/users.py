"""Trusted local-console account administration; passwords never enter argv.

Examples:
    python -m vla_control.users create-admin --username admin --database-url sqlite:///.../control.db
    python -m vla_control.users reset-password --username admin --database-url sqlite:///.../control.db
    python -m vla_control.users revoke-sessions --all --database-url sqlite:///.../control.db
"""

import argparse
import getpass
import os

from sqlalchemy import delete, select

from .auth import AuthManager, validate_password
from .auth_models import User, UserSession
from .common import APIError
from .db import Database
from .settings import Settings


def create_initial_admin(db: Database, username: str, password: str, display_name: str = "") -> dict:
    """Atomically seed an empty installation, without changing existing users."""
    return AuthManager(db, Settings()).create_user(username, password, "admin", display_name, initial_only=True)


def revoke_sessions(db: Database, username: str | None = None) -> int:
    """Revoke all sessions, or only one account, without changing credentials."""
    manager = AuthManager(db, Settings(database_url=str(db.engine.url)))
    with db.write() as session:
        manager.lock_accounts(session)
        statement = delete(UserSession)
        details = {}
        if username is not None:
            user = session.scalar(select(User).where(User.username == username.strip().lower()))
            if user is None:
                raise APIError(404, "NOT_FOUND", "账户不存在")
            statement = statement.where(UserSession.user_id == user.id)
            details.update(target_user_id=user.id, target_username=user.username)
        count = session.execute(statement).rowcount
        manager._audit(session, None, "auth.sessions_revoked", 200, {**details, "count": count})
        return count


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="本机控制台管理平台账户与会话；设置密码时交互输入")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create-admin", "reset-password"):
        command = commands.add_parser(name)
        command.add_argument("--username", required=True)
        command.add_argument("--database-url", default=None,
                             help="精确数据库 URL；未提供时使用 VLA_DATABASE_URL")
        if name == "create-admin":
            command.add_argument("--display-name", default="")
    revoke = commands.add_parser("revoke-sessions", help="撤销登录会话；HTTP 切换 HTTPS 时应执行一次")
    selector = revoke.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all", action="store_true", dest="all_sessions", help="撤销全部账户的会话")
    selector.add_argument("--username", help="仅撤销指定账户的会话")
    revoke.add_argument("--database-url", default=None,
                        help="精确数据库 URL；未提供时使用 VLA_DATABASE_URL")
    args = parser.parse_args(argv)
    database_url = args.database_url or os.getenv("VLA_DATABASE_URL")
    if not database_url:
        parser.error("请提供 --database-url 或设置 VLA_DATABASE_URL，避免修改错误的数据库")
    db = Database(database_url)
    try:
        db.initialize()
        if args.command == "revoke-sessions":
            count = revoke_sessions(db, None if args.all_sessions else args.username)
            target = "全部账户" if args.all_sessions else args.username.strip().lower()
            print(f"已撤销会话: {count}（{target}）")
            return
        password = getpass.getpass("新密码（至少 12 位）: ")
        validate_password(password)
        confirmation = getpass.getpass("再次输入新密码: ")
        if password != confirmation:
            parser.error("两次输入的密码不一致")
        manager = AuthManager(db, Settings(database_url=database_url))
        if args.command == "create-admin":
            user = manager.create_user(args.username, password, "admin", args.display_name)
            print(f"管理员已创建: {user['username']}（{user['id']}）")
        else:
            with db.read() as session:
                user = session.scalar(select(User).where(User.username == args.username.strip().lower()))
            if not user:
                raise APIError(404, "NOT_FOUND", "账户不存在")
            manager.reset_password(user.id, password, force_change=False)
            print(f"密码已重置，原有会话已撤销: {user.username}")
    except APIError as exc:
        parser.exit(2, f"{exc.code}: {exc.message}\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
