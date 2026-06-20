"""JWT-based authentication helpers.

Passwords are hashed with bcrypt.
Tokens are signed HS256 JWTs.

All endpoints that need a user call get_current_user() as a FastAPI Depends.
It returns "anonymous" when no valid token is supplied so existing unauthenticated
flows keep working.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Header, HTTPException

from api.config import get_settings
from api.db import DBUser, SessionLocal

settings = get_settings()

SECRET_KEY: str = settings.jwt_secret
ALGORITHM = "HS256"
TOKEN_HOURS = settings.jwt_expire_hours


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return str(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_current_user(authorization: str | None = Header(default=None)) -> str:
    """Returns user_id from Bearer token, or 'anonymous' if none supplied."""
    if not authorization or not authorization.startswith("Bearer "):
        return "anonymous"
    try:
        return _decode_token(authorization[7:])
    except HTTPException:
        return "anonymous"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def register_user(username: str, email: str, password: str) -> DBUser:
    with SessionLocal() as db:
        conflict = (
            db.query(DBUser)
            .filter((DBUser.username == username) | (DBUser.email == email))
            .first()
        )
        if conflict:
            raise HTTPException(status_code=409, detail="Username or email already taken.")
        user = DBUser(username=username, email=email, password_hash=hash_password(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


def authenticate_user(username_or_email: str, password: str) -> DBUser:
    with SessionLocal() as db:
        user = (
            db.query(DBUser)
            .filter(
                (DBUser.username == username_or_email)
                | (DBUser.email == username_or_email)
            )
            .first()
        )
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    return user


def get_user_by_id(user_id: str) -> DBUser | None:
    with SessionLocal() as db:
        return db.get(DBUser, user_id)
