"""
Authentication module for Scanner API.
JWT-based authentication with API key support.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Configuration
_default_secret = secrets.token_urlsafe(32)
SECRET_KEY = os.getenv("SCANNER_SECRET_KEY", _default_secret)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# API Key for simple authentication (alternative to JWT)
API_KEY = os.getenv("SCANNER_API_KEY", secrets.token_urlsafe(24))

# Production safety checks
if SECRET_KEY == _default_secret:
    import warnings
    warnings.warn(
        "SCANNER_SECRET_KEY not set — using a random ephemeral key. "
        "Tokens will not survive restarts. Set SCANNER_SECRET_KEY in your environment.",
        stacklevel=2,
    )
if not os.getenv("SCANNER_API_KEY"):
    import warnings
    warnings.warn(
        "SCANNER_API_KEY not set — a random API key was generated. "
        "Set SCANNER_API_KEY in your environment for stable access.",
        stacklevel=2,
    )

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list = []


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: list = []


class UserInDB(User):
    hashed_password: str


# User database — configure via environment variables for production.
# Set SCANNER_ADMIN_PASSWORD (required) to enable the admin account.
# Falls back to demo accounts ONLY when SCANNER_ENV != "production".
def _build_users_db() -> dict:
    """Build user database from environment or demo defaults."""
    db = {}
    admin_pw = os.getenv("SCANNER_ADMIN_PASSWORD")
    analyst_pw = os.getenv("SCANNER_ANALYST_PASSWORD")
    viewer_pw = os.getenv("SCANNER_VIEWER_PASSWORD")
    is_production = os.getenv("SCANNER_ENV", "").lower() == "production"

    if admin_pw:
        db["admin"] = {
            "username": "admin",
            "email": os.getenv("SCANNER_ADMIN_EMAIL", "admin@scanner.ai"),
            "full_name": "Scanner Admin",
            "disabled": False,
            "hashed_password": pwd_context.hash(admin_pw),
            "scopes": ["read", "write", "admin"],
        }
    if analyst_pw:
        db["analyst"] = {
            "username": "analyst",
            "email": os.getenv("SCANNER_ANALYST_EMAIL", "analyst@scanner.ai"),
            "full_name": "Security Analyst",
            "disabled": False,
            "hashed_password": pwd_context.hash(analyst_pw),
            "scopes": ["read", "write"],
        }
    if viewer_pw:
        db["viewer"] = {
            "username": "viewer",
            "email": os.getenv("SCANNER_VIEWER_EMAIL", "viewer@scanner.ai"),
            "full_name": "Read-Only User",
            "disabled": False,
            "hashed_password": pwd_context.hash(viewer_pw),
            "scopes": ["read"],
        }

    if not db and not is_production:
        # Development-only demo accounts
        db = {
            "admin": {
                "username": "admin",
                "email": "admin@scanner.ai",
                "full_name": "Scanner Admin",
                "disabled": False,
                "hashed_password": pwd_context.hash("scanner2026"),
                "scopes": ["read", "write", "admin"],
            },
            "analyst": {
                "username": "analyst",
                "email": "analyst@scanner.ai",
                "full_name": "Security Analyst",
                "disabled": False,
                "hashed_password": pwd_context.hash("analyst2026"),
                "scopes": ["read", "write"],
            },
            "viewer": {
                "username": "viewer",
                "email": "viewer@scanner.ai",
                "full_name": "Read-Only User",
                "disabled": False,
                "hashed_password": pwd_context.hash("viewer2026"),
                "scopes": ["read"],
            },
        }
        import warnings
        warnings.warn(
            "Using demo credentials. Set SCANNER_ADMIN_PASSWORD for production.",
            stacklevel=2,
        )
    elif not db and is_production:
        raise RuntimeError(
            "SCANNER_ENV=production but no user passwords configured. "
            "Set SCANNER_ADMIN_PASSWORD to create the admin account."
        )

    return db


DEMO_USERS_DB = _build_users_db()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in DEMO_USERS_DB:
        user_dict = DEMO_USERS_DB[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        scopes: list = payload.get("scopes", [])

        if username is None:
            return None

        return TokenData(username=username, scopes=scopes)
    except JWTError:
        return None


async def get_current_user(
    bearer_credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Security(api_key_header)
) -> User:
    """
    Get the current authenticated user.
    Supports both JWT Bearer token and API Key authentication.
    """
    # Try API Key first (simpler)
    if api_key:
        if api_key == API_KEY:
            # Return a default API key user
            return User(
                username="api_user",
                email="api@scanner.ai",
                full_name="API Key User",
                disabled=False,
                scopes=["read", "write"]
            )
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "API-Key"}
            )

    # Try JWT Bearer token
    if bearer_credentials:
        token = bearer_credentials.credentials
        token_data = decode_token(token)

        if token_data is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        user = get_user(token_data.username)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"}
            )

        if user.disabled:
            raise HTTPException(status_code=403, detail="User account disabled")

        return User(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            disabled=user.disabled,
            scopes=user.scopes
        )

    # No authentication provided
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"}
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user and verify they are active."""
    if current_user.disabled:
        raise HTTPException(status_code=403, detail="User account disabled")
    return current_user


def require_scope(required_scope: str):
    """Dependency to require a specific scope."""
    async def scope_checker(current_user: User = Depends(get_current_active_user)):
        if required_scope not in current_user.scopes and "admin" not in current_user.scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required scope: {required_scope}"
            )
        return current_user
    return scope_checker


# Convenience dependencies
require_read = require_scope("read")
require_write = require_scope("write")
require_admin = require_scope("admin")
