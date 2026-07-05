"""
Opt-in API key authentication.

Off by default (backward compatible with every test and example in this
project so far): if CNC_API_KEY isn't set in the environment, every
request is allowed through, same as before this module existed. Set it
to turn authentication on - every protected endpoint then requires a
matching X-API-Key header. /health is deliberately never protected, so
load balancers and container orchestrators can check liveness without
needing a credential.

    export CNC_API_KEY="some-long-random-string"
    uvicorn app.main:app

    curl -H "X-API-Key: some-long-random-string" http://127.0.0.1:8000/predict ...

This is a shared-secret scheme, appropriate for a service reached by a
small number of trusted internal systems (an MES, a QC script) - not a
substitute for a real identity provider if this is ever exposed to many
distinct external users who each need their own revocable credential.
"""

import os
import secrets

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_ENV_VAR = "CNC_API_KEY"

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def auth_enabled() -> bool:
    return bool(os.environ.get(API_KEY_ENV_VAR))


def require_api_key(provided_key: str = Security(_api_key_header)) -> None:
    expected_key = os.environ.get(API_KEY_ENV_VAR)

    if not expected_key:
        return  # auth not configured for this deployment - allow through

    if not provided_key or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )
