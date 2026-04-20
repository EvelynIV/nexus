from __future__ import annotations

import json
from typing import Any, Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile

from nexus.application.container import AppContainer, get_container

from .runtime import (
    DEFAULT_REALTIME_MODEL,
    get_realtime_api_runtime,
    normalize_session_config,
)


def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Authorization header must use Bearer.")
    return token


def parse_json_form_value(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, UploadFile):
        raw = value.file.read()
        return json.loads(raw.decode("utf-8"))
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    raise HTTPException(status_code=400, detail="Unsupported JSON form field type.")


async def parse_sdp_form_value(value: Any) -> str:
    if value is None:
        raise HTTPException(status_code=400, detail="Missing sdp form field.")
    if isinstance(value, UploadFile):
        raw = await value.read()
        return raw.decode("utf-8")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise HTTPException(status_code=400, detail="Unsupported sdp form field type.")


async def create_client_secret_endpoint(
    request: Request,
    container: Annotated[AppContainer, Depends(get_container)],
):
    runtime = await get_realtime_api_runtime(container)
    bearer_token = extract_bearer_token(request.headers.get("authorization"))
    if runtime.api_key_required() and not runtime.check_api_key(bearer_token):
        raise HTTPException(status_code=401, detail="Invalid realtime API key.")

    body = await request.json()
    expires_after = body.get("expires_after") or {}
    if expires_after and expires_after.get("anchor", "created_at") != "created_at":
        raise HTTPException(status_code=400, detail="expires_after.anchor must be 'created_at'.")

    ttl_seconds = expires_after.get("seconds") or container.config.realtime_client_secret_ttl_seconds
    session = normalize_session_config(
        body.get("session"),
        default_model=DEFAULT_REALTIME_MODEL,
    )
    record = await runtime.client_secrets.create(session=session, ttl_seconds=int(ttl_seconds))
    return JSONResponse(
        {
            "value": record.value,
            "expires_at": record.expires_at,
            "session": record.session,
        }
    )


async def create_realtime_call_endpoint(
    request: Request,
    container: Annotated[AppContainer, Depends(get_container)],
):
    runtime = await get_realtime_api_runtime(container)
    bearer_token = extract_bearer_token(request.headers.get("authorization"))
    content_type = request.headers.get("content-type", "")

    session_config: dict[str, Any] | None = None
    if content_type.startswith("application/sdp"):
        sdp_offer = (await request.body()).decode("utf-8")
        if not sdp_offer.strip():
            raise HTTPException(status_code=400, detail="Missing SDP offer body.")

        if bearer_token and bearer_token.startswith("ek_"):
            secret = await runtime.client_secrets.get(bearer_token)
            if secret is None:
                raise HTTPException(status_code=401, detail="Invalid or expired client secret.")
            session_config = secret.session
        elif runtime.api_key_required():
            raise HTTPException(
                status_code=401,
                detail="Application/sdp call creation requires a valid client secret.",
            )
        elif bearer_token and not runtime.check_api_key(bearer_token):
            raise HTTPException(status_code=401, detail="Unknown bearer token.")
        if session_config is None:
            session_config = normalize_session_config(None, default_model=DEFAULT_REALTIME_MODEL)
    elif content_type.startswith("multipart/form-data"):
        if runtime.api_key_required() and not runtime.check_api_key(bearer_token):
            raise HTTPException(status_code=401, detail="Invalid realtime API key.")
        form = await request.form()
        sdp_offer = await parse_sdp_form_value(form.get("sdp"))
        session_config = normalize_session_config(
            parse_json_form_value(form.get("session")),
            default_model=DEFAULT_REALTIME_MODEL,
        )
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Content-Type. Use application/sdp or multipart/form-data.",
        )

    call = await runtime.calls.create_call(
        sdp_offer=sdp_offer,
        session_config=session_config,
    )
    return Response(
        content=call.peer_connection.localDescription.sdp,
        status_code=201,
        media_type="application/sdp",
        headers={"Location": f"/v1/realtime/calls/{call.call_id}"},
    )
