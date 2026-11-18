"""AI assistants: the owner's settings (/api/ai-access) and the OAuth endpoints cloud assistants sign in with."""
import json
from urllib.parse import parse_qsl

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .. import aiaccess

router = APIRouter()
_NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}
_PAGE_HEADERS = {**_NO_STORE, "X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'; form-action 'self' https: http://localhost:* http://127.0.0.1:*",
                 "Referrer-Policy": "same-origin"}  # "no-referrer" makes browsers send Origin: null on the form post


# ---------- Owner settings (this device only; not reachable through MCP) ----------

@router.get("/api/ai-access")
def ai_access_status():
    return aiaccess.status()


@router.patch("/api/ai-access")
def ai_access_update(body: dict):
    try:
        aiaccess.update_config(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return aiaccess.status()


@router.post("/api/ai-access/pairing-code")
def ai_access_new_code():
    aiaccess.new_pairing_code()
    return aiaccess.status()


@router.delete("/api/ai-access/connections/{grant_id}")
def ai_access_revoke(grant_id: int):
    if not aiaccess.revoke(grant_id):
        raise HTTPException(404, "No such connection")
    return aiaccess.status()


# ---------- OAuth 2.1 for cloud assistants ----------

def _request_base(request: Request) -> str:
    return f"{request.url.scheme}://{request.headers.get('host', 'localhost:8000')}"


async def _form(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > 65536:
        raise aiaccess.OAuthError("invalid_request", "Request too large")
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            raise aiaccess.OAuthError("invalid_request", "Invalid JSON")
        return data if isinstance(data, dict) else {}
    return dict(parse_qsl(raw.decode("utf-8", "replace"), keep_blank_values=True))


def _oauth_error(e: aiaccess.OAuthError) -> JSONResponse:
    return JSONResponse(e.body(), status_code=e.status, headers=_NO_STORE)


@router.get("/.well-known/oauth-protected-resource", include_in_schema=False)
@router.get("/.well-known/oauth-protected-resource/mcp", include_in_schema=False)
def protected_resource(request: Request):
    return aiaccess.resource_metadata(_request_base(request))


@router.get("/.well-known/oauth-authorization-server", include_in_schema=False)
@router.get("/.well-known/oauth-authorization-server/mcp", include_in_schema=False)
@router.get("/.well-known/openid-configuration", include_in_schema=False)
def authorization_server(request: Request):
    return aiaccess.server_metadata(_request_base(request))


@router.post("/oauth/register", include_in_schema=False)
async def oauth_register(request: Request):
    try:
        if not aiaccess.public_active():
            raise aiaccess.OAuthError("access_denied", "Public access for AI assistants is turned off in Blocky's Settings.", 403)
        return JSONResponse(aiaccess.register_client(await _form(request)), status_code=201, headers=_NO_STORE)
    except aiaccess.OAuthError as e:
        return _oauth_error(e)


@router.get("/oauth/authorize", include_in_schema=False)
def oauth_authorize_page(request: Request):
    if not aiaccess.public_active():
        return HTMLResponse(aiaccess.page(None, "Public access for AI assistants is turned off in Blocky's Settings."), 403, headers=_PAGE_HEADERS)
    try:
        req = aiaccess.check_authorize_request(dict(request.query_params))
    except aiaccess.OAuthError as e:
        return HTMLResponse(aiaccess.page(None, e.description), 400, headers=_PAGE_HEADERS)
    return HTMLResponse(aiaccess.page(req), headers=_PAGE_HEADERS)


@router.post("/oauth/authorize", include_in_schema=False)
async def oauth_authorize_submit(request: Request):
    try:
        form = await _form(request)
        req = aiaccess.check_authorize_request(form)
    except aiaccess.OAuthError as e:
        return HTMLResponse(aiaccess.page(None, e.description), 400, headers=_PAGE_HEADERS)
    if form.get("decision") == "deny":
        return RedirectResponse(aiaccess.deny(req), 303, headers=_NO_STORE)
    try:
        return RedirectResponse(aiaccess.approve(req, form.get("pairing_code", ""), form.get("level", "")), 303, headers=_NO_STORE)
    except aiaccess.OAuthError as e:
        return HTMLResponse(aiaccess.page(req, e.description), e.status, headers=_PAGE_HEADERS)


@router.post("/oauth/token", include_in_schema=False)
async def oauth_token(request: Request):
    try:
        return JSONResponse(aiaccess.token(await _form(request)), headers=_NO_STORE)
    except aiaccess.OAuthError as e:
        return _oauth_error(e)


@router.post("/oauth/revoke", include_in_schema=False)
async def oauth_revoke(request: Request):
    try:
        aiaccess.revoke_token((await _form(request)).get("token", ""))
    except aiaccess.OAuthError as e:
        return _oauth_error(e)
    return JSONResponse({}, headers=_NO_STORE)
