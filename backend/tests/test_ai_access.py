"""AI assistants over MCP: access levels, the general API tools, read-only SQL, and the OAuth sign-in cloud assistants use.

Local requests come from 127.0.0.1 with Host: localhost. Cloud requests are simulated the way a tunnel delivers them:
from loopback, but with the public Host header (and usually forwarding headers).
"""
import base64
import hashlib
import json
import secrets
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from app import aiaccess, db, mcp_server, remote

PUBLIC = "https://blocky.example.com"
CLOUD = {"host": "blocky.example.com", "x-forwarded-for": "203.0.113.9"}
REDIRECT = "https://chatgpt.com/connector_platform_oauth_redirect"


@pytest.fixture
def api(http_client, fresh_db):
    aiaccess._failed.clear()
    return http_client


def rpc(client, method, params=None, headers=None):
    h = {"accept": "application/json, text/event-stream", "content-type": "application/json", **(headers or {})}
    return client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}, headers=h)


def call(client, name, args=None, headers=None) -> dict:
    r = rpc(client, "tools/call", {"name": name, "arguments": args or {}}, headers)
    assert r.status_code == 200, r.text
    return json.loads(r.json()["result"]["content"][0]["text"])


def go_public(client) -> str:
    r = client.patch("/api/ai-access", json={"public_url": PUBLIC + "/", "public_enabled": True})
    assert r.status_code == 200, r.text
    assert r.json()["public_url"] == PUBLIC and r.json()["public_mcp_url"] == PUBLIC + "/mcp"
    return r.json()["pairing_code"]


def pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def sign_in(client, level="read", code=None) -> dict:
    """The whole OAuth flow a cloud assistant goes through. Returns the token response."""
    code = code or client.get("/api/ai-access").json()["pairing_code"]
    reg = client.post("/oauth/register", json={"redirect_uris": [REDIRECT], "client_name": "ChatGPT"}, headers=CLOUD)
    assert reg.status_code == 201, reg.text
    cid = reg.json()["client_id"]
    verifier, challenge = pkce()
    q = {"client_id": cid, "redirect_uri": REDIRECT, "response_type": "code", "code_challenge": challenge,
         "code_challenge_method": "S256", "state": "xyz", "resource": PUBLIC + "/mcp"}
    page = client.get("/oauth/authorize?" + urlencode(q), headers=CLOUD)
    assert page.status_code == 200 and "ChatGPT" in page.text and "chatgpt.com" in page.text
    assert page.headers["x-frame-options"] == "DENY"
    # With "no-referrer", browsers send Origin: null on the approval form, which the cross-site check refuses.
    assert page.headers["referrer-policy"] == "same-origin"
    ok = client.post("/oauth/authorize", data={**q, "pairing_code": code.lower().replace("-", ""), "level": level, "decision": "approve"},
                     headers={**CLOUD, "origin": PUBLIC}, follow_redirects=False)
    assert ok.status_code == 303, ok.text
    loc = urlsplit(ok.headers["location"])
    params = parse_qs(loc.query)
    assert f"{loc.scheme}://{loc.netloc}{loc.path}" == REDIRECT and params["state"] == ["xyz"] and params["iss"] == [PUBLIC]
    tok = client.post("/oauth/token", data={"grant_type": "authorization_code", "code": params["code"][0], "client_id": cid,
                                            "redirect_uri": REDIRECT, "code_verifier": verifier}, headers=CLOUD)
    assert tok.status_code == 200, tok.text
    return {**tok.json(), "client_id": cid, "auth_code": params["code"][0], "verifier": verifier}


# ---------- This device ----------

def test_local_assistant_gets_the_local_level(api):
    names = {t["name"] for t in rpc(api, "tools/list").json()["result"]["tools"]}
    assert {"get_status", "list_api", "api_get", "api_call", "api_delete", "query_database"} <= names
    assert call(api, "api_get", {"path": "/api/settings"})["status"] == 200

    api.patch("/api/ai-access", json={"local_level": "read"})
    denied = call(api, "api_delete", {"path": "/api/experiments/1"})
    assert "needs 'full'" in denied["error"] and "given 'read'" in denied["error"]
    assert "needs 'write'" in call(api, "create_experiment", {"model": "x.gguf"})["error"]
    assert call(api, "api_get", {"path": "/api/settings"})["status"] == 200

    api.patch("/api/ai-access", json={"local_level": "write"})
    assert "needs 'full'" in call(api, "api_delete", {"path": "/api/experiments/1"})["error"]
    assert api.patch("/api/ai-access", json={"local_level": "admin"}).status_code == 400


def test_full_level_reaches_every_endpoint(api, make_gguf):
    make_gguf("assist.gguf")
    listing = call(api, "list_api", {"filter": "experiments"})
    assert listing["count"] > 5 and any(e["path"] == "/api/experiments" and e["method"] == "POST" for e in listing["endpoints"])
    assert not any(e["path"].startswith(("/api/ai-access", "/api/remote-access", "/api/agent/key")) for e in call(api, "list_api")["endpoints"])

    created = call(api, "api_call", {"method": "POST", "path": "/api/experiments", "body": {"name": "Via MCP", "model": "assist.gguf"}})
    assert created["status"] == 200
    eid = json.loads(created["body"])["id"]
    assert json.loads(call(api, "api_get", {"path": f"/api/experiments/{eid}"})["body"])["name"] == "Via MCP"
    assert call(api, "api_delete", {"path": f"/api/experiments/{eid}"})["status"] == 200
    assert call(api, "api_get", {"path": f"/api/experiments/{eid}"})["status"] == 404
    assert "method must be" in call(api, "api_call", {"method": "DELETE", "path": "/api/experiments/1"})["error"]


def test_secrets_and_access_endpoints_stay_out_of_reach(api):
    token = remote.set_access(True)
    db.update_settings({"anthropic_api_key": "sk-ant-secret"})
    settings = call(api, "api_get", {"path": "/api/settings"})["body"]
    assert token not in settings and "sk-ant-secret" not in settings and '"remote_token": "[hidden]"' in settings
    for path in ("/api/ai-access", "/api/remote-access", "/api/agent/key", "/mcp", "/oauth/token", "/", "/docs", "/../etc/passwd",
                 "//evil.example/api/x", "http://evil.example/api/x"):
        assert "error" in call(api, "api_get", {"path": path}), path
    assert "error" in call(api, "api_call", {"method": "POST", "path": "/api/remote-access", "body": {"enabled": True}})
    assert remote.access_token() == token  # unchanged


def test_query_database_reads_everything_but_credentials(api):
    with db.connect() as c:
        c.executescript(remote.SCHEMA)
    remote.add_device({"name": "box", "url": "http://192.0.2.5:8000", "token": "device-secret"})
    schema = call(api, "query_database")["tables"]
    assert "exp_runs" in schema and "devices" in schema and "settings" not in schema and "ai_tokens" not in schema
    rows = call(api, "query_database", {"sql": "SELECT name, token FROM devices"})
    assert rows["rows"] == [["box", None]]
    assert "Not readable" in call(api, "query_database", {"sql": "SELECT * FROM settings"})["error"]
    assert "Not readable" in call(api, "query_database", {"sql": "SELECT * FROM ai_grants"})["error"]
    assert "error" in call(api, "query_database", {"sql": "DELETE FROM devices"})
    assert "error" in call(api, "query_database", {"sql": "ATTACH DATABASE '/tmp/x.db' AS x"})
    assert remote.list_devices()[0]["name"] == "box"
    many = call(api, "query_database", {"sql": "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 900) SELECT i FROM n", "max_rows": 10})
    assert len(many["rows"]) == 10 and many["more_rows"]


def test_cross_site_and_forwarded_requests_are_refused(api):
    evil = rpc(api, "tools/list", headers={"origin": "https://evil.example"})
    assert evil.status_code == 403
    # Through a tunnel on this machine with Host rewritten to localhost: still someone else's request.
    assert api.get("/api/settings", headers={"x-forwarded-for": "203.0.113.9"}).status_code == 403
    assert rpc(api, "tools/list", headers={"cf-connecting-ip": "203.0.113.9"}).status_code == 403


# ---------- Cloud assistants (ChatGPT, Claude) through the owner's tunnel ----------

def test_public_access_is_off_by_default(api):
    assert api.get("/.well-known/oauth-protected-resource/mcp", headers=CLOUD).status_code == 403
    assert rpc(api, "tools/list", headers=CLOUD).status_code == 403
    assert api.patch("/api/ai-access", json={"public_enabled": True}).status_code == 400  # needs an address
    assert api.patch("/api/ai-access", json={"public_url": "http://blocky.example.com"}).status_code == 400  # https only
    assert api.patch("/api/ai-access", json={"public_url": "https://x.example.com/path"}).status_code == 400


def test_oauth_sign_in_and_read_level(api):
    code = go_public(api)
    assert "pairing_code" not in api.get("/api/settings").json()

    unauth = rpc(api, "tools/list", headers=CLOUD)
    assert unauth.status_code == 401
    assert f'resource_metadata="{PUBLIC}/.well-known/oauth-protected-resource/mcp"' in unauth.headers["www-authenticate"]
    meta = api.get("/.well-known/oauth-protected-resource/mcp", headers=CLOUD).json()
    assert meta["resource"] == PUBLIC + "/mcp" and meta["authorization_servers"] == [PUBLIC]
    asm = api.get("/.well-known/oauth-authorization-server", headers=CLOUD).json()
    assert asm["issuer"] == PUBLIC and asm["code_challenge_methods_supported"] == ["S256"]
    assert asm["registration_endpoint"] == PUBLIC + "/oauth/register"

    # Only MCP and sign-in are reachable from outside; the rest of Blocky isn't.
    assert api.get("/api/settings", headers=CLOUD).status_code == 403
    assert api.get("/api/ai-access", headers=CLOUD).status_code == 403

    tok = sign_in(api, "read", code)
    assert api.get("/api/ai-access").json()["pairing_code"] != code  # one use per code
    bearer = {**CLOUD, "authorization": f"Bearer {tok['access_token']}"}
    assert "hardware" in call(api, "get_status", headers=bearer)
    assert call(api, "api_get", {"path": "/api/settings"}, bearer)["status"] == 200
    assert "needs 'write'" in call(api, "api_call", {"method": "POST", "path": "/api/experiments", "body": {}}, bearer)["error"]
    assert "needs 'full'" in call(api, "api_delete", {"path": "/api/experiments/1"}, bearer)["error"]
    assert rpc(api, "tools/list", headers={**CLOUD, "authorization": "Bearer nope"}).status_code == 401

    conns = api.get("/api/ai-access").json()["connections"]
    assert len(conns) == 1 and conns[0]["client_name"] == "ChatGPT" and conns[0]["level"] == "read" and conns[0]["last_used_at"]
    assert conns[0]["redirect"] == "chatgpt.com"


def test_full_level_and_revocation(api, make_gguf):
    go_public(api)
    make_gguf("cloud.gguf")
    tok = sign_in(api, "full")
    bearer = {**CLOUD, "authorization": f"Bearer {tok['access_token']}"}
    created = call(api, "api_call", {"method": "POST", "path": "/api/experiments", "body": {"name": "Cloud", "model": "cloud.gguf"}}, bearer)
    eid = json.loads(created["body"])["id"]
    assert call(api, "api_delete", {"path": f"/api/experiments/{eid}"}, bearer)["status"] == 200

    gid = api.get("/api/ai-access").json()["connections"][0]["id"]
    assert api.delete(f"/api/ai-access/connections/{gid}").status_code == 200
    assert rpc(api, "tools/list", headers=bearer).status_code == 401
    refreshed = api.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"],
                                               "client_id": tok["client_id"]}, headers=CLOUD)
    assert refreshed.status_code == 400 and refreshed.json()["error"] == "invalid_grant"


def test_refresh_rotation_and_reuse_detection(api):
    go_public(api)
    tok = sign_in(api, "write")
    form = {"grant_type": "refresh_token", "refresh_token": tok["refresh_token"], "client_id": tok["client_id"]}
    new = api.post("/oauth/token", data=form, headers=CLOUD)
    assert new.status_code == 200 and new.json()["refresh_token"] != tok["refresh_token"]
    assert rpc(api, "tools/list", headers={**CLOUD, "authorization": f"Bearer {new.json()['access_token']}"}).status_code == 200
    # Reusing the old refresh token means it leaked: the whole connection is cancelled.
    assert api.post("/oauth/token", data=form, headers=CLOUD).status_code == 400
    assert rpc(api, "tools/list", headers={**CLOUD, "authorization": f"Bearer {new.json()['access_token']}"}).status_code == 401


def test_authorization_code_checks(api):
    go_public(api)
    tok = sign_in(api, "read")
    # The code is single-use; reusing it cancels what it produced.
    again = api.post("/oauth/token", data={"grant_type": "authorization_code", "code": tok["auth_code"], "client_id": tok["client_id"],
                                           "redirect_uri": REDIRECT, "code_verifier": tok["verifier"]}, headers=CLOUD)
    assert again.status_code == 400
    assert rpc(api, "tools/list", headers={**CLOUD, "authorization": f"Bearer {tok['access_token']}"}).status_code == 401

    reg = api.post("/oauth/register", json={"redirect_uris": [REDIRECT]}, headers=CLOUD).json()
    verifier, challenge = pkce()
    q = {"client_id": reg["client_id"], "redirect_uri": REDIRECT, "response_type": "code", "code_challenge": challenge,
         "code_challenge_method": "S256"}
    code = api.get("/api/ai-access").json()["pairing_code"]
    ok = api.post("/oauth/authorize", data={**q, "pairing_code": code, "level": "read", "decision": "approve"}, headers=CLOUD,
                  follow_redirects=False)
    auth_code = parse_qs(urlsplit(ok.headers["location"]).query)["code"][0]
    wrong = api.post("/oauth/token", data={"grant_type": "authorization_code", "code": auth_code, "client_id": reg["client_id"],
                                           "redirect_uri": REDIRECT, "code_verifier": pkce()[0]}, headers=CLOUD)
    assert wrong.status_code == 400 and "PKCE" in wrong.json()["error_description"]


def test_authorize_rejects_bad_requests_and_wrong_codes(api):
    go_public(api)
    assert api.post("/oauth/register", json={"redirect_uris": ["http://evil.example/cb"]}, headers=CLOUD).status_code == 400
    assert api.post("/oauth/register", json={"redirect_uris": []}, headers=CLOUD).status_code == 400
    cid = api.post("/oauth/register", json={"redirect_uris": [REDIRECT]}, headers=CLOUD).json()["client_id"]
    _, challenge = pkce()
    base = {"client_id": cid, "redirect_uri": REDIRECT, "response_type": "code", "code_challenge": challenge, "code_challenge_method": "S256"}
    assert api.get("/oauth/authorize?" + urlencode({**base, "redirect_uri": "https://evil.example/cb"}), headers=CLOUD).status_code == 400
    assert api.get("/oauth/authorize?" + urlencode({**base, "code_challenge_method": "plain"}), headers=CLOUD).status_code == 400
    assert api.get("/oauth/authorize?" + urlencode({**base, "client_id": "unknown"}), headers=CLOUD).status_code == 400

    denied = api.post("/oauth/authorize", data={**base, "decision": "deny"}, headers=CLOUD, follow_redirects=False)
    assert denied.status_code == 303 and "error=access_denied" in denied.headers["location"]

    for _ in range(aiaccess.MAX_FAILED):
        bad = api.post("/oauth/authorize", data={**base, "pairing_code": "AAAAA-AAAAA", "level": "full", "decision": "approve"}, headers=CLOUD)
        assert bad.status_code == 403 and "Wrong pairing code" in bad.text
    code = api.get("/api/ai-access").json()["pairing_code"]
    locked = api.post("/oauth/authorize", data={**base, "pairing_code": code, "level": "full", "decision": "approve"}, headers=CLOUD)
    assert locked.status_code == 429  # even the right code, until the owner makes a new one or the lock expires
    api.post("/api/ai-access/pairing-code")
    code = api.get("/api/ai-access").json()["pairing_code"]
    assert api.post("/oauth/authorize", data={**base, "pairing_code": code, "level": "full", "decision": "approve"}, headers=CLOUD,
                    follow_redirects=False).status_code == 303


def test_turning_public_access_off_closes_everything(api):
    go_public(api)
    tok = sign_in(api, "full")
    api.patch("/api/ai-access", json={"public_enabled": False})
    assert rpc(api, "tools/list", headers={**CLOUD, "authorization": f"Bearer {tok['access_token']}"}).status_code == 403
    assert aiaccess.verify(tok["access_token"]) is None
    # Another public name than the configured one never gets in.
    api.patch("/api/ai-access", json={"public_enabled": True})
    assert rpc(api, "tools/list", headers={"host": "other.example.com", "authorization": f"Bearer {tok['access_token']}"}).status_code == 403


def test_level_context_defaults_to_full_in_process():
    assert mcp_server.LEVEL.get() == "full" and mcp_server._denied("full") is None


def test_token_request_with_basic_client_auth(api):
    """Some clients send client_id in an HTTP Basic header (empty secret) instead of the form."""
    go_public(api)
    cid = api.post("/oauth/register", json={"redirect_uris": [REDIRECT]}, headers=CLOUD).json()["client_id"]
    verifier, challenge = pkce()
    q = {"client_id": cid, "redirect_uri": REDIRECT, "response_type": "code", "code_challenge": challenge, "code_challenge_method": "S256"}
    code = api.get("/api/ai-access").json()["pairing_code"]
    ok = api.post("/oauth/authorize", data={**q, "pairing_code": code, "level": "read", "decision": "approve"}, headers=CLOUD,
                  follow_redirects=False)
    auth_code = parse_qs(urlsplit(ok.headers["location"]).query)["code"][0]
    basic = base64.b64encode(f"{cid}:".encode()).decode()
    tok = api.post("/oauth/token", data={"grant_type": "authorization_code", "code": auth_code, "redirect_uri": REDIRECT,
                                         "code_verifier": verifier}, headers={**CLOUD, "authorization": f"Basic {basic}"})
    assert tok.status_code == 200, tok.text
    assert api.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": tok.json()["refresh_token"]},
                    headers={**CLOUD, "authorization": "Basic !!!"}).status_code == 401
