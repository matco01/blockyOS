"""AI assistants in the cloud (ChatGPT, Claude on the web, desktop and phone) connecting to this Blocky's MCP server.

Assistants on this device (Claude Code, a local Claude Desktop config) reach http://localhost:8000/mcp directly and get
the local access level. Cloud assistants can't reach localhost: the owner publishes Blocky's address through an HTTPS
tunnel (Cloudflare Tunnel, Tailscale Funnel, ...), enters that address here, and each assistant then signs in with
OAuth 2.1 (dynamic client registration, authorization code + PKCE, rotating refresh tokens).

The owner approves every connection on Blocky's own approval page by entering the one-time pairing code shown in
Settings and choosing what that assistant may do:
  read   look at everything (experiments, runs, models, jobs, logs, settings without secrets, the database)
  write  also start, queue, build, change settings and run code; nothing is deleted
  full   also delete
Connections can be revoked at any time; every token stops working immediately.

Through the tunnel only the MCP endpoint and the OAuth endpoints are reachable, and only while public access is on.
Everything else still needs the device itself (or the remote-access token). Tokens and codes are stored as SHA-256
hashes; the pairing code changes after every successful approval and locks after repeated wrong entries.
"""
import base64
import hashlib
import hmac
import html
import json
import secrets
import time
from urllib.parse import urlencode, urlsplit

from . import db

LEVELS = ("read", "write", "full")
LEVEL_TEXT = {
    "read": "See everything: experiments, results, models, jobs, logs and settings (never passwords or tokens).",
    "write": "See everything, and start, queue, build and change things, including running code on this machine. Nothing is deleted.",
    "full": "Everything, including deleting experiments, runs, models and other data.",
}
ACCESS_TTL = 3600
REFRESH_TTL = 30 * 86400
CODE_TTL = 300
MAX_FAILED = 5
LOCK_S = 600
MAX_CLIENTS = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_clients (
    client_id     TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    name          TEXT NOT NULL,
    redirect_uris TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ai_grants (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   REAL NOT NULL,
    client_id    TEXT NOT NULL,
    client_name  TEXT NOT NULL,
    redirect     TEXT NOT NULL,
    level        TEXT NOT NULL,
    last_used_at REAL,
    revoked      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ai_codes (
    code_hash    TEXT PRIMARY KEY,
    grant_id     INTEGER NOT NULL,
    redirect_uri TEXT NOT NULL,
    challenge    TEXT NOT NULL,
    expires_at   REAL NOT NULL,
    used         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ai_tokens (
    token_hash TEXT PRIMARY KEY,
    grant_id   INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    expires_at REAL NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0
);
"""
SECRET_TABLES = frozenset({"settings", "ai_clients", "ai_grants", "ai_codes", "ai_tokens"})
_failed: list[float] = []


def init() -> None:
    with db.connect() as c:
        c.executescript(SCHEMA)


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# ---------- Owner settings ----------

def config() -> dict:
    s = db.get_settings()
    cfg = {"local_level": "full", "public_enabled": False, "public_url": "", **(s.get("ai_access") or {})}
    if cfg["local_level"] not in LEVELS:
        cfg["local_level"] = "full"
    return cfg


def normalize_public_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    u = urlsplit(url)
    local = (u.hostname or "") in ("localhost", "127.0.0.1")
    if u.scheme not in ("https", "http") or (u.scheme == "http" and not local) or not u.hostname:
        raise ValueError("The public address must start with https:// (for example https://blocky.example.com)")
    if u.path or u.query or u.fragment or u.username or u.password:
        raise ValueError("Enter only the address, like https://blocky.example.com, without a path")
    return f"{u.scheme}://{u.netloc.lower()}"


def update_config(body: dict) -> dict:
    cfg = config()
    if "local_level" in body:
        if body["local_level"] not in LEVELS:
            raise ValueError(f"Access level must be one of {', '.join(LEVELS)}")
        cfg["local_level"] = body["local_level"]
    if "public_url" in body:
        cfg["public_url"] = normalize_public_url(body["public_url"])
    if "public_enabled" in body:
        cfg["public_enabled"] = bool(body["public_enabled"])
    if cfg["public_enabled"] and not cfg["public_url"]:
        raise ValueError("Enter the public https address first")
    db.update_settings({"ai_access": cfg})
    if cfg["public_enabled"] and not db.get_settings().get("ai_pairing_code"):
        new_pairing_code()
    return cfg


def new_pairing_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    raw = "".join(secrets.choice(alphabet) for _ in range(10))
    code = f"{raw[:5]}-{raw[5:]}"
    db.update_settings({"ai_pairing_code": code})
    _failed.clear()
    return code


def public_base() -> str:
    return config()["public_url"]


def public_host() -> str:
    base = public_base()
    return urlsplit(base).netloc.lower() if base else ""


def public_active() -> bool:
    cfg = config()
    return bool(cfg["public_enabled"] and cfg["public_url"])


def status() -> dict:
    cfg = config()
    with db.connect() as c:
        grants = [dict(r) for r in c.execute(
            "SELECT id, created_at, client_name, redirect, level, last_used_at FROM ai_grants WHERE revoked = 0 ORDER BY id DESC")]
    base = cfg["public_url"]
    return {**cfg, "levels": {k: LEVEL_TEXT[k] for k in LEVELS}, "pairing_code": db.get_settings().get("ai_pairing_code") or None,
            "local_mcp_url": "http://localhost:8000/mcp", "public_mcp_url": f"{base}/mcp" if base else None,
            "locked": _locked(), "connections": grants}


def revoke(grant_id: int) -> bool:
    with db.connect() as c:
        return c.execute("UPDATE ai_grants SET revoked = 1 WHERE id = ? AND revoked = 0", (grant_id,)).rowcount > 0


# ---------- OAuth metadata ----------

def _base(request_base: str) -> str:
    return public_base() or request_base.rstrip("/")


def resource_metadata(request_base: str) -> dict:
    base = _base(request_base)
    return {"resource": f"{base}/mcp", "authorization_servers": [base], "bearer_methods_supported": ["header"],
            "resource_name": "Blocky research lab"}


def server_metadata(request_base: str) -> dict:
    base = _base(request_base)
    return {"issuer": base, "authorization_endpoint": f"{base}/oauth/authorize", "token_endpoint": f"{base}/oauth/token",
            "registration_endpoint": f"{base}/oauth/register", "revocation_endpoint": f"{base}/oauth/revoke",
            "response_types_supported": ["code"], "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"], "token_endpoint_auth_methods_supported": ["none"],
            "revocation_endpoint_auth_methods_supported": ["none"], "scopes_supported": ["blocky"]}


class OAuthError(Exception):
    def __init__(self, error: str, description: str, status: int = 400):
        super().__init__(description)
        self.error, self.description, self.status = error, description, status

    def body(self) -> dict:
        return {"error": self.error, "error_description": self.description}


# ---------- Dynamic client registration ----------

def _valid_redirect(uri: str) -> bool:
    try:
        u = urlsplit(uri)
    except ValueError:
        return False
    if u.fragment or not u.hostname:
        return False
    return u.scheme == "https" or (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1", "::1"))


def register_client(body: dict) -> dict:
    uris = body.get("redirect_uris")
    if not isinstance(uris, list) or not uris or len(uris) > 10 or not all(isinstance(u, str) and _valid_redirect(u) for u in uris):
        raise OAuthError("invalid_redirect_uri", "redirect_uris must list 1-10 https addresses (http only for localhost)")
    name = str(body.get("client_name") or "").strip()[:80] or urlsplit(uris[0]).hostname
    client_id = "blk_" + secrets.token_urlsafe(18)
    with db.connect() as c:
        if c.execute("SELECT COUNT(*) FROM ai_clients").fetchone()[0] >= MAX_CLIENTS:
            # Registration is open to anyone who can reach the address; drop the oldest never-approved clients.
            c.execute("DELETE FROM ai_clients WHERE client_id IN (SELECT client_id FROM ai_clients WHERE client_id NOT IN "
                      "(SELECT client_id FROM ai_grants) ORDER BY created_at LIMIT 100)")
        c.execute("INSERT INTO ai_clients (client_id, created_at, name, redirect_uris) VALUES (?, ?, ?, ?)",
                  (client_id, time.time(), name, json.dumps(uris)))
    return {"client_id": client_id, "client_id_issued_at": int(time.time()), "client_name": name, "redirect_uris": uris,
            "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"], "token_endpoint_auth_method": "none"}


def client(client_id: str) -> dict | None:
    with db.connect() as c:
        r = c.execute("SELECT * FROM ai_clients WHERE client_id = ?", (client_id or "",)).fetchone()
    return {**dict(r), "redirect_uris": json.loads(r["redirect_uris"])} if r else None


# ---------- Authorization (the owner's approval page) ----------

def check_authorize_request(q: dict) -> dict:
    """Validate the request. Errors before the redirect URI is trusted are shown on the page, never redirected."""
    cl = client(q.get("client_id", ""))
    if not cl:
        raise OAuthError("invalid_client", "Unknown app. Remove the connector in your assistant and add it again.")
    redirect = q.get("redirect_uri") or (cl["redirect_uris"][0] if len(cl["redirect_uris"]) == 1 else "")
    if redirect not in cl["redirect_uris"]:
        raise OAuthError("invalid_request", "The app's return address doesn't match its registration.")
    if q.get("response_type") != "code":
        raise OAuthError("unsupported_response_type", "Only response_type=code is supported.")
    if q.get("code_challenge_method") != "S256" or not 43 <= len(q.get("code_challenge") or "") <= 128:
        raise OAuthError("invalid_request", "PKCE with S256 is required.")
    resource = q.get("resource")
    if resource and public_base() and not resource.rstrip("/").startswith(public_base()):
        raise OAuthError("invalid_target", "This sign-in is for a different server.")
    keep = ("client_id", "redirect_uri", "response_type", "code_challenge", "code_challenge_method", "state", "resource", "scope")
    return {"client": cl, "redirect_uri": redirect, "state": q.get("state"), "challenge": q["code_challenge"],
            "query": {k: str(q[k]) for k in keep if q.get(k) is not None}}


def _locked() -> bool:
    now = time.time()
    _failed[:] = [t for t in _failed if now - t < LOCK_S]
    return len(_failed) >= MAX_FAILED


def approve(req: dict, pairing_code: str, level: str) -> str:
    """Owner approved with the pairing code: returns the redirect URL carrying the authorization code."""
    if not public_active():
        raise OAuthError("access_denied", "Public access for AI assistants is turned off in Blocky's Settings.", 403)
    if _locked():
        raise OAuthError("access_denied", "Too many wrong codes. Wait 10 minutes, or create a new code in Settings.", 429)
    expected = db.get_settings().get("ai_pairing_code") or ""
    given = (pairing_code or "").strip().upper().replace(" ", "")
    if len(given) == 10:
        given = f"{given[:5]}-{given[5:]}"
    if not expected or not hmac.compare_digest(given.encode(), expected.encode()):
        _failed.append(time.time())
        raise OAuthError("access_denied", "Wrong pairing code. It's shown in Blocky under Settings → AI assistants.", 403)
    if level not in LEVELS:
        raise OAuthError("invalid_request", "Choose an access level.")
    new_pairing_code()  # one use per code
    code = secrets.token_urlsafe(32)
    with db.connect() as c:
        gid = c.execute("INSERT INTO ai_grants (created_at, client_id, client_name, redirect, level) VALUES (?, ?, ?, ?, ?)",
                        (time.time(), req["client"]["client_id"], req["client"]["name"], urlsplit(req["redirect_uri"]).netloc,
                         level)).lastrowid
        c.execute("INSERT INTO ai_codes (code_hash, grant_id, redirect_uri, challenge, expires_at) VALUES (?, ?, ?, ?, ?)",
                  (_h(code), gid, req["redirect_uri"], req["challenge"], time.time() + CODE_TTL))
    return _redirect(req, {"code": code})


def deny(req: dict) -> str:
    return _redirect(req, {"error": "access_denied", "error_description": "The owner declined."})


def _redirect(req: dict, params: dict) -> str:
    if req.get("state") is not None:
        params["state"] = req["state"]
    params["iss"] = public_base()
    sep = "&" if "?" in req["redirect_uri"] else "?"
    return req["redirect_uri"] + sep + urlencode(params)


# ---------- Tokens ----------

def _issue(c, grant_id: int, level: str) -> dict:
    access, refresh = "blka_" + secrets.token_urlsafe(32), "blkr_" + secrets.token_urlsafe(32)
    now = time.time()
    c.execute("INSERT INTO ai_tokens (token_hash, grant_id, kind, expires_at) VALUES (?, ?, 'access', ?)", (_h(access), grant_id, now + ACCESS_TTL))
    c.execute("INSERT INTO ai_tokens (token_hash, grant_id, kind, expires_at) VALUES (?, ?, 'refresh', ?)", (_h(refresh), grant_id, now + REFRESH_TTL))
    c.execute("DELETE FROM ai_tokens WHERE expires_at < ?", (now - 86400,))
    return {"access_token": access, "token_type": "Bearer", "expires_in": ACCESS_TTL, "refresh_token": refresh, "scope": f"blocky {level}"}


def token(form: dict) -> dict:
    if not public_active():
        raise OAuthError("access_denied", "Public access for AI assistants is turned off.", 403)
    grant_type = form.get("grant_type")
    cl = client(form.get("client_id", ""))
    if not cl:
        raise OAuthError("invalid_client", "Unknown client", 401)
    with db.connect() as c:
        if grant_type == "authorization_code":
            row = c.execute("SELECT k.*, g.client_id, g.level, g.revoked FROM ai_codes k JOIN ai_grants g ON g.id = k.grant_id "
                            "WHERE k.code_hash = ?", (_h(form.get("code") or ""),)).fetchone()
            if not row or row["client_id"] != cl["client_id"] or row["expires_at"] < time.time() or row["revoked"]:
                raise OAuthError("invalid_grant", "The sign-in code is invalid or expired. Connect again.")
            if row["used"]:
                # A code used twice was intercepted: cancel everything it produced.
                c.execute("UPDATE ai_grants SET revoked = 1 WHERE id = ?", (row["grant_id"],))
                c.commit()  # keep the revocation although the request fails
                raise OAuthError("invalid_grant", "The sign-in code was already used.")
            if (form.get("redirect_uri") or row["redirect_uri"]) != row["redirect_uri"]:
                raise OAuthError("invalid_grant", "redirect_uri doesn't match.")
            verifier = form.get("code_verifier") or ""
            computed = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            if not 43 <= len(verifier) <= 128 or not hmac.compare_digest(computed, row["challenge"]):
                raise OAuthError("invalid_grant", "PKCE verification failed.")
            c.execute("UPDATE ai_codes SET used = 1 WHERE code_hash = ?", (row["code_hash"],))
            return _issue(c, row["grant_id"], row["level"])
        if grant_type == "refresh_token":
            row = c.execute("SELECT t.*, g.client_id, g.level, g.revoked FROM ai_tokens t JOIN ai_grants g ON g.id = t.grant_id "
                            "WHERE t.token_hash = ? AND t.kind = 'refresh'", (_h(form.get("refresh_token") or ""),)).fetchone()
            if not row or row["client_id"] != cl["client_id"] or row["revoked"] or row["expires_at"] < time.time():
                raise OAuthError("invalid_grant", "The connection was revoked or expired. Connect again.")
            if row["used"]:
                c.execute("UPDATE ai_grants SET revoked = 1 WHERE id = ?", (row["grant_id"],))  # refresh-token reuse
                c.commit()  # keep the revocation although the request fails
                raise OAuthError("invalid_grant", "This refresh token was already used.")
            c.execute("UPDATE ai_tokens SET used = 1 WHERE token_hash = ?", (row["token_hash"],))
            return _issue(c, row["grant_id"], row["level"])
    raise OAuthError("unsupported_grant_type", "Use authorization_code or refresh_token.")


def revoke_token(value: str) -> None:
    with db.connect() as c:
        row = c.execute("SELECT grant_id FROM ai_tokens WHERE token_hash = ?", (_h(value or ""),)).fetchone()
        if row:
            c.execute("UPDATE ai_grants SET revoked = 1 WHERE id = ?", (row["grant_id"],))


def verify(bearer: str) -> str | None:
    """Access level of a valid access token, else None."""
    if not bearer or not public_active():
        return None
    with db.connect() as c:
        row = c.execute("SELECT t.grant_id, t.expires_at, g.level, g.revoked FROM ai_tokens t JOIN ai_grants g ON g.id = t.grant_id "
                        "WHERE t.token_hash = ? AND t.kind = 'access'", (_h(bearer),)).fetchone()
        if not row or row["revoked"] or row["expires_at"] < time.time():
            return None
        c.execute("UPDATE ai_grants SET last_used_at = ? WHERE id = ?", (time.time(), row["grant_id"]))
    return row["level"] if row["level"] in LEVELS else None


# ---------- The approval page ----------

def page(req: dict | None, error: str = "", done: str = "") -> str:
    e = html.escape
    body = ""
    if done:
        body = f"<p>{e(done)}</p>"
    elif req is None:
        body = f"<p class='err'>{e(error)}</p>"
    else:
        hidden = "".join(f"<input type='hidden' name='{e(k)}' value='{e(v)}'>" for k, v in req["query"].items())
        options = "".join(
            f"<label class='opt'><input type='radio' name='level' value='{k}'{' checked' if k == 'write' else ''}>"
            f"<span><b>{k.capitalize()}</b><br><small>{e(LEVEL_TEXT[k])}</small></span></label>" for k in LEVELS)
        body = (f"<p><b>{e(req['client']['name'])}</b> wants to use this Blocky.</p>"
                f"<p class='muted'>After you approve, you return to <b>{e(urlsplit(req['redirect_uri']).netloc)}</b>. "
                "Only approve if you just started connecting an assistant yourself.</p>"
                + (f"<p class='err'>{e(error)}</p>" if error else "") +
                f"<form method='post' action='/oauth/authorize'>{hidden}<fieldset><legend>What may it do?</legend>{options}</fieldset>"
                "<label class='code'>Pairing code from Blocky → Settings → AI assistants"
                "<input name='pairing_code' autocomplete='off' autocapitalize='characters' placeholder='XXXXX-XXXXX' required></label>"
                "<div class='buttons'><button name='decision' value='approve'>Approve</button>"
                "<button name='decision' value='deny' class='ghost' formnovalidate>Deny</button></div></form>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Connect to Blocky</title><style>
:root{{--bg:#f5f5f7;--card:#fff;--text:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--accent:#007aff;--err:#d70015}}
@media (prefers-color-scheme:dark){{:root{{--bg:#000;--card:#1c1c1e;--text:#f5f5f7;--muted:#a1a1a6;--line:#38383a;--accent:#0a84ff;--err:#ff453a}}}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
main{{max-width:460px;margin:40px auto;padding:0 16px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px}}
h1{{font-size:20px;margin:0 0 12px}}.muted,small{{color:var(--muted)}}.err{{color:var(--err)}}
fieldset{{border:0;padding:0;margin:16px 0}}legend{{font-weight:600;margin-bottom:8px}}
.opt{{display:flex;gap:10px;align-items:flex-start;padding:10px;border:1px solid var(--line);border-radius:10px;margin-bottom:8px;cursor:pointer}}
.code{{display:block;font-weight:600}}input[name=pairing_code]{{display:block;width:100%;box-sizing:border-box;margin-top:6px;padding:10px;
border:1px solid var(--line);border-radius:10px;background:transparent;color:inherit;font:600 18px ui-monospace,monospace;letter-spacing:2px}}
.buttons{{display:flex;gap:10px;margin-top:18px}}button{{flex:1;padding:11px;border-radius:10px;border:0;background:var(--accent);color:#fff;font-weight:600;font-size:15px;cursor:pointer}}
button.ghost{{background:transparent;color:var(--text);border:1px solid var(--line)}}
</style></head><body><main><div class="card"><h1>Connect an AI assistant to Blocky</h1>{body}</div></main></body></html>"""
