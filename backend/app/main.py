"""blockyOS web backend.

One process that:
  - serves the web UI
  - streams chat to/from llama-server
  - stores conversations and long-term memory in SQLite
  - reports hardware stats
  - runs the research lab (experiments, engines, weights, PyTorch jobs) and the MCP server

This module wires the app together: startup/shutdown, access control, and the routers in app/routes/ (one per area).
"""
import asyncio
from contextlib import asynccontextmanager

import httpx
import psutil
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings

from . import agent, aiaccess, arena, benchmarks, config, db, engines, evals, jobstore, lab, mcp_server, profile, projects, quantlab, remote, research_jobs, scans
from . import rpcworker, tools, torchlab, weights
from .routes import (agent as agent_routes, aiaccess as aiaccess_routes, arena as arena_routes, bench, chat, common, devices, engines as engine_routes, evals as eval_routes,
                     experiments, models, openai, projects as project_routes, quantlab as quantlab_routes, research, settings, system, torch)

# Include order matters only where two paths could match the same URL; it follows the order these routes always had.
ROUTERS = (aiaccess_routes, research, chat, settings, agent_routes, devices, system, models, bench, openai, eval_routes, quantlab_routes, experiments,
           engine_routes, torch, arena_routes, project_routes)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    jobstore.init()
    arena.init()
    evals.init()
    quantlab.init()
    tools.init()
    profile.init()
    weights.init()
    engines.init()
    scans.init()
    torchlab.init()
    remote.init()
    agent.init()
    research_jobs.init()
    projects.init()  # after experiments and research projects: it adds a column to both
    rpcworker.init()
    aiaccess.init()
    for name, jobs in (("downloads", lab.downloads), ("benchmark-downloads", benchmarks.downloads),
                       ("weights", weights.jobs), ("quantize", quantlab.jobs)):
        jobstore.persist(name, jobs)
    psutil.cpu_percent(interval=None)  # prime the CPU counter so the first reading isn't 0
    app.state.llm = httpx.AsyncClient(base_url=config.LLM_URL, timeout=httpx.Timeout(10, read=300))
    app.state.background = set()  # keep references so tasks aren't garbage-collected
    common.state.llm, common.state.background = app.state.llm, app.state.background  # what the route modules use
    worker = asyncio.create_task(profile.worker(app.state.llm))
    app.state.background.add(worker)
    research_worker = asyncio.create_task(research_jobs.worker(app.state.llm))
    app.state.background.add(research_worker)
    saver = asyncio.create_task(jobstore.saver())
    app.state.background.add(saver)
    async with mcp_server.server.session_manager.run():
        yield
    saver.cancel()
    jobstore.save()
    worker.cancel()
    research_worker.cancel()
    _, pending = await asyncio.wait({worker, research_worker}, timeout=10)
    for task in pending:
        task.cancel()
    await asyncio.gather(worker, research_worker, return_exceptions=True)
    rpcworker.stop()
    await app.state.llm.aclose()


app = FastAPI(title="blockyOS", lifespan=lifespan)


FORWARDED = ("x-forwarded-for", "forwarded", "cf-connecting-ip", "x-real-ip", "true-client-ip")
OAUTH_BACKCHANNEL = ("/oauth/token", "/oauth/register", "/oauth/revoke")


def _cloud_path(path: str) -> bool:
    return path in ("/mcp", "/mcp/") or path.startswith(("/oauth/", "/.well-known/oauth-", "/.well-known/openid-configuration"))


class LocalOnly:
    """Only this device may use Blocky, unless its owner turned on remote access (then the token is required).

    Plain ASGI (not BaseHTTPMiddleware) so streaming responses pass through untouched. A browser on another device can
    open http://<box>:8000/?key=<token> once; the key is then kept in a cookie.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        from http.cookies import SimpleCookie
        from urllib.parse import parse_qs
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = remote.effective_client((scope.get("client") or ("",))[0], headers.get("host", ""))
        if host in remote.LOOPBACK_NAMES and any(h in headers for h in FORWARDED):
            host = "forwarded"  # arrived through a tunnel or reverse proxy on this machine: someone else's request
        path = scope.get("path", "")
        unsafe = scope["type"] == "websocket" or scope.get("method") not in ("GET", "HEAD", "OPTIONS")
        # A browser can't attach an Authorization header cross-site without a CORS preflight (which Blocky never
        # approves), and the OAuth token endpoints carry no ambient authority, so those skip the Origin check.
        exempt = "authorization" in headers or path in OAUTH_BACKCHANNEL
        if unsafe and not exempt and not remote.same_origin(headers.get("origin", ""), headers.get("host", "")):
            # Cross-site requests (CSRF): browsers attach no preflight to "simple" POSTs, so refuse them here.
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            resp = JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
            return await resp(scope, receive, send)
        key = (parse_qs(scope.get("query_string", b"").decode("latin-1")).get("key") or [""])[0]
        jar = SimpleCookie(headers.get("cookie", ""))
        cookie = jar["blocky_key"].value if "blocky_key" in jar else ""
        if remote.allowed(host, headers.get("authorization", ""), cookie or key):
            scope["blocky.trusted"] = True  # this device, or another device with the remote-access token
            if key and host not in ("127.0.0.1", "::1"):
                async def send_with_cookie(msg):
                    if msg["type"] == "http.response.start":
                        msg = {**msg, "headers": [*msg.get("headers", []),
                                                  (b"set-cookie", f"blocky_key={key}; HttpOnly; SameSite=Strict; Max-Age=2592000; Path=/".encode())]}
                    await send(msg)
                return await self.app(scope, receive, send_with_cookie)
            return await self.app(scope, receive, send)
        if scope["type"] == "http" and _cloud_path(path) and aiaccess.public_active() and remote.host_name(headers.get("host", "")) == aiaccess.public_host():
            # A cloud AI assistant through the owner's tunnel: only MCP (OAuth-protected) and the sign-in endpoints.
            scope["blocky.trusted"] = False
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        resp = JSONResponse({"detail": "This Blocky only accepts connections from itself. To use it from another device, turn on "
                                       "remote access in its Settings, or connect through an SSH tunnel."}, status_code=403)
        await resp(scope, receive, send)


app.add_middleware(LocalOnly)
for _r in ROUTERS:
    app.include_router(_r.router)


# ---------- MCP: AI assistants (ChatGPT, Claude, Claude Code, other MCP clients) use the lab at /mcp ----------

def _mcp_entry(inner):
    """Decide the caller's access level: this device (or a remote-access token) gets the owner's local level; a cloud
    assistant needs a valid OAuth access token and gets the level the owner approved for it."""
    async def entry(scope, receive, send):
        if scope.get("blocky.trusted"):
            level = aiaccess.config()["local_level"]
        else:
            auth = dict(scope.get("headers", [])).get(b"authorization", b"").decode("latin-1")
            level = aiaccess.verify(auth[7:].strip()) if auth[:7].lower() == "bearer " else None
            if not level:
                meta = f"{aiaccess.public_base()}/.well-known/oauth-protected-resource/mcp"
                resp = JSONResponse({"error": "invalid_token", "error_description": "Sign in to use this Blocky."}, status_code=401,
                                    headers={"WWW-Authenticate": f'Bearer resource_metadata="{meta}"'})
                return await resp(scope, receive, send)
        token = mcp_server.LEVEL.set(level)
        try:
            await inner(scope, receive, send)
        finally:
            mcp_server.LEVEL.reset(token)
    return entry


# Stateless Streamable HTTP. Host and Origin checks (DNS rebinding, cross-site requests) are done by LocalOnly above,
# which also knows the owner's public address, so the SDK's localhost-only check is off.
_mcp_app = mcp_server.server.streamable_http_app(streamable_http_path="/mcp", stateless_http=True, json_response=True,
                                                 transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
for _route in _mcp_app.routes:
    if getattr(_route, "path", "") == "/mcp":
        _route.app = _mcp_entry(_route.app)
app.router.routes.extend(_mcp_app.routes)
mcp_server.APP = app


# ---------- Web UI (must be last: catches all other paths) ----------

if config.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        file = config.FRONTEND_DIST / path
        if path and file.is_file() and config.FRONTEND_DIST in file.resolve().parents:
            return FileResponse(file)
        return FileResponse(config.FRONTEND_DIST / "index.html")
