# AI assistants: ChatGPT, Claude and other MCP clients

blockyOS is an [MCP](https://modelcontextprotocol.io) server, so an AI assistant you already use can see and use all of
it. The server is at `/mcp`. It works through the assistant's own app and account; blockyOS never contacts an AI
provider itself.

## What an assistant can do

You choose an access level for each assistant:

| Level | What it allows |
|---|---|
| **Read** | See everything: experiments, runs, results, models, engines, jobs, logs, chat history, settings, and read-only SQL over all stored data |
| **Write** | Read, plus start, queue, build and change things: experiments, sweeps, benchmarks, downloads, quality tests, training, research jobs (which run code on the machine), settings. Nothing is deleted |
| **Full** | Write, plus delete: experiments, runs, models, conversations, engines and other data |

At every level, some things are never available:
- **Secrets:** the remote-access token, the agent's Anthropic API key, the pairing code, and other devices' tokens.
- **Access endpoints:** the ones that decide who may connect, which are remote access, AI-assistant access, and the
  agent key.

An assistant can never grant itself more access.

### Tools

| Tool | Level | Purpose |
|---|---|---|
| `get_status`, `get_experiment`, `get_jobs`, `get_layer_scans`, `get_instrument_check`, … | read | The common research views, with guidance |
| `create_experiment`, `queue_run`, `queue_sweep`, `build_weight_variant`, `build_engine`, `start_training`, … | write | The common research steps, with validation |
| `list_api` | read | Every endpoint of Blocky's API: method, path, parameters, purpose |
| `api_get` | read | GET any endpoint |
| `api_call` | write | POST, PUT or PATCH any endpoint |
| `api_delete` | full | DELETE through any endpoint (marked destructive, so assistants ask before using it) |
| `query_database` | read | Read-only SQL over the database. With no query, it lists the tables and columns |

Answers longer than 60,000 characters are cut off, with a note that says so. Requests that take longer than 120
seconds return an error: an assistant should start a long job, then poll its status. Files such as zip bundles aren't
sent through MCP; open them on the device.

## Assistants on this machine

These connect straight to `http://localhost:8000/mcp` and get the **local level**. It's set under **Settings → AI
assistants**, and the default is Full.

| Assistant | How to connect |
|---|---|
| Claude Code | `claude mcp add --transport http blocky http://localhost:8000/mcp` |
| Claude Desktop | Settings → Developer → Edit config. Add `"blocky": {"command": "npx", "args": ["mcp-remote", "http://localhost:8000/mcp"]}` under `mcpServers` (needs Node.js) |
| VS Code, Cursor, Codex and others | An MCP server of type `http` with the URL `http://localhost:8000/mcp` |

The `research/` folder already contains `.mcp.json`, so a coding agent started there connects automatically.

## ChatGPT and Claude on the web and phone

Cloud assistants run on their provider's servers and can't reach `localhost`. To connect them, you do three things:
1. Publish blockyOS through an HTTPS tunnel.
2. Turn on public access.
3. Approve each assistant once.

### 1. Publish blockyOS through an HTTPS tunnel

You need a stable `https://` address that forwards to `http://localhost:8000` on the machine. Any of these work:

- **Cloudflare Tunnel:** `cloudflared tunnel --url http://localhost:8000` gives a temporary
  `https://….trycloudflare.com` address for trying it out. That address changes every time the tunnel restarts, so for
  daily use set up a named tunnel with your own domain.
- **Tailscale Funnel:** `tailscale funnel 8000` gives a stable `https://<machine>.<tailnet>.ts.net` address.
- **ngrok** or your own reverse proxy with a certificate.

Rules for the tunnel:
- **Keep the original Host header.** Most tunnels do this by default. Don't rewrite the Host header to `localhost`:
  blockyOS recognizes outside requests by the address they were sent to and by their forwarding headers.
- **Only the assistant connection is reachable.** Through the tunnel, only `/mcp` and the sign-in endpoints answer.
  The web page and the rest of the API still answer only on the device itself, or with the remote-access token.

### 2. Turn on public access

Go to **Settings → AI assistants → ChatGPT, Claude on the web and phone** and:
1. Enter the tunnel address, for example `https://blocky.example.com`.
2. Click **Save address**, then **On**.

Settings now shows two things:
- the **connector address**, which is the tunnel address followed by `/mcp`;
- a **pairing code**, which approves one assistant.

### 3. Connect the assistant

**ChatGPT:**
1. Go to **Settings → Apps & Connectors → Advanced settings** and turn on **Developer mode**.
2. Create a connector. Use the connector address, and choose OAuth for authentication.

ChatGPT's menus change from time to time. If the labels differ, look for creating a custom connector or app from an MCP
server URL.

**Claude** (claude.ai, the desktop app and the phone apps): go to **Settings → Connectors → Add custom connector** and
enter the connector address. Custom connectors may depend on your Claude plan.

Either way, a blockyOS page then opens:
1. It shows which app is asking and where it will send you back to, for example `chatgpt.com` or `claude.ai`.
2. Choose **Read**, **Write** or **Full**, and enter the pairing code.
3. Click **Approve**. You return to the assistant, which is now connected.

Each pairing code works once. After five wrong attempts, approvals are locked; creating a new code in Settings unlocks
them.

### Managing connections

**Settings → AI assistants → Connected assistants** lists each connection with its level and when it was last used.

- **Disconnect** a connection and its tokens stop working immediately.
- **Change a level** by disconnecting the assistant and connecting it again.
- **Close everything:** turning public access **Off** closes all cloud connections at once. Turning it on again
  reconnects the ones you haven't disconnected.

### How the sign-in works

blockyOS is its own OAuth 2.1 authorization server, following the MCP authorization specification:
- protected-resource and authorization-server metadata;
- dynamic client registration;
- authorization code with PKCE (S256 only);
- access tokens that last 1 hour;
- refresh tokens that last 30 days and change on every use.

Two protections against stolen tokens:
- **Reuse detection.** A sign-in code or refresh token used twice is treated as stolen, and the whole connection is
  cancelled.
- **Hashed storage.** Tokens and codes are stored only as SHA-256 hashes.

The approval page:
- can't be shown inside another site;
- is never cached;
- only redirects to the exact address the app registered.

## Things to keep in mind

- **Write and Full mean real control of the machine.** At Write, an assistant can start research jobs, which run
  arbitrary code as the blockyOS service account, and it can use the GPU for hours. Give cloud assistants the lowest
  level that does the job.
- **Your data goes to the assistant's provider.** What an assistant reads (results, chat history, logs) becomes part of
  your conversation with that provider, under its terms.
- **Assistants usually ask before acting.** ChatGPT and Claude ask before write actions and before tools marked
  destructive; keep those confirmations on.
- **The tunnel is the outside door.** Turn public access off when you don't need it, and disconnect assistants you no
  longer use.
