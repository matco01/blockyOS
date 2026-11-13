# Research workspace

Open this folder with an AI coding agent that runs on the Blocky device: Claude Code, Codex, or an editor agent.

- **VS Code:** connect to the Blocky device (Remote-SSH for a dedicated box, or the WSL extension when Blocky runs in
  WSL), open this folder, start Claude Code. It picks up `CLAUDE.md` (how to do research here) and `.mcp.json` (Blocky's
  lab tools) automatically; approve the `blocky` server when asked.
- **Terminal:** `cd` here and run `claude` (or `codex`).

The agent then has everything: the lab tools over MCP, the Python client for analysis, PyTorch for new techniques, a
shell, and this folder for its scripts and notes (`notes/`, ignored by git).
