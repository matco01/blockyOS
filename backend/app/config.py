"""Runtime configuration. Everything is overridable via environment variables
(set in the systemd unit on the device), so the code never hardcodes hardware."""
import os
from pathlib import Path

# Where llama-server listens. Only reachable from this machine.
LLM_URL = os.environ.get("BLOCKY_LLM_URL", "http://127.0.0.1:8080")

# Where we keep the database and other state.
DATA_DIR = Path(os.environ.get("BLOCKY_DATA_DIR", "/opt/blocky/data"))
DB_PATH = DATA_DIR / "blocky.db"

# Built frontend (npm run build output).
FRONTEND_DIST = Path(
    os.environ.get(
        "BLOCKY_FRONTEND_DIST",
        Path(__file__).resolve().parents[2] / "frontend" / "dist",
    )
)

# How many past messages to send to the model with each request.
MAX_HISTORY_MESSAGES = int(os.environ.get("BLOCKY_MAX_HISTORY", "20"))

SYSTEM_PROMPT = os.environ.get(
    "BLOCKY_SYSTEM_PROMPT",
    "You are Blocky, a helpful AI assistant running fully locally on the user's "
    "own device. Be concise and accurate. Use Markdown when it helps.",
)
