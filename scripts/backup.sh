#!/usr/bin/env bash
# Back up your personal Blocky data (chats, memories, test/benchmark/arena results, settings).
# Safe while Blocky is running (uses SQLite's online backup).
#   ./scripts/backup.sh [destination-folder]
# From Windows:  wsl -d blocky -- /mnt/c/dev/blockyOS/scripts/backup.sh /mnt/c/Users/<you>/Documents
set -euo pipefail
DEST="${1:-$HOME}"
FILE="$DEST/blocky-backup-$(date +%Y%m%d-%H%M).db"
python3 - "$FILE" <<'EOF'
import sqlite3, sys
src = sqlite3.connect("/opt/blocky/data/blocky.db")
dst = sqlite3.connect(sys.argv[1])
src.backup(dst)
dst.close(); src.close()
EOF
cp /opt/blocky/data/llm.env "$DEST/blocky-llm.env" 2>/dev/null || true
echo "Saved $FILE"
