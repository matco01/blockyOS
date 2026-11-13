#!/bin/bash
# JupyterLab on the Blocky box: local-only (127.0.0.1:8888), token-protected, runs as the `blocky` user.
# The `blocky` Python client (client/blocky.py in the repo) is importable in every notebook.
# Run with sudo. Safe to run again.
set -euo pipefail
PREFIX=/opt/blocky
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PREFIX/notebook-venv"
NB="$PREFIX/data/notebooks"

echo "==> Notebook environment"
[[ -d "$VENV" ]] || sudo -u blocky python3 -m venv "$VENV"
sudo -u blocky "$VENV/bin/pip" install -q --upgrade pip
sudo -u blocky "$VENV/bin/pip" install -q jupyterlab pandas matplotlib numpy mlflow-skinny sqlalchemy alembic wandb
# Make `import blocky` work, always pointing at the repo's current client.
SITE=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
echo "$REPO_DIR/client" | sudo -u blocky tee "$SITE/blocky-client.pth" >/dev/null

echo "==> Token and notebooks folder"
sudo -u blocky mkdir -p "$NB"
if [[ ! -s "$PREFIX/data/notebook-token" ]]; then
  sudo -u blocky sh -c "umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(24))' > '$PREFIX/data/notebook-token'"
fi
for f in "$REPO_DIR"/client/examples/*.ipynb; do
  [[ -e "$NB/$(basename "$f")" ]] || sudo -u blocky cp "$f" "$NB/"
done

echo "==> Service"
cat > /etc/systemd/system/blocky-notebook.service <<EOF
[Unit]
Description=Blocky notebooks (JupyterLab, local only)
After=network.target blocky-web.service

[Service]
User=blocky
WorkingDirectory=$NB
Environment=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/bin/bash -c 'exec $VENV/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser --ServerApp.root_dir=$NB --IdentityProvider.token="\$(cat $PREFIX/data/notebook-token)" --ServerApp.allow_origin=http://localhost:8000'
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now blocky-notebook.service
echo "    JupyterLab: http://localhost:8888/lab (token in $PREFIX/data/notebook-token)"
