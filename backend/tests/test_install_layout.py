"""The service runs as user blocky and can't create folders in a root-owned /opt/blocky (a fresh install, the Docker
image). So install.sh and the Dockerfile must create, and hand to blocky, every folder the backend keeps next to its
data folder. Before this, both only made models and data, and the backend failed at startup (engines, hf)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _folders_next_to_data() -> set[str]:
    used = set()
    for f in (ROOT / "backend" / "app").glob("*.py"):
        used |= set(re.findall(r'DATA_DIR\.parent / "([\w.-]+)"', f.read_text(encoding="utf-8")))
    # Python environments are made by the setup scripts (as root); the service only runs them.
    return {n for n in used if not n.endswith("-venv")} | {"models", "data"}


def _line(text: str, part: str) -> str:
    """The line that makes or hands over the folders: the one that includes the models folder."""
    return next(ln for ln in text.splitlines() if part in ln and not ln.lstrip().startswith("#"))


def test_found_the_folders():
    assert {"engines", "hf", "models", "data"} <= _folders_next_to_data()


def test_install_sh_creates_them_for_the_service():
    install = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    made, owned = _line(install, 'mkdir -p "$PREFIX/models"'), _line(install, 'chown -R blocky:blocky "$PREFIX/models"')
    for name in _folders_next_to_data():
        assert f'"$PREFIX/{name}"' in made, f"install.sh doesn't create {name}"
        assert f'"$PREFIX/{name}"' in owned, f"install.sh doesn't give {name} to blocky"


def test_the_image_creates_them_for_the_service():
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    made, owned = _line(docker, "mkdir -p /opt/blocky/models"), _line(docker, "chown -R blocky:blocky /opt/blocky/models")
    for name in _folders_next_to_data():
        pattern = rf"/opt/blocky/{re.escape(name)}(\s|$)"
        assert re.search(pattern, made), f"the image doesn't create {name}"
        assert re.search(pattern, owned), f"the image doesn't give {name} to blocky"
