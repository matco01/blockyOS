"""Remote access: local-only by default, token required from other devices; device registry validation."""
import pytest

from app import remote


def test_local_only_by_default(fresh_db):
    remote.init()
    assert remote.allowed("127.0.0.1", "", "")
    assert remote.allowed("::1", "", "")
    assert not remote.allowed("192.168.1.20", "", "")
    assert not remote.allowed("192.168.1.20", "Bearer anything", "")


def test_token_when_remote_access_is_on(fresh_db):
    remote.init()
    token = remote.set_access(True)
    assert len(token) >= 24
    assert remote.allowed("192.168.1.20", f"Bearer {token}", "")
    assert remote.allowed("192.168.1.20", "", token)               # browser cookie
    assert not remote.allowed("192.168.1.20", "Bearer wrong", "")
    assert not remote.allowed("192.168.1.20", f"Basic {token}", "")
    remote.set_access(False)
    assert not remote.allowed("192.168.1.20", f"Bearer {token}", "")


def test_device_registry(fresh_db):
    remote.init()
    dev = remote.add_device({"url": "http://192.168.1.40:8000/", "token": "abc"})
    assert remote.list_devices() == [{**remote.list_devices()[0], "id": dev, "url": "http://192.168.1.40:8000", "token": True}]
    for bad in ("ftp://x", "http://a b", "javascript:alert(1)", "http://host/path?x=1"):
        with pytest.raises(ValueError):
            remote.add_device({"url": bad})
    remote.delete_device(dev)
    assert remote.list_devices() == []
