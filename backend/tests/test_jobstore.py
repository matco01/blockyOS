"""Background-job records survive a restart; jobs that were active come back as interrupted, not as running."""
from app import db, jobstore


def test_jobs_survive_restart(fresh_db):
    jobstore.init()
    jobs = {}
    jobstore.persist("t", jobs)
    jobs["a"] = {"id": "a", "status": "done", "out": "x.gguf"}
    jobs["b"] = {"id": "b", "status": "running", "step": "quantizing"}
    jobstore.save()
    jobstore._last.clear()

    restored = {}
    jobstore.persist("t", restored)
    assert restored["a"] == {"id": "a", "status": "done", "out": "x.gguf"}
    assert restored["b"]["status"] == "error" and "restarted" in restored["b"]["error"]
    assert restored["b"]["interrupted_status"] == "running"


def test_removed_jobs_are_deleted_and_int_keys_restore(fresh_db):
    jobstore.init()
    jobs = {}
    jobstore.persist("n", jobs, int)
    jobs[1] = {"status": "done"}
    jobs[2] = {"status": "done"}
    jobstore.save()
    del jobs[1]
    jobstore.save()
    jobstore._last.clear()
    restored = {}
    jobstore.persist("n", restored, int)
    assert list(restored) == [2]
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM bg_jobs WHERE store='n'").fetchone()[0] == 1
