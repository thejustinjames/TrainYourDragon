import json
import threading
import urllib.error
import urllib.request

import pytest

from dragon import studio
from dragon.errors import DragonError
from dragon.studio import Paths, snapshot

LOG = """Loading pretrained model
Loading datasets
Training
Trainable parameters: 0.151% (11.534M/7615.616M)
Iter 1: Val loss 4.063, Val took 64.971s
Iter 20: Train loss 2.911, Learning Rate 1.000e-05, It/sec 0.103, Tokens/sec 90.474, Trained Tokens 20000, Peak mem 13.178 GB
Calculating loss...:  76%|████████  | 19/25 [01:02<00:12,  2.13s/it]
Iter 100: Val loss 2.001, Val took 97.092s
Iter 200: Val loss 1.709, Val took 70.516s
Iter 200: Saved adapter weights to adapters/adapters.safetensors and adapters/0000200_adapters.safetensors.
Iter 200: Train loss 1.502, Learning Rate 1.000e-05, It/sec 0.110, Tokens/sec 95.0, Trained Tokens 200000, Peak mem 13.2 GB
"""


def project(tmp_path, log=LOG, iters=400, stats=True):
    (tmp_path / "train.log").write_text(log)
    (tmp_path / "adapters").mkdir(exist_ok=True)
    st = tmp_path / "stats.json"
    if stats:
        st.write_text(
            json.dumps(
                {
                    "train": 100,
                    "valid": 10,
                    "test": 2,
                    "target_words": 50_000,
                    "brief_words": 5_000,
                    "by_tag": {"section": 60, "recall": 40},
                }
            )
        )
    return Paths(
        log=tmp_path / "train.log",
        adapters=tmp_path / "adapters",
        stats=st if stats else None,
        iters=iters,
        name="test run",
    )


def test_snapshot_reads_progress_and_the_latest_figures(tmp_path):
    s = snapshot(project(tmp_path))
    assert s.stage == "training" and s.live
    assert s.iteration == 200 and s.iters == 400 and s.progress == 0.5
    assert [p["iteration"] for p in s.val] == [1, 100, 200]
    assert s.latest["peak_mem_gb"] == 13.2
    assert s.saved == [200]
    assert s.lowest == {"iteration": 200, "loss": 1.709}
    assert s.trainable["percent"] == 0.151
    assert s.eta_seconds and abs(s.eta_seconds - 200 / 0.110) < 1
    assert s.passes and abs(s.passes - 200_000 / (55_000 * 1.3)) < 0.01
    assert s.corpus["by_tag"]["section"] == 60


def test_checkpoints_on_disk_count_even_if_the_log_does_not_mention_them(tmp_path):
    paths = project(tmp_path)
    (tmp_path / "adapters" / "0000400_adapters.safetensors").write_bytes(b"")
    assert snapshot(paths).saved == [200, 400]


def test_stages(tmp_path):
    assert snapshot(project(tmp_path, log="Loading pretrained model\n")).stage == "loading"
    assert snapshot(project(tmp_path, iters=200)).stage == "finished"
    missing = Paths(log=tmp_path / "nope.log", adapters=tmp_path / "adapters")
    assert snapshot(missing).stage == "waiting"


def test_a_quiet_log_is_reported_as_stalled(tmp_path, monkeypatch):
    paths = project(tmp_path)
    monkeypatch.setattr(studio, "STALE_AFTER_SECONDS", -1)
    s = snapshot(paths)
    assert s.stage == "stalled" and not s.live
    assert any("dozing" in line for line in s.reading)


def test_without_stats_there_is_no_pass_count_and_no_crash(tmp_path):
    s = snapshot(project(tmp_path, stats=False))
    assert s.passes is None and s.corpus is None


def test_the_page_and_the_api_are_served(tmp_path):
    paths = project(tmp_path)
    server = studio.make_server(paths, port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        port = server.server_address[1]
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3).read().decode()
        assert "<title>Observability studio · Agencie.io Labs</title>" in body
        raw = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3).read()
        state = json.loads(raw)
        assert state["iteration"] == 200 and state["stage"] == "training"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=3)
    finally:
        server.shutdown()
        server.server_close()


def test_the_log_tail_leaves_out_progress_bars(tmp_path):
    tail = snapshot(project(tmp_path)).log_tail
    assert not any("Calculating loss" in line for line in tail)
    assert any(line.startswith("Iter 100: Val loss") for line in tail)


def test_discover_finds_every_run_newest_first_with_its_own_adapters(tmp_path):
    import os
    import time

    (tmp_path / "train-run1.log").write_text(LOG.replace("Iter 200: Val", "Iter 200: Val"))
    (tmp_path / "adapters-run1").mkdir()
    (tmp_path / "adapters").mkdir()
    old = time.time() - 600
    os.utime(tmp_path / "train-run1.log", (old, old))
    (tmp_path / "train.log").write_text(LOG)
    runs = studio.discover(tmp_path, iters=400)
    assert [r.key for r in runs] == ["train", "train-run1"]
    assert runs[0].adapters == tmp_path / "adapters" and runs[0].iters == 400
    assert runs[1].adapters == tmp_path / "adapters-run1" and runs[1].iters is None
    assert runs[1].name.endswith("run1") and runs[0].name.endswith("current")


def test_discover_with_nothing_logged_still_points_at_train_log(tmp_path):
    runs = studio.discover(tmp_path)
    assert len(runs) == 1 and runs[0].log == tmp_path / "train.log"
    assert snapshot(runs[0]).stage == "waiting"


def test_a_finished_run_without_a_config_knows_its_own_length(tmp_path):
    (tmp_path / "train.log").write_text(
        LOG + "Saved final weights to adapters/adapters.safetensors.\n"
    )
    s = snapshot(studio.discover(tmp_path)[0])
    assert s.stage == "finished" and s.iters == 200 and s.progress == 1.0


def test_an_old_unfinished_run_is_stopped_not_stalled(tmp_path, monkeypatch):
    paths = project(tmp_path)
    monkeypatch.setattr(studio, "STOPPED_AFTER_SECONDS", -1)
    s = snapshot(paths)
    assert s.stage == "stopped"
    assert any("stopped or it crashed" in line for line in s.reading)


def test_the_runs_index_and_run_selection_are_served(tmp_path):
    (tmp_path / "train-run1.log").write_text(LOG)
    paths = project(tmp_path)
    runs = studio.discover(tmp_path, iters=400)
    server = studio.make_server(runs, port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        port = server.server_address[1]
        index = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=3).read()
        )
        assert [r["key"] for r in index] == ["train", "train-run1"]
        one = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/state?run=train-run1", timeout=3
            ).read()
        )
        assert one["name"].endswith("run1")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state?run=nope", timeout=3)
    finally:
        server.shutdown()
        server.server_close()
    del paths


def _occupy(port):
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    return s


def test_a_free_port_is_returned_as_is():
    port = studio.next_free_port("127.0.0.1", 40000)
    assert studio.choose_port("127.0.0.1", port, if_busy="fail") == port


def test_a_busy_port_moves_to_the_next_free_one_or_fails():
    port = studio.next_free_port("127.0.0.1", 41000)
    held = _occupy(port)
    try:
        said = []
        chosen = studio.choose_port("127.0.0.1", port, if_busy="next", say=said.append)
        assert chosen > port and studio.port_free("127.0.0.1", chosen)
        assert any(str(port) in line for line in said)
        with pytest.raises(DragonError, match=f"port {port} is in use"):
            studio.choose_port("127.0.0.1", port, if_busy="fail")
    finally:
        held.close()


def test_asking_offers_end_another_or_a_typed_port():
    port = studio.next_free_port("127.0.0.1", 42000)
    held = _occupy(port)
    try:
        alt = studio.next_free_port("127.0.0.1", port)
        assert studio.choose_port("127.0.0.1", port, ask=lambda _: "a", say=lambda *_: None) == alt
        typed = studio.next_free_port("127.0.0.1", alt)
        assert (
            studio.choose_port("127.0.0.1", port, ask=lambda _: str(typed), say=lambda *_: None)
            == typed
        )
        with pytest.raises(DragonError, match="stopped"):
            studio.choose_port("127.0.0.1", port, ask=lambda _: "q", say=lambda *_: None)
    finally:
        held.close()


def test_ending_the_occupant_frees_the_port():
    import shutil
    import subprocess
    import sys
    import time

    if not shutil.which("lsof"):
        pytest.skip("lsof is needed to find the port's owner")
    port = studio.next_free_port("127.0.0.1", 43000)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,time; s=socket.socket(); "
            f"s.bind(('127.0.0.1',{port})); s.listen(1); time.sleep(30)",
        ]
    )
    try:
        for _ in range(50):
            if not studio.port_free("127.0.0.1", port):
                break
            time.sleep(0.1)
        owners = studio.port_owner(port)
        assert child.pid in [pid for pid, _ in owners]
        said = []
        assert studio.choose_port("127.0.0.1", port, if_busy="kill", say=said.append) == port
        child.wait(timeout=5)
        assert any("ended" in line for line in said)
    finally:
        if child.poll() is None:
            child.kill()


def test_a_log_that_appears_later_becomes_a_tab_without_a_restart(tmp_path):
    project(tmp_path)
    server = studio.make_server(lambda: studio.discover(tmp_path, iters=400), port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        port = server.server_address[1]
        before = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=3).read()
        )
        assert [r["key"] for r in before] == ["train"]
        (tmp_path / "train-run1.log").write_text(LOG)
        after = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=3).read()
        )
        assert sorted(r["key"] for r in after) == ["train", "train-run1"]
        one = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/state?run=train-run1", timeout=3
            ).read()
        )
        assert one["iteration"] == 200
    finally:
        server.shutdown()
        server.server_close()
