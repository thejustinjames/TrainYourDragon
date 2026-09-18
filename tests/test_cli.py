import pytest
import yaml

from conftest import EXAMPLE_CONFIG
from dragon.cli import main


def test_build_dry_run_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["-c", str(EXAMPLE_CONFIG), "build", "--dry-run"])
    assert code == 0
    assert "words of target text" in capsys.readouterr().out
    assert not (tmp_path / "data").exists()


def test_init_writes_a_config_that_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert parsed["model"]["base"]
    assert parsed["voice"]["system_prompt"]


def test_init_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init"])
    assert main(["init"]) == 1
    assert "already exists" in capsys.readouterr().err


def test_a_missing_config_fails_with_a_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 1
    assert "config.yaml" in capsys.readouterr().out


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main([])


def test_doctor_flags_a_holdout_that_matches_nothing(tmp_path, capsys):
    import shutil

    import yaml

    project = tmp_path / "project"
    shutil.copytree(EXAMPLE_CONFIG.parent, project)
    cfg = yaml.safe_load((project / "config.yaml").read_text())
    cfg["corpus"]["holdout"] = ["the-quiet-part", "no-such-piece"]
    cfg["corpus"]["exclude"] = ["also-missing"]
    (project / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert main(["-c", str(project / "config.yaml"), "doctor"]) == 1
    out = capsys.readouterr().out
    assert "'no-such-piece' matches no file" in out
    assert "'also-missing' matches no file" in out
    assert "the-quiet-part" in out
