import pytest

from conftest import EXAMPLE_CONFIG
from dragon.config import Config
from dragon.errors import DragonError
from dragon.runlog import checkpoints, curve, describe, passes, promote

LOG = """Loading pretrained model
Iter 1: Val loss 3.842, Val took 40.1s
Iter 20: Train loss 2.911, Learning Rate 1.000e-05, It/sec 0.4
Iter 100: Val loss 1.980, Val took 39.2s
Iter 200: Val loss 1.960, Val took 39.0s
Iter 300: Val loss 1.840, Val took 39.3s
Iter 300: Train loss 1.702, Learning Rate 1.000e-05, It/sec 0.4
Iter 400: Val loss 1.630, Val took 39.1s
Iter 500: Val loss 1.635, Val took 39.1s
Iter 600: Val loss 1.660, Val took 39.1s
"""


def test_curve_reads_validation_and_training_points(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(LOG)
    val, train = curve(log)
    assert [p.iteration for p in val] == [1, 100, 200, 300, 400, 500, 600]
    assert val[0].loss == pytest.approx(3.842)
    assert train[-1].iteration == 300


def test_curve_without_a_log_is_a_clear_error(tmp_path):
    with pytest.raises(DragonError, match="Nothing has been trained"):
        curve(tmp_path / "train.log")


def test_describe_names_the_low_point_and_the_upturn(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(LOG)
    val, _ = curve(log)
    text = "\n".join(describe(val))
    assert "1.630 at iteration 400" in text
    assert "memorisation" in text


def test_promote_copies_the_named_checkpoint(tmp_path, monkeypatch):
    config = Config.load(EXAMPLE_CONFIG)
    monkeypatch.setattr(type(config), "adapter_dir", property(lambda self: tmp_path))
    (tmp_path / "0000200_adapters.safetensors").write_bytes(b"two hundred")
    (tmp_path / "0000400_adapters.safetensors").write_bytes(b"four hundred")
    assert [i for i, _ in checkpoints(tmp_path)] == [200, 400]
    live = promote(config, 400)
    assert live.read_bytes() == b"four hundred"
    with pytest.raises(DragonError, match="Saved: 200, 400"):
        promote(config, 300)


def test_passes_scale_with_iterations():
    config = Config.load(EXAMPLE_CONFIG)
    one = passes(config, target_words=100_000)
    config.training["iters"] *= 2
    assert passes(config, target_words=100_000) == pytest.approx(one * 2)
    assert passes(config, target_words=0) == 0.0
