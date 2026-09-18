from conftest import EXAMPLE_CONFIG
from dragon import mlxops
from dragon.config import Config


def fused_setup(tmp_path, monkeypatch, bits=None):
    config = Config.load(EXAMPLE_CONFIG)
    monkeypatch.setattr(type(config), "root", tmp_path, raising=False)
    config.root = tmp_path
    if bits is not None:
        config.raw["export"] = {"fuse_bits": bits}
    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters" / "adapters.safetensors").write_bytes(b"")
    calls = []

    def fake_run(args, cwd=None, tee=None):
        calls.append(args)
        # pretend the tool wrote its output
        if "--save-path" in args:
            (tmp_path / "fused-fp16").mkdir(exist_ok=True)
            (tmp_path / "fused-fp16" / "config.json").write_text("{}")
        if "--mlx-path" in args:
            (tmp_path / "fused").mkdir(exist_ok=True)
            (tmp_path / "fused" / "config.json").write_text("{}")
        return 0

    monkeypatch.setattr(mlxops, "require_mlx", lambda: None)
    monkeypatch.setattr(mlxops, "run", fake_run)
    return config, calls


def test_fuse_goes_by_way_of_fp16_and_requantises_at_eight_bits(tmp_path, monkeypatch):
    config, calls = fused_setup(tmp_path, monkeypatch)
    out = mlxops.fuse(config)
    assert out == tmp_path / "fused"
    assert "mlx_lm.fuse" in calls[0] and "--dequantize" in calls[0]
    assert "mlx_lm.convert" in calls[1]
    assert calls[1][calls[1].index("--q-bits") + 1] == "8"


def test_dequantize_stops_at_fp16(tmp_path, monkeypatch):
    config, calls = fused_setup(tmp_path, monkeypatch)
    assert mlxops.fuse(config, dequantize=True) == tmp_path / "fused-fp16"
    assert len(calls) == 1


def test_four_bits_is_allowed_but_warned_about(tmp_path, monkeypatch, capsys):
    config, calls = fused_setup(tmp_path, monkeypatch, bits=4)
    mlxops.fuse(config)
    assert calls[1][calls[1].index("--q-bits") + 1] == "4"
    assert "rounded away" in capsys.readouterr().out


def test_sixteen_bits_means_the_fp16_fuse_is_the_model(tmp_path, monkeypatch):
    config, calls = fused_setup(tmp_path, monkeypatch, bits=16)
    assert mlxops.fuse(config) == tmp_path / "fused-fp16"
    assert len(calls) == 1
