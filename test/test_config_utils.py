from types import SimpleNamespace
import os
import yaml

import pytest

from src.utils import load_config, transfer_namespace, path_name


def test_load_config(tmp_path):
    cfg = {
        "data": {"data_dir": "data", "metadata_file": "ds"},
        "model": {"name": "m", "type": "demo"},
        "task": {"type": "T", "name": "N"},
        "trainer": {"name": "tr"},
    }
    cfg_file = tmp_path / "cfg.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    loaded = load_config(str(cfg_file))
    assert loaded.data.metadata_file == "ds"
    assert loaded.model.name == "m"
    assert loaded.model.type == "demo"
    assert loaded.task.name == "N"
    assert loaded.trainer.name == "tr"


def test_transfer_namespace():
    ns = transfer_namespace({"a": 1, "b": "c"})
    assert isinstance(ns, SimpleNamespace)
    assert ns.a == 1
    assert ns.b == "c"


def test_path_name(tmp_path):
    configs = {
        "data": {"data_dir": "data", "metadata_file": "ds"},
        "model": {"name": "m", "type": "demo"},
        "task": {"type": "T", "name": "N"},
        "trainer": {"name": "tr"},
    }
    configs_ns = load_config(configs)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result_dir, exp_name = path_name(configs_ns, 1)
    finally:
        os.chdir(cwd)
    assert (tmp_path / result_dir).is_dir()
    assert exp_name.startswith("ds/M_m/T_TN_")


def test_load_config_with_include(tmp_path):
    shared = tmp_path / 'shared.yaml'
    shared.write_text("""data:
  data_dir: data
  metadata_file: ds
model:
  name: demo
  type: demo
task:
  name: demo
  type: demo
trainer:
  num_epochs: 5
  logging:
    contrastive_metrics: false
""", encoding='utf-8')

    base = tmp_path / 'base.yaml'
    base.write_text("""__include__:
  - shared.yaml
trainer:
  num_epochs: 10
""", encoding='utf-8')

    config = load_config(str(base))
    assert config.trainer.num_epochs == 10
    assert config.trainer.logging.contrastive_metrics is False



def test_load_config_include_cycle(tmp_path):
    path_a = tmp_path / 'a.yaml'
    path_b = tmp_path / 'b.yaml'
    path_a.write_text("""__include__:
  - b.yaml
""", encoding='utf-8')
    path_b.write_text("""__include__:
  - a.yaml
""", encoding='utf-8')

    with pytest.raises(RuntimeError):
        load_config(str(path_a))
