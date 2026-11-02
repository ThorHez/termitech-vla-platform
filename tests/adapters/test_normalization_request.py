"""A per-training normalization choice is separate from numeric runtime controls."""

import ast
import copy
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from vla_adapters import build_launch
from vla_adapters.automatic_config import build_automatic_base
from vla_adapters.generated_config import generated_training_config
from vla_adapters.openpi_entry import resolve_config
from vla_adapters.runtime_options import normalize_normalization_method
from vla_adapters.training_plan import fingerprint, inspect_training_config, norm_contract

from test_automatic_config import AutomaticTrainConfig, automatic  # noqa: F401
from test_data_processors import processor_module  # noqa: F401
from test_training_plan import Config, training  # noqa: F401


@dataclasses.dataclass
class ChoiceConfig(Config):
    normalization_method: str | None = None


@dataclasses.dataclass
class AutomaticChoiceConfig(AutomaticTrainConfig):
    normalization_method: str | None = None


def choice_base(base):
    return ChoiceConfig(**{field.name: getattr(base, field.name) for field in dataclasses.fields(base)})


def all_statistics(spec):
    for role, asset in spec["inputs"].items():
        if not role.startswith("norm"):
            continue
        path = Path(asset["path"])
        document = json.loads(path.read_text())
        for values in document["norm_stats"].values():
            values.update(min=[-3., -4.], max=[5., 6.])
        path.write_text(json.dumps(document))


def frozen_request(generated):
    node = next(node for node in ast.parse(generated["source"]).body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FROZEN_REQUEST"
                                                       for target in node.targets))
    return ast.literal_eval(node.value)


@pytest.mark.parametrize("method", ["minmax", "quantile", "zscore"])
@pytest.mark.parametrize("single", [False, True])
def test_recipe_single_and_multi_freeze_explicit_global_choice_without_changing_factories(training, method, single):
    base, spec, module, loaders = training
    base = choice_base(base)
    if single:
        base.data = base.data.datasets[0]
        spec["dataset_bindings"] = [spec["dataset_bindings"][0]]
        spec["inputs"] = {key: value for key, value in spec["inputs"].items() if key in {"model", "dataset_0", "norm_0"}}
    all_statistics(spec)
    spec["normalization_method"] = method
    original = copy.deepcopy(base)
    config = resolve_config(base, spec, module, loaders)
    assert config.normalization_method == method and base == original
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["normalization_method"] == method
    assert all(data["normalization_method"] == method for data in plan["datasets"])
    assert frozen_request(plan["generated_config"])["normalization_method"] == method
    assert "normalization_method=" in plan["generated_config"]["source"]
    assert all("normalization_method" not in vars(factory) for factory in
               (config.data.datasets if hasattr(config.data, "datasets") else [config.data]))


@pytest.mark.parametrize("method", ["minmax", "quantile", "zscore"])
def test_automatic_config_uses_the_same_global_choice(automatic, method):
    spec, module, loaders = automatic
    module.TrainConfig = AutomaticChoiceConfig
    all_statistics(spec)
    spec["normalization_method"] = method
    base = build_automatic_base(spec, module, loaders)
    config = resolve_config(base, spec, module, loaders)
    assert base.normalization_method is None and config.normalization_method == method
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["normalization_method"] == method
    assert frozen_request(plan["generated_config"])["normalization_method"] == method


def test_missing_and_none_choices_preserve_legacy_request_and_plan_shape(training):
    base, spec, module, loaders = training
    config = resolve_config(base, spec, module, loaders)
    plan = inspect_training_config(config, spec, checks=[])
    assert "normalization_method" not in plan
    assert "normalization_method" not in frozen_request(plan["generated_config"])
    spec["normalization_method"] = None
    same = inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert same == plan


@pytest.mark.parametrize("value", ["", "min_max", "q01_q99", "unknown", True, 1, {}, []])
def test_invalid_normalization_methods_are_never_silently_coerced(value):
    with pytest.raises(ValueError, match="NORMALIZATION_METHOD_INVALID"):
        normalize_normalization_method(value)


def test_explicit_choice_requires_a_runtime_that_supports_the_global_field(training):
    base, spec, module, loaders = training
    spec["normalization_method"] = "quantile"
    with pytest.raises(ValueError, match="NORMALIZATION_METHOD_UNSUPPORTED.*升级"):
        resolve_config(base, spec, module, loaders)


def test_changing_the_choice_changes_frozen_config_and_plan_identity(training):
    base, spec, module, loaders = training
    base = choice_base(base)
    all_statistics(spec)
    plans = []
    for method in ("minmax", "quantile", "zscore"):
        spec["normalization_method"] = method
        plans.append(inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[]))
    assert len({fingerprint(plan) for plan in plans}) == 3
    assert len({plan["generated_config"]["sha256"] for plan in plans}) == 3


def test_plan_refuses_a_runtime_that_did_not_apply_the_requested_choice(training):
    base, spec, module, loaders = training
    base = choice_base(base)
    spec["normalization_method"] = "quantile"
    config = resolve_config(base, spec, module, loaders)
    config.normalization_method = None
    with pytest.raises(ValueError, match="NORMALIZATION_METHOD_MISMATCH"):
        inspect_training_config(config, spec, checks=[])


@pytest.mark.parametrize("method,fields", [("minmax", ("min", "max")), ("quantile", ("q01", "q99")), ("zscore", ())])
def test_explicit_method_validates_its_bounds_instead_of_inheriting_legacy_quantiles(training, method, fields):
    base, spec, _, _ = training
    factory = base.data.datasets[0]
    factory.base_config = SimpleNamespace(use_quantile_norm=True)
    norm = Path(factory.assets.assets_dir) / "norm_stats.json"
    document = {"norm_stats": {key: {"mean": [0., 0.], "std": [1., 1.],
                                         **{name: [-2., -2.] if name in {"q01", "min"} else [2., 2.]
                                            for name in fields}} for key in ("state", "actions")}}
    norm.write_text(json.dumps(document))
    selected = {**spec, "normalization_method": method}
    assert norm_contract(factory, selected)["norm_path"] == str(norm)
    if method != "quantile":
        with pytest.raises(ValueError, match="缺少分位数"):
            norm_contract(factory, spec)
    if fields:
        del document["norm_stats"]["actions"][fields[0]]
        norm.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="NORM_FORMAT"):
            norm_contract(factory, selected)


def test_smoke_cannot_claim_an_explicit_normalization_method():
    with pytest.raises(ValueError, match="NORMALIZATION_METHOD_UNSUPPORTED"):
        build_launch({}, "smoke", {}, {}, "/unused", [], normalization_method="minmax")


@pytest.mark.parametrize("method", [None, "minmax", "quantile", "zscore"])
def test_launch_spec_carries_only_an_explicit_top_level_method(training, method):
    _, spec, _, _ = training
    profile = {"adapter": "openpi-sft", "execution_mode": "native", "code_dir": spec["code_dir"],
               "python": sys.executable, "recipe": spec["recipe"], "metadata": {}}
    launch = build_launch(profile, "openpi-sft", spec["inputs"], spec["parameters"], spec["output_dir"],
                          [f"gpu-{index}" for index in range(spec["gpu_count"])],
                          dataset_bindings=spec["dataset_bindings"], normalization_method=method)
    frozen = json.loads(launch["argv"][launch["argv"].index("--spec-json") + 1])
    if method is None:
        assert "normalization_method" not in frozen
    else:
        assert frozen["normalization_method"] == method
    assert "normalization_method" not in frozen["parameters"]
