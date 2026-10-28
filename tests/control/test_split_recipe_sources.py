"""Split recipe sources retain immutable revisions and exact writeback scope."""

from copy import deepcopy

import pytest
from vla_adapters.recipe_definitions import RECIPE_PART_PATHS, SOURCE_PATHS
from vla_control.data_processors import _source

from .test_recipe_details import ready, result as detail_result
from .test_recipe_writeback import ack, create, get, result, saved


@pytest.mark.parametrize("path", sorted(RECIPE_PART_PATHS | SOURCE_PATHS))
def test_split_details_saved_revisions_and_writeback_keep_the_actual_source(platform, path):
    detail = detail_result()
    detail["source"]["path"] = path
    node, recipe, _ = saved(platform, value=detail)
    assert recipe["latest_revision"]["definition"]["source"]["path"] == path
    old = deepcopy(recipe["latest_revision"])
    response = create(platform, recipe)
    assert response.status_code == 202, response.text
    operation = response.json()
    command = next(item for item in platform.commands(node) if item["id"] == operation["id"])
    assert command["payload"]["definition"]["source"]["path"] == path
    receipt = result(platform, node, operation)
    assert receipt["changed_paths"] == [path]
    assert ack(platform, node, operation, receipt).status_code == 200
    completed = get(platform, operation).json()
    assert completed["status"] == "SUCCEEDED", completed
    assert completed["recipe"]["latest_revision"]["definition"]["source"]["path"] == path
    assert recipe["latest_revision"] == old


@pytest.mark.parametrize("path", ["src/openpi/training/recipes/other.py", "src/openpi/training/recipes/__init__.py",
    "src/openpi/training/recipes/sft.yaml", "src/openpi/training/recipes/../config.py",
    "/src/openpi/training/recipes/sft.py", "src/openpi/training/recipes/sft.py/child",
    "src/openpi/training/recipes/SFT.py", "src/openpi/training/mem/recipes/private.py"])
def test_unlisted_or_noncanonical_part_source_cannot_be_saved(platform, path):
    from .test_recipe_details import register
    node = register(platform)
    detail = detail_result()
    detail["source"]["path"] = path
    completed = ready(platform, node, value=detail)
    assert completed["status"] == "FAILED"
    assert completed["error"]["code"] == "RECIPE_DETAIL_INVALID_RESULT"


def test_writeback_cannot_report_a_different_allowed_part(platform):
    detail = detail_result()
    detail["source"]["path"] = "src/openpi/training/recipes/sft.py"
    node, recipe, _ = saved(platform, value=detail)
    operation = create(platform, recipe).json()
    receipt = result(platform, node, operation)
    receipt["changed_paths"] = ["src/openpi/training/recipes/mem.py"]
    receipt["detail"]["source"]["path"] = receipt["changed_paths"][0]
    assert ack(platform, node, operation, receipt).status_code == 200
    completed = get(platform, operation).json()
    assert completed["status"] == "FAILED"
    assert completed["error"]["code"] == "RECIPE_WRITE_INVALID_RESULT"


@pytest.mark.parametrize("path", sorted(RECIPE_PART_PATHS))
def test_processor_catalog_still_requires_the_root_config_facade(path):
    value = {"path": path, "sha256": "a" * 64, "code_dir": "/machine/code", "commit_sha": "b" * 40,
             "changes_digest": "c" * 64, "dirty": False}
    assert _source(value, "/machine/code", pinned=True) is None
