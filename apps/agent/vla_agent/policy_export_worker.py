"""Executed by the source OpenPI interpreter; never loads model weights."""

import json
import logging
from pathlib import Path
import signal
import sys


def _record(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(path)


def export(manifest):
    if (Path(manifest["output_dir"]) / "cancelled.json").exists():
        raise ValueError("POLICY_EXPORT_CANCELLED: inference export was cancelled")
    sys.path.insert(0, manifest["adapter_root"])
    from vla_adapters.portable_policy import _code_files, export_portable_policy, portable_digest
    if _code_files(manifest["code_dir"]) != manifest["code_files_baseline"]:
        raise ValueError("POLICY_EXPORT_CODE_CHANGED: source code changed after checking its training snapshot")
    from vla_adapters.openpi_entry import prepare_config
    from vla_adapters.generated_config import generated_training_config
    from openpi.training import config
    if not Path(config.__file__).resolve().is_relative_to(Path(manifest["code_dir"]).resolve()):
        raise ValueError("MODULE_OUTSIDE_CODE_DIR: OpenPI resolves outside its registered checkout")
    spec = dict(manifest["frozen_request"], output_dir=str(Path(manifest["output_dir"]) / "configuration"))
    if (Path(manifest["output_dir"]) / "cancelled.json").exists():
        raise ValueError("POLICY_EXPORT_CANCELLED: inference export was cancelled")
    train_config, _ = prepare_config(spec)
    if generated_training_config(train_config, spec)["sha256"] != manifest["generated_sha256"]:
        raise ValueError("CONFIG_CHANGED: resolved inference configuration differs from confirmed training")
    portable = export_portable_policy(train_config, manifest["code_dir"])
    if portable["code_files"] != manifest["code_files_baseline"] or _code_files(manifest["code_dir"]) != manifest["code_files_baseline"]:
        raise ValueError("POLICY_EXPORT_CODE_CHANGED: source code changed while exporting inference transforms")
    return {"state": "READY", "portable_config": portable, "portable_digest": portable_digest(portable),
            "source_code": {"files": len(portable["code_files"]), "digest": portable_digest(portable["code_files"]),
                            "model_family": portable["model_family"]}}


def main():
    path = Path(sys.argv[1])
    manifest = json.loads(path.read_text())
    manifest["output_dir"] = str(path.parent)
    def timeout(_sig, _frame):
        raise TimeoutError("POLICY_EXPORT_TIMEOUT: exporting inference configuration exceeded 180 seconds")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    try:
        result = export(manifest)
    except Exception as exc:
        prefix = str(exc).split(":", 1)[0]
        code = prefix if prefix.isupper() and prefix.replace("_", "").isalnum() and len(prefix) <= 100 else "POLICY_EXPORT_FAILED"
        result = {"state": "FAILED", "error": {"code": code, "message": str(exc)[:2000]}}
        logging.exception("Inference configuration export failed")
    finally:
        signal.alarm(0)
    if (path.parent / "cancelled.json").exists():
        result = {"state": "FAILED", "error": {"code": "POLICY_EXPORT_CANCELLED", "message": "Inference configuration export was cancelled"}}
    _record(path.parent / "result.json", result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
