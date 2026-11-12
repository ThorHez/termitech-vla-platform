"""Human model labels backed by recorded training and verified-copy lineage.

Names are presentation metadata only. These helpers never change a checkpoint
path, frozen transfer binding, recipe, asset identity, or byte-copy contract.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import bindparam, select, update


MAX_MODEL_NAME = 180
SYNC_SUFFIX = " · 资产同步"


def _field(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and not any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)


def _metadata(asset):
    value = _field(asset, "metadata_json")
    if value is None:
        value = _field(asset, "metadata")
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class _Generated:
    canonical: str
    aliases: frozenset[str]


def _generated(run, attempt, output):
    metadata = _metadata(output)
    recipe, step, version = metadata.get("recipe"), metadata.get("step"), _field(output, "version")
    if (_field(output, "kind") != "MODEL" or metadata.get("format") not in ("openpi-orbax", "openpi-pytorch")
            or not _text(recipe) or type(step) is not int or step < 0
            or not isinstance(version, str) or not version.isascii() or not version.isdecimal()
            or len(version) > 200 or int(version) != step):
        return None
    run_id, title, number = _field(run, "id"), _field(run, "name"), _field(attempt, "number", 1)
    if not _text(run_id) or not _text(title) or type(number) is not int or number < 1:
        return None
    # A pathological step must not drop either identity or step to fit a label.
    # Such old/custom data simply retains its existing name instead.
    tail = f" · {run_id[:8]}" + (f" · 执行 {number}" if number > 1 else "") + f" / {step}"
    if len(tail) >= MAX_MODEL_NAME:
        return None
    canonical = title[:MAX_MODEL_NAME - len(tail)].rstrip() + tail
    return _Generated(canonical, frozenset({recipe + " / " + version, canonical}))


def model_output_name(run, attempt, output) -> str:
    """Replace only the scanner's exact legacy OpenPI model label."""
    name = _field(output, "name")
    generated = _generated(run, attempt, output)
    if generated and name == _metadata(output)["recipe"] + " / " + _field(output, "version"):
        return generated.canonical
    return name


class _Resolver:
    def __init__(self, asset, attempt, run):
        self.asset, self.attempt, self.run = asset, attempt, run
        self.memo: dict[str, _Generated | None] = {}

    def resolve(self, asset_id):
        """Iterative traversal handles long copy chains and rejects cycles."""
        if asset_id in self.memo:
            return self.memo[asset_id]
        trail, seen, current = [], set(), asset_id
        while True:
            if current in self.memo:
                generated = self.memo[current]
                break
            if current in seen:
                generated = None
                break
            asset = self.asset(current)
            if asset is None or _field(asset, "kind") != "MODEL":
                generated = None
                break
            trail.append(asset)
            seen.add(current)
            origin = _field(asset, "origin_attempt_id")
            if origin:
                attempt = self.attempt(origin)
                run = self.run(_field(attempt, "run_id")) if attempt is not None else None
                if (attempt is None or run is None
                        or _field(attempt, "node_id") != _field(asset, "node_id")
                        or _field(run, "node_id") != _field(asset, "node_id")
                        or _field(attempt, "run_id") != _field(run, "id")):
                    generated = None
                else:
                    generated = _generated(run, attempt, asset)
                break
            sync = _metadata(asset).get("asset_sync")
            if not isinstance(sync, dict) or sync.get("byte_verified") is not True or not _text(sync.get("source_asset_id")):
                generated = None
                break
            source = self.asset(sync["source_asset_id"])
            if (source is None or _field(source, "kind") != "MODEL"
                    or sync.get("source_node_id") != _field(source, "node_id")
                    or sync.get("source_path") != _field(source, "path")
                    or _field(asset, "version") != _field(source, "version")):
                generated = None
                break
            current = sync["source_asset_id"]
        for asset in reversed(trail):
            if generated and _field(asset, "name") not in generated.aliases:
                generated = None
            self.memo[_field(asset, "id")] = generated
        return self.memo.get(asset_id)


def _session_resolver(session):
    from .models import Asset, Attempt, Run
    assets, attempts, runs = {}, {}, {}

    def get(model, cache, identifier):
        if identifier not in cache:
            cache[identifier] = session.get(model, identifier) if identifier else None
        return cache[identifier]

    return _Resolver(lambda identifier: get(Asset, assets, identifier),
                     lambda identifier: get(Attempt, attempts, identifier),
                     lambda identifier: get(Run, runs, identifier))


def model_asset_name(session, asset) -> str:
    """Resolve an existing auto label without trusting names as provenance."""
    generated = _session_resolver(session).resolve(_field(asset, "id"))
    return generated.canonical if generated else _field(asset, "name")


def equivalent_model_names(session, asset_id, left, right) -> bool:
    """Only generated aliases of one verified asset are cosmetic equivalents."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    generated = _session_resolver(session).resolve(asset_id)
    return bool(generated and left in generated.aliases and right in generated.aliases)


def backfill_model_names(connection) -> dict:
    """Idempotent Core migration, running inside the caller's transaction.

    Bulk maps avoid a query per checkpoint or per copy-chain hop. Only the two
    name columns change; request digests and immutable snapshots remain intact.
    """
    from .models import Asset, Attempt, Run
    from .asset_transfer_models import AssetTransfer
    assets = {row["id"]: row for row in connection.execute(select(Asset.__table__).where(
        Asset.__table__.c.kind == "MODEL")).mappings()}
    attempts = {row["id"]: row for row in connection.execute(select(
        Attempt.__table__.c.id, Attempt.__table__.c.run_id,
        Attempt.__table__.c.node_id, Attempt.__table__.c.number)).mappings()}
    runs = {row["id"]: row for row in connection.execute(select(
        Run.__table__.c.id, Run.__table__.c.name, Run.__table__.c.node_id)).mappings()}
    resolver = _Resolver(assets.get, attempts.get, runs.get)
    changes = []
    for asset_id, asset in assets.items():
        generated = resolver.resolve(asset_id)
        if generated and asset["name"] != generated.canonical:
            changes.append({"_id": asset_id, "_name": generated.canonical})
    if changes:
        connection.execute(update(Asset.__table__).where(Asset.__table__.c.id == bindparam("_id"))
                           .values(name=bindparam("_name")), changes)
    transfers = []
    table = AssetTransfer.__table__
    for row in connection.execute(select(table.c.id, table.c.name, table.c.source_asset_id,
            table.c.source_kind, table.c.source_node_id, table.c.source_path).where(table.c.source_kind == "MODEL")).mappings():
        asset = assets.get(row["source_asset_id"])
        if (asset is None or row["source_node_id"] != asset["node_id"] or row["source_path"] != asset["path"]):
            continue
        generated = resolver.resolve(asset["id"])
        if generated:
            aliases = {(name + SYNC_SUFFIX)[:200] for name in generated.aliases}
            name = (generated.canonical + SYNC_SUFFIX)[:200]
            if row["name"] in aliases and row["name"] != name:
                transfers.append({"_id": row["id"], "_name": name})
    if transfers:
        connection.execute(update(table).where(table.c.id == bindparam("_id"))
                           .values(name=bindparam("_name")), transfers)
    return {"assets": len(changes), "asset_transfers": len(transfers)}
