"""Server pagination over visible training and registered-asset copy records."""

from typing import Literal

from fastapi import Query
from sqlalchemy import false, func, literal, select, union_all
from sqlalchemy.orm import aliased, defer

from .asset_transfer_models import AssetTransfer
from .asset_transfers import asset_transfer_dict
from .models import Node, Run
from .serialization import run_dict


TaskType = Literal["ALL", "training", "data-sync"]
TaskStatus = Literal["ALL", "ACTIVE", "SUCCEEDED", "FAILED", "CANCELLED"]
ACTIVE_TRAINING = ("QUEUED", "STARTING", "RUNNING", "CANCELLING", "UNKNOWN")
ACTIVE_SYNC = ("QUEUED", "PREPARING", "SCANNING", "TRANSFERRING", "VERIFYING", "CANCELLING")


def _search(fields, query):
    content = func.coalesce(fields[0], "")
    for field in fields[1:]:
        content = content + literal(" ") + func.coalesce(field, "")
    # Search text is a literal substring, including user-entered '%' and '_'.
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return content.ilike("%" + escaped + "%", escape="\\")


def _task_queries(task_type="ALL", status="ALL", search=""):
    runs = select(Run.id.label("id"), literal("training").label("kind"),
                  Run.created_at.label("created_at")).where(Run.deleted_at.is_(None))
    copies = select(AssetTransfer.id.label("id"), literal("data-sync").label("kind"),
                    AssetTransfer.created_at.label("created_at")).where(AssetTransfer.deleted_at.is_(None))
    if task_type == "training":
        copies = copies.where(false())
    elif task_type == "data-sync":
        runs = runs.where(false())
    if status == "ACTIVE":
        runs = runs.where(Run.state.in_(ACTIVE_TRAINING))
        copies = copies.where(AssetTransfer.state.in_(ACTIVE_SYNC))
    elif status == "FAILED":
        runs = runs.where(Run.state.in_(("FAILED", "INTERRUPTED")))
        copies = copies.where(AssetTransfer.state.in_(("FAILED", "INTERRUPTED")))
    elif status != "ALL":
        runs = runs.where(Run.state == status)
        copies = copies.where(AssetTransfer.state == status)
    if search:
        node, source, target = aliased(Node), aliased(Node), aliased(Node)
        runs = runs.outerjoin(node, node.id == Run.node_id).where(_search([
            Run.name, Run.id, Run.node_id, func.coalesce(node.display_name, node.name),
            node.hostname, literal("模型训练")], search))
        copies = copies.outerjoin(source, source.id == AssetTransfer.source_node_id).outerjoin(
            target, target.id == AssetTransfer.target_node_id).where(_search([
                AssetTransfer.name, AssetTransfer.id, AssetTransfer.source_node_id,
                func.coalesce(source.display_name, source.name), source.hostname,
                AssetTransfer.target_node_id, func.coalesce(target.display_name, target.name), target.hostname,
                AssetTransfer.source_path, AssetTransfer.target_path, AssetTransfer.target_directory,
                literal("数据同步")], search))
    return runs, copies


def task_page(session, *, page=1, page_size=20, task_type="ALL", status="ALL", query=""):
    visible = union_all(*_task_queries()).subquery()
    total_all = session.scalar(select(func.count()).select_from(visible))
    filtered = union_all(*_task_queries(task_type, status, query.strip())).subquery()
    total = session.scalar(select(func.count()).select_from(filtered))
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    rows = list(session.execute(select(filtered).order_by(
        filtered.c.created_at.desc(), filtered.c.id.desc(), filtered.c.kind.desc())
        .offset((page - 1) * page_size).limit(page_size)))
    run_ids = [row.id for row in rows if row.kind == "training"]
    copy_ids = [row.id for row in rows if row.kind == "data-sync"]
    # Fetch large configuration/asset details only for the selected page. The
    # transfer serializer needs neither the file manifest nor SSH bindings.
    runs = {row.id: row for row in session.scalars(select(Run).where(
        Run.id.in_(run_ids), Run.deleted_at.is_(None)))} if run_ids else {}
    copies = {row.id: row for row in session.scalars(select(AssetTransfer).options(
        defer(AssetTransfer.manifest), defer(AssetTransfer.binding_snapshot)).where(
            AssetTransfer.id.in_(copy_ids), AssetTransfer.deleted_at.is_(None)))} if copy_ids else {}
    items = []
    for row in rows:
        item = runs.get(row.id) if row.kind == "training" else copies.get(row.id)
        # A concurrent deletion can disappear between the ID and detail reads.
        if item is not None:
            items.append({"key": f"{row.kind}:{row.id}", "kind": row.kind,
                          "item": run_dict(session, item) if row.kind == "training" else asset_transfer_dict(item)})
    return {"items": items, "total": total, "total_all": total_all, "page": page,
            "page_size": page_size, "total_pages": total_pages}


def register_routes(app, admin, db):
    @app.get("/api/v1/background-tasks", dependencies=admin)
    def background_tasks(page: int = Query(default=1, ge=1),
                         page_size: int = Query(default=20, ge=1, le=100),
                         type: TaskType = Query(default="ALL"),
                         status: TaskStatus = Query(default="ALL"),
                         q: str | None = Query(default=None, max_length=200)):
        with db.read() as session:
            return task_page(session, page=page, page_size=page_size, task_type=type,
                             status=status, query=q or "")
