"""Explicit human API permissions, shared by every controller route.

Unlisted management endpoints fail closed, including for administrators. Agent
protocol routes have their own enrollment and machine credentials.
"""

from .common import APIError


def request_route_path(request) -> str:
    """Use the matched route rather than the externally prefixed URL path."""
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path
    path = request.scope.get("path", "")
    root = request.scope.get("root_path", "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        path = path[len(root):]
    return path


AUTH_ROUTES = {
    ("GET", "/api/v1/auth/me"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/logout"),
    ("POST", "/api/v1/auth/password"),
    ("POST", "/api/v1/auth/activity"),
    ("POST", "/api/v1/auth/reauth"),
}

PERMISSIONS: dict[tuple[str, str], str] = {}


def _allow(permission, method, *paths):
    for path in paths:
        PERMISSIONS[(method, "/api/v1" + path)] = permission


_allow("platform.read", "GET",
       "/templates", "/overview", "/nodes", "/nodes/{node_id}", "/profiles", "/assets",
       "/nodes/{node_id}/resource-history",
       "/nodes/{node_id}/path-suggestions/{query_id}",
       "/nodes/{node_id}/recipe-suggestions/{query_id}",
       "/nodes/{node_id}/inference-config-suggestions/{query_id}",
       "/nodes/{node_id}/recipe-details/{query_id}",
       "/nodes/{node_id}/data-processors/{query_id}",
       "/recipes", "/recipes/{recipe_id}", "/recipe-revisions/{revision_id}",
       "/recipe-writebacks/{operation_id}",
       "/run-previews/{preview_id}", "/runs", "/runs/{run_id}",
       "/runs/{run_id}/events", "/runs/{run_id}/logs", "/alerts")
_allow("assets.write", "POST", "/assets", "/assets/{asset_id}/archive",
       "/nodes/{node_id}/path-suggestions")
_allow("assets.write", "PATCH", "/assets/{asset_id}/dataset-profile")
_allow("assets.write", "POST", "/nodes/{node_id}/dataset-inspections")
_allow("assets.write", "GET", "/nodes/{node_id}/dataset-inspections/{query_id}")
_allow("training.write", "POST", "/run-previews", "/runs", "/runs/{run_id}/cancel")
_allow("platform.read", "GET", "/policy-deployments", "/policy-deployments/{deployment_id}")
_allow("platform.read", "GET", "/deployment-records", "/deployment-records/{record_id}")
_allow("training.write", "POST", "/policy-deployments", "/policy-deployments/{deployment_id}/refresh",
       "/policy-deployments/{deployment_id}/start",
       "/policy-deployments/{deployment_id}/stop", "/policy-deployments/{deployment_id}/test",
       "/nodes/{node_id}/inference-config-suggestions")
_allow("platform.read", "GET", "/checkpoint-transfers", "/checkpoint-transfers/{transfer_id}")
_allow("training.write", "POST", "/checkpoint-transfers", "/checkpoint-transfers/{transfer_id}/retry",
       "/checkpoint-transfers/{transfer_id}/cancel")
_allow("training.write", "GET", "/training-draft")
_allow("training.write", "PUT", "/training-draft")
_allow("training.write", "DELETE", "/training-draft")
_allow("training.write", "GET", "/training-templates", "/training-templates/{template_id}")
_allow("training.write", "POST", "/training-templates")
_allow("training.write", "PUT", "/training-templates/{template_id}")
_allow("training.write", "DELETE", "/training-templates/{template_id}")
_allow("recipes.write", "POST", "/recipes", "/recipes/{recipe_id}/revisions",
       "/nodes/{node_id}/recipe-suggestions", "/nodes/{node_id}/recipe-details",
       "/nodes/{node_id}/data-processors")
_allow("recipes.writeback", "POST", "/recipe-revisions/{revision_id}/writeback")
_allow("machines.manage", "GET", "/agent-config-defaults", "/agent-package",
       "/nodes/{node_id}/storage-configurations", "/nodes/{node_id}/storage-configurations/{operation_id}",
       "/nodes/{node_id}/profile-rebindings", "/nodes/{node_id}/profile-rebindings/{operation_id}",
       "/nodes/{node_id}/management", "/remote-deployments", "/remote-deployments/{job_id}",
       "/remote-deployments/{job_id}/logs/download")
_allow("machines.manage", "POST", "/agent-configs", "/ssh-connections/test", "/ssh-connections/connect",
       "/nodes/{node_id}/storage-configurations",
       "/nodes/{node_id}/profile-rebindings",
       "/ssh-connections/{connection_id}/path-suggestions", "/ssh-connections/{connection_id}/test",
       "/nodes/{node_id}/disconnect", "/nodes/{node_id}/uninstall",
       "/remote-deployments", "/remote-deployments/{job_id}/retry")
_allow("machines.manage", "PATCH", "/nodes/{node_id}")
_allow("users.manage", "GET", "/users")
_allow("users.manage", "POST", "/users", "/users/{user_id}/reset-password")
_allow("users.manage", "PATCH", "/users/{user_id}")
_allow("audit.read", "GET", "/audit")

# Reauthentication supplements role authorization and never replaces it.
# Directory suggestions and ordinary training/asset writes do not grant
# management access and are not sensitive account/machine mutations. Changing a
# display label does not modify SSH credentials, Agent config or machine access.
SENSITIVE_ROUTES = {
    (method, path) for (method, path), permission in PERMISSIONS.items()
    if method not in {"GET", "HEAD", "OPTIONS"} and (
        permission in {"users.manage", "recipes.writeback"}
        or permission == "machines.manage" and not path.endswith("/path-suggestions")
        and (method, path) != ("PATCH", "/api/v1/nodes/{node_id}"))
}


def permission_for(method: str, route_path: str) -> str | None:
    key = (method, route_path)
    if key in AUTH_ROUTES:
        return None
    permission = PERMISSIONS.get(key)
    if permission is None:
        raise APIError(403, "PERMISSION_NOT_CONFIGURED", "此接口尚未配置访问权限")
    return permission
