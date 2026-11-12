import time

import pytest

from app.api.workspaces import WorkspaceStore
from app.config import settings


def test_workspace_token_is_required_and_only_hash_is_retained(tmp_path):
    store = WorkspaceStore(root=tmp_path)
    workspace, token = store.create()
    try:
        assert token not in workspace.token_digest
        assert store.authorize(workspace.workspace_id, token) is workspace
        with pytest.raises(PermissionError):
            store.authorize(workspace.workspace_id, "wrong-token")
    finally:
        store.close()


def test_expired_workspace_is_removed(tmp_path):
    store = WorkspaceStore(root=tmp_path)
    workspace, token = store.create()
    workspace.expires_at = time.time() - 1

    assert store.cleanup_expired() == 1
    assert not workspace.root.exists()
    with pytest.raises(LookupError):
        store.authorize(workspace.workspace_id, token)
    store.close()


def test_process_rate_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_requests_per_minute", 1)
    store = WorkspaceStore(root=tmp_path)
    try:
        assert store.allow_request()
        assert not store.allow_request()
    finally:
        store.close()
