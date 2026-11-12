"""Short-lived, bearer-token-protected document workspaces."""

from __future__ import annotations

import hashlib
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.ingestion.loader import RawDocument
from app.retrieval.workspace import WorkspacePipeline


@dataclass
class Workspace:
    workspace_id: str
    token_digest: str
    root: Path
    created_at: float
    expires_at: float
    documents: list[RawDocument] = field(default_factory=list)
    pipeline: WorkspacePipeline | None = None
    ask_count: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class WorkspaceStore:
    """Process-local sessions; uploaded files are erased on expiry or restart."""

    def __init__(self, root: Path | None = None):
        self.root = root or settings.workspace_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.RLock()
        self._request_minute = 0
        self._request_count = 0
        self._daily_date = ""
        self._daily_asks = 0
        # An interrupted upload can leave a temporary file behind; no
        # workspace survives a process restart, so remove only our own
        # direct child directories before accepting new sessions.
        for path in self.root.glob("workspace-*"):
            if path.is_dir() and not path.is_symlink() and path.parent.resolve() == self.root.resolve():
                self._delete_root(path)

    def create(self) -> tuple[Workspace, str]:
        self.cleanup_expired()
        with self._lock:
            if len(self._workspaces) >= settings.max_active_workspaces:
                raise RuntimeError("The demo is busy. Please try again later.")
            workspace_id = secrets.token_urlsafe(18)
            token = secrets.token_urlsafe(32)
            directory = Path(tempfile.mkdtemp(prefix="workspace-", dir=self.root))
            now = time.time()
            workspace = Workspace(
                workspace_id=workspace_id,
                token_digest=hashlib.sha256(token.encode("utf-8")).hexdigest(),
                root=directory,
                created_at=now,
                expires_at=now + settings.workspace_ttl_minutes * 60,
            )
            self._workspaces[workspace_id] = workspace
            return workspace, token

    def close(self) -> None:
        with self._lock:
            workspaces = list(self._workspaces.values())
            self._workspaces.clear()
        for workspace in workspaces:
            with workspace.lock:
                self._delete_root(workspace.root)

    def authorize(self, workspace_id: str, bearer_token: str | None) -> Workspace:
        self.cleanup_expired()
        if not bearer_token:
            raise PermissionError("A workspace bearer token is required.")
        candidate = hashlib.sha256(bearer_token.encode("utf-8")).hexdigest()
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None:
                raise LookupError("Workspace not found or expired.")
            if not secrets.compare_digest(candidate, workspace.token_digest):
                raise PermissionError("Workspace token is invalid.")
            workspace.expires_at = time.time() + settings.workspace_ttl_minutes * 60
            return workspace

    def remove(self, workspace_id: str, bearer_token: str | None) -> None:
        workspace = self.authorize(workspace_id, bearer_token)
        with workspace.lock:
            with self._lock:
                removed = self._workspaces.pop(workspace_id, None)
            if removed is None:
                raise LookupError("Workspace not found or expired.")
            self._delete_root(workspace.root)

    def cleanup_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [
                workspace_id
                for workspace_id, workspace in self._workspaces.items()
                if workspace.expires_at <= now
            ]
            workspaces = [self._workspaces.pop(workspace_id) for workspace_id in expired]
        for workspace in workspaces:
            with workspace.lock:
                self._delete_root(workspace.root)
        return len(workspaces)

    def allow_request(self) -> bool:
        now_minute = int(time.time() // 60)
        with self._lock:
            if now_minute != self._request_minute:
                self._request_minute = now_minute
                self._request_count = 0
            if self._request_count >= settings.max_requests_per_minute:
                return False
            self._request_count += 1
            return True

    def allow_question(self, workspace: Workspace) -> bool:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        with self._lock:
            if self._daily_date != today:
                self._daily_date = today
                self._daily_asks = 0
            if self._daily_asks >= settings.max_daily_questions:
                return False
            if workspace.ask_count >= settings.max_questions_per_workspace:
                return False
            self._daily_asks += 1
            workspace.ask_count += 1
            return True

    def _delete_root(self, path: Path) -> None:
        # All paths originate from mkdtemp under self.root; refuse deletion if
        # the path was altered or escaped the managed workspace directory.
        try:
            resolved_root = self.root.resolve()
            resolved_path = path.resolve()
            if resolved_path.parent == resolved_root and resolved_path.name.startswith("workspace-"):
                shutil.rmtree(resolved_path, ignore_errors=True)
        except OSError:
            pass
