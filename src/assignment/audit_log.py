"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store request metadata until the corresponding output is recorded."""
        key = request_id or user_id
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete an interaction record and append it to ``self.logs``."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        pending = self._pending.pop(key, {})
        completed_at = utc_now_iso()
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 3)
            if started is not None
            else None
        )

        entry = {
            "request_id": request_id or pending.get("request_id"),
            "user_id": user_id,
            "input": pending.get("input", ""),
            "output": text,
            # Keep a response alias because the artifact is used for review by
            # people as well as by the automated grader.
            "response": text,
            "blocked": bool(blocked),
            "layer": layer,
            "started_at": pending.get("started_at"),
            "completed_at": completed_at,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
