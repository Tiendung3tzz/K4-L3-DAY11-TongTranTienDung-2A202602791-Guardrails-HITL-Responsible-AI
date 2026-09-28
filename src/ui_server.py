"""Local web UI API backed by the real Blue agent and CP3 guardrail pipeline.

Run from the repository root with ``python src/ui_server.py``. The server
binds to loopback only and serves an explicit UI/API allowlist; it does not
expose repository files such as ``.env`` or protected demo data.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.agent import create_blue_agent  # noqa: E402
from assignment.audit_log import AuditLogPlugin  # noqa: E402
from assignment.monitoring import MonitoringAlert  # noqa: E402
from assignment.pipeline import build_production_plugins  # noqa: E402
from core.config import (  # noqa: E402
    DEMO_SECRETS,
    blue_provider_label,
    get_openrouter_api_key,
)
from guardrails.output_guardrails import content_filter  # noqa: E402

MAX_MESSAGE_LENGTH = 8_000
MAX_AUDIT_ROWS = 500
UI_FILES = {
    "/ui/": ROOT / "ui" / "index.html",
    "/ui/index.html": ROOT / "ui" / "index.html",
    "/ui/app.js": ROOT / "ui" / "app.js",
    "/ui/styles.css": ROOT / "ui" / "styles.css",
}

plugins = build_production_plugins(max_requests=10, window_seconds=60)
rate_limiter, input_guardrail, output_guardrail = plugins
agent, runner = create_blue_agent(plugins)
audit = AuditLogPlugin()
monitor = MonitoringAlert()
request_lock = threading.Lock()


def _safe_for_audit(text: str) -> str:
    """Avoid persisting common PII and the lab's synthetic secrets verbatim."""
    safe_text = content_filter(text)["redacted"]
    for secret in DEMO_SECRETS:
        if secret:
            safe_text = re.sub(re.escape(secret), "[REDACTED]", safe_text, flags=re.I)
    return safe_text[:MAX_MESSAGE_LENGTH]


def _dashboard_payload() -> dict:
    alerts = monitor.check_metrics()
    snapshot = monitor.snapshot()
    snapshot["alerts"] = [
        {
            "metric": alert.metric,
            "value": alert.value,
            "threshold": alert.threshold,
            "message": alert.message,
        }
        for alert in alerts
    ]
    return {
        "metrics": snapshot,
        "audit": audit.logs[-8:],
        "backend": {
            "ready": bool(get_openrouter_api_key()),
            "provider": blue_provider_label(),
        },
    }


def _save_observability() -> None:
    audit.logs[:] = audit.logs[-MAX_AUDIT_ROWS:]
    try:
        audit.export_json(str(ROOT / "outputs" / "ui_audit_log.json"))
        monitor.export_json(str(ROOT / "outputs" / "ui_metrics.json"))
    except OSError:
        # Chat delivery should not fail just because the local artifact folder
        # is temporarily unavailable; the in-memory activity remains visible.
        pass


def _run_chat(message: str, user_id: str) -> dict:
    """Run the production plugin order and return only the guarded answer."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    rate_before = rate_limiter.blocked_count
    input_before = input_guardrail.blocked_count
    redacted_before = output_guardrail.redacted_count
    audit.record_input(
        user_id=user_id,
        text=_safe_for_audit(message),
        request_id=request_id,
    )
    monitor.total_requests += 1

    if not get_openrouter_api_key():
        reply = "Backend chưa được cấu hình OPENROUTER_API_KEY. Hãy thêm API key vào file .env rồi khởi động lại server."
        audit.record_output(
            user_id=user_id,
            text=reply,
            blocked=False,
            layer="configuration",
            request_id=request_id,
        )
        _save_observability()
        return {
            "request_id": request_id,
            "reply": reply,
            "blocked": False,
            "redacted": False,
            "layer": "configuration",
            "decision": "error",
            "provider": blue_provider_label(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "http_status": HTTPStatus.SERVICE_UNAVAILABLE,
        }

    try:
        reply = asyncio.run(runner.chat(agent, message, user_id=user_id))
    except Exception:
        # Provider/library exceptions can contain request metadata; keep those
        # details in the server console only, never return them to the browser.
        print("Blue agent request failed; check provider configuration/logs.", file=sys.stderr)
        reply = "Hiện chưa thể kết nối trợ lý. Vui lòng thử lại sau ít phút."
        audit.record_output(
            user_id=user_id,
            text=reply,
            blocked=False,
            layer="model_error",
            request_id=request_id,
        )
        _save_observability()
        return {
            "request_id": request_id,
            "reply": reply,
            "blocked": False,
            "redacted": False,
            "layer": "model_error",
            "decision": "error",
            "provider": blue_provider_label(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "http_status": HTTPStatus.BAD_GATEWAY,
        }

    rate_limited = rate_limiter.blocked_count > rate_before
    input_blocked = input_guardrail.blocked_count > input_before
    redacted = output_guardrail.redacted_count > redacted_before
    blocked = rate_limited or input_blocked
    layer = (
        "rate_limiter" if rate_limited else
        "input_guardrail" if input_blocked else
        "output_guardrail" if redacted else None
    )
    if blocked:
        monitor.blocked_requests += 1
    if rate_limited:
        monitor.rate_limit_hits += 1

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    audit.record_output(
        user_id=user_id,
        text=_safe_for_audit(reply),
        blocked=blocked,
        layer=layer,
        request_id=request_id,
    )
    _save_observability()
    return {
        "request_id": request_id,
        "reply": reply,
        "blocked": blocked,
        "redacted": redacted,
        "layer": layer,
        "decision": "blocked" if blocked else "redacted" if redacted else "allowed",
        "provider": blue_provider_label(),
        "latency_ms": latency_ms,
        "http_status": HTTPStatus.OK,
    }


class UIRequestHandler(BaseHTTPRequestHandler):
    server_version = "VinBankDemo/1.0"

    def log_message(self, format: str, *args) -> None:
        # Keep request paths/status visible without logging message bodies.
        super().log_message(format, *args)

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/dashboard":
            self._send_json(_dashboard_payload())
            return
        file_path = UI_FILES.get(path)
        if file_path is None or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Content-Length không hợp lệ."}, HTTPStatus.BAD_REQUEST)
            return
        if content_length <= 0 or content_length > MAX_MESSAGE_LENGTH * 4:
            self._send_json({"error": "Request rỗng hoặc vượt giới hạn."}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "JSON không hợp lệ."}, HTTPStatus.BAD_REQUEST)
            return
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str) or not message.strip():
            self._send_json({"error": "Hãy nhập nội dung câu hỏi."}, HTTPStatus.BAD_REQUEST)
            return
        if len(message) > MAX_MESSAGE_LENGTH:
            self._send_json({"error": f"Câu hỏi tối đa {MAX_MESSAGE_LENGTH} ký tự."}, HTTPStatus.BAD_REQUEST)
            return

        # The server is loopback-only. IP-based ID is adequate for this local
        # lab UI; the plugin still enforces its configured rolling window.
        user_id = self.client_address[0]
        with request_lock:
            result = _run_chat(message.strip(), user_id=user_id)
        status = result.pop("http_status")
        self._send_json(result, status)


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    server = HTTPServer((host, port), UIRequestHandler)
    print(f"VinBank Guardrails UI: http://{host}:{port}/ui/")
    print("Local demo only — browser requests use the Blue model/provider configured in .env.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VinBank UI server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
