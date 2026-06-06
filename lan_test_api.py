#!/usr/bin/env python3
"""Tiny LAN-visible test API using only the Python standard library."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


HOST = "0.0.0.0"
PORT = 8765


def local_hostname() -> str:
    return socket.gethostname()


class Handler(BaseHTTPRequestHandler):
    server_version = "LanTestAPI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/health", "/ping"}:
            self.write_json(
                {
                    "ok": True,
                    "service": "lan-test-api",
                    "host": local_hostname(),
                    "path": parsed.path,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "client": self.client_address[0],
                }
            )
            return

        if parsed.path == "/echo":
            query = parse_qs(parsed.query)
            self.write_json(
                {
                    "ok": True,
                    "echo": query.get("msg", [""])[0],
                    "client": self.client_address[0],
                }
            )
            return

        self.write_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {fmt % args}", flush=True)

    def write_json(self, payload: dict[str, object], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serving LAN test API on http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
