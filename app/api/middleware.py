"""HTTP request guards for public upload endpoints."""

from __future__ import annotations

import json


class RequestBodyLimitMiddleware:
    """Enforce a streaming body-size cap without buffering uploads in RAM."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or not scope.get("path", "").endswith("/documents")):
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value.decode("latin1") for key, value in scope.get("headers", []) if key.lower() == b"content-length"),
            None,
        )
        try:
            if content_length is not None and int(content_length) > self.max_bytes:
                await self._reject(send)
                return
        except ValueError:
            await self._reject(send)
            return

        total = 0
        exceeded = False
        response_started = False
        error_body = json.dumps({"detail": "Upload is too large."}).encode()

        async def limited_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal response_started
            if exceeded:
                if message["type"] == "http.response.start" and not response_started:
                    response_started = True
                    await send({
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(error_body)).encode())],
                    })
                    await send({"type": "http.response.body", "body": error_body})
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if exceeded and not response_started:
                await self._reject(send)
                return
            raise

    async def _reject(self, send):
        body = json.dumps({"detail": "Upload is too large."}).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
