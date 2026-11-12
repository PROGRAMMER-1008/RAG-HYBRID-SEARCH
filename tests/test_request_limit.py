from app.api.middleware import RequestBodyLimitMiddleware


def _run_without_event_loop(coroutine):
    """Drive these immediate ASGI fakes without opening a Windows socket pair."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("The fake ASGI stack unexpectedly performed asynchronous I/O.")


def _invoke(body: bytes, *, content_length: int | None, path: str = "/v1/workspaces/id/documents"):
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []
    downstream_called = False

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def app(scope, app_receive, app_send):
        nonlocal downstream_called
        downstream_called = True
        while True:
            message = await app_receive()
            if message["type"] == "http.disconnect" or not message.get("more_body", False):
                break
        await app_send({"type": "http.response.start", "status": 200, "headers": []})
        await app_send({"type": "http.response.body", "body": b"ok"})

    headers = [] if content_length is None else [(b"content-length", str(content_length).encode())]
    scope = {"type": "http", "method": "POST", "path": path, "headers": headers}
    _run_without_event_loop(RequestBodyLimitMiddleware(app, max_bytes=5)(scope, receive, send))
    return sent, downstream_called


def test_streaming_upload_limit_returns_413():
    sent, downstream_called = _invoke(b"123456", content_length=None)

    assert downstream_called
    assert sent[0]["status"] == 413
    assert b"too large" in sent[1]["body"]


def test_content_length_limit_returns_413_before_processing():
    sent, downstream_called = _invoke(b"", content_length=6)

    assert not downstream_called
    assert sent[0]["status"] == 413


def test_non_upload_endpoint_is_not_limited():
    sent, _ = _invoke(b"123456", content_length=6, path="/v1/workspaces")

    assert sent[0]["status"] == 200
