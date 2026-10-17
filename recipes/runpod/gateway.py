"""Authenticated HTTP/SSE gateway while a private Eole process is prepared."""

import asyncio
import base64
import binascii
from html import escape
from contextlib import asynccontextmanager, suppress
import secrets
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse, RedirectResponse
from starlette.background import BackgroundTask

try:
    from .auth import BrowserSessions, COOKIE_NAME, SESSION_SECONDS
except ImportError:
    from auth import BrowserSessions, COOKIE_NAME, SESSION_SECONDS

HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


SSE_HEARTBEAT_SECONDS = 15


async def response_body(upstream):
    """Keep a long compilation/prefill from leaving the public SSE proxy idle."""
    sse = "text/event-stream" in upstream.headers.get("content-type", "")
    iterator = (upstream.aiter_bytes() if sse else upstream.aiter_raw()).__aiter__()
    pending = None
    try:
        if not sse:
            async for chunk in iterator:
                yield chunk
            return
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=SSE_HEARTBEAT_SECONDS)
            if not done:
                yield b": eole keepalive\n\n"
                continue
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                return
            pending = None
            yield chunk
    finally:
        with anyio.CancelScope(shield=True):
            if pending is not None:
                pending.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await pending
            await upstream.aclose()


def credentials(username, password):
    if not username or ":" in username:
        raise ValueError("EOLE_USERNAME must be nonempty and cannot contain a colon")
    if "RUNPOD_SECRET" in password:
        raise ValueError("EOLE_PASSWORD contains an unresolved RunPod Secret reference; check its binding")
    if len(password) < 16:
        raise ValueError("EOLE_PASSWORD must contain at least 16 characters; update its Secret and reset the Pod")
    return username.encode(), password.encode()


def authorized(header, username, password):
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer":
        return secrets.compare_digest(value.encode(), password)
    if scheme.lower() == "basic":
        try:
            user, separator, supplied = base64.b64decode(value, validate=True).partition(b":")
            return bool(separator) and secrets.compare_digest(user, username) & secrets.compare_digest(
                supplied, password
            )
        except (ValueError, binascii.Error):
            pass
    return False


def forwarding_headers(headers):
    # Connection can nominate additional hop-by-hop headers.
    excluded = HOP_HEADERS | {v.strip().lower() for v in headers.get("connection", "").split(",")}
    items = headers.multi_items() if hasattr(headers, "multi_items") else headers.items()
    return [(k, v) for k, v in items if k.lower() not in excluded]


def create_gateway(username, password, state, prepare=None, transport=None, cookie_secure=True, public_origin=None):
    username, password = credentials(username, password)
    sessions = BrowserSessions()

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:5001", timeout=httpx.Timeout(30, read=None), trust_env=False, transport=transport
        ) as client:
            app.state.client = client
            async with prepare() if prepare else empty_context():
                yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    page = (Path(__file__).parent / "preparing.html").read_text()

    login_page = (
        (Path(__file__).parent / "login.html")
        .read_text()
        .replace("EOLE_USERNAME_VALUE", escape(username.decode(), quote=True))
    )

    def login_form(error="", status_code=200):
        return HTMLResponse(login_page.replace("EOLE_LOGIN_ERROR", error), status_code=status_code)

    async def login(request):
        if not request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
            return login_form("Invalid sign-in request.", 400)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                return login_form("Invalid sign-in request.", 413)
        try:
            form = parse_qs(body.decode(), max_num_fields=2)
            user, supplied = form.get("username", [""])[0], form.get("password", [""])[0]
        except (UnicodeError, ValueError):
            return login_form("Invalid sign-in request.", 400)
        if not (secrets.compare_digest(user.encode(), username) & secrets.compare_digest(supplied.encode(), password)):
            return login_form("Incorrect username or password.", 401)
        response = (
            JSONResponse({"authenticated": True})
            if "application/json" in request.headers.get("accept", "")
            else RedirectResponse("/", status_code=303)
        )
        response.set_cookie(
            COOKIE_NAME,
            sessions.issue(),
            max_age=SESSION_SECONDS,
            httponly=True,
            secure=cookie_secure,
            samesite="strict",
            path="/",
        )
        return response

    async def authenticated(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if request.headers.get("sec-fetch-site") == "cross-site" or (
                origin
                and (
                    origin.rstrip("/") != public_origin
                    if public_origin
                    else urlsplit(origin).netloc.lower() != request.headers.get("host", "").lower()
                )
            ):
                return Response("Cross-origin requests are not allowed", 403)
        if request.url.path == "/_deployment/login" and request.method == "POST":
            return await login(request)
        owner = authorized(request.headers.get("authorization", ""), username, password) or sessions.valid(
            request.cookies.get(COOKIE_NAME, "")
        )
        if not owner:
            if request.method in {"GET", "HEAD"} and request.url.path in {"/", "/_deployment/login"}:
                return login_form()
            return Response("Authentication required", 401, headers={"WWW-Authenticate": 'Bearer realm="Eole"'})
        if request.url.path == "/_deployment/logout" and request.method == "POST":
            response = RedirectResponse("/", status_code=303)
            response.delete_cookie(COOKIE_NAME, path="/", secure=cookie_secure, httponly=True, samesite="strict")
            return response
        return await call_next(request)

    @app.middleware("http")
    async def authenticate(request, call_next):
        response = await authenticated(request, call_next)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def route(request: Request, path: str):
        if request.url.path == "/_deployment/status":
            return JSONResponse(dict(state))
        if state["stage"] != "ready":
            if request.method in ("GET", "HEAD") and request.url.path == "/":
                return HTMLResponse(page)
            return JSONResponse({"error": "Eole is not ready", **state}, status_code=503, headers={"Retry-After": "5"})
        client = app.state.client
        url = client.base_url.copy_with(path=request.url.path, query=request.url.query.encode())
        headers = [
            (k, v)
            for k, v in forwarding_headers(request.headers)
            if k.lower()
            not in {
                "authorization",
                "host",
                "cookie",
                "forwarded",
                "x-forwarded-for",
                "x-forwarded-host",
                "x-forwarded-proto",
            }
        ]
        try:
            upstream = await client.send(
                client.build_request(request.method, url, headers=headers, content=request.stream()), stream=True
            )
        except httpx.RequestError:
            return JSONResponse({"error": "Eole backend is unavailable"}, status_code=502)

        response = StreamingResponse(
            response_body(upstream), status_code=upstream.status_code, background=BackgroundTask(upstream.aclose)
        )
        headers = forwarding_headers(upstream.headers)
        if "text/event-stream" in upstream.headers.get("content-type", ""):
            headers = [(k, v) for k, v in headers if k.lower() not in {"content-length", "content-encoding"}]
        # Keep redirects on the authenticated public origin, never loopback.
        headers = [
            (
                (key, value.removeprefix(str(client.base_url).rstrip("/")))
                if key.lower() == "location" and value.startswith(str(client.base_url))
                else (key, value)
            )
            for key, value in headers
        ]
        response.raw_headers = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers]
        return response

    return app


@asynccontextmanager
async def empty_context():
    yield
