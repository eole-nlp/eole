"""Serialize API inference with workspace model switching when requested."""


class ModelLifecycleMiddleware:
    """Hold the model lease until the entire streaming response has drained.

    Workspace jobs acquire this same lock themselves, after publishing their
    cancellable queued request. State, cancellation and static assets stay live.
    """

    def __init__(self, app, server):
        self.app = app
        self.server = server

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "").rstrip("/")
        api_path = path.removeprefix("/anthropic") if path.startswith("/anthropic/v1/") else path
        guarded = api_path in ("/infer", "/unload_model", "/openai/chat/completions") or (
            api_path.startswith("/v1/") and api_path != "/v1/models"
        )
        if scope["type"] == "http" and getattr(self.server, "model_policy", "keep_loaded") == "single" and guarded:
            async with self.server.model_lock:
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)
