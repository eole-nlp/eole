"""Short-lived signed browser sessions, independent of Eole's inference process."""

import hashlib
import hmac
import secrets
import time

COOKIE_NAME = "eole_session"
SESSION_SECONDS = 12 * 60 * 60


class BrowserSessions:
    def __init__(self):
        # Restarting the gateway invalidates all previous browser sessions.
        self.key = secrets.token_bytes(32)

    def issue(self):
        payload = f"{int(time.time()) + SESSION_SECONDS}.{secrets.token_urlsafe(24)}"
        signature = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{signature}"

    def valid(self, token):
        if len(token) > 256:
            return False
        try:
            expires, nonce, signature = token.split(".")
            payload = f"{expires}.{nonce}"
            expected = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
            return int(expires) > time.time() and secrets.compare_digest(signature, expected)
        except (ValueError, TypeError):
            return False
