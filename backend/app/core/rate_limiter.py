import time
import threading
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("backend.app.core.rate_limiter")


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Thread-safe, in-memory sliding-window rate limiting middleware.

    Restricts request frequency on ingestion, prediction, and authentication
    endpoints per JWT client token or client IP address.

    Two limits apply:
      - ``limit``      (default 60/min) for upload and explain paths.
      - ``auth_limit`` (default 10/min) for the login endpoint — prevents
        credential brute-force attacks.

    Stale keys are evicted lazily to prevent unbounded memory growth.
    """

    def __init__(
        self,
        app,
        limit: int = 60,
        window_seconds: int = 60,
        auth_limit: int = 10,
    ):
        super().__init__(app)
        self.limit = limit
        self.auth_limit = auth_limit
        self.window_seconds = window_seconds
        self.requests: dict = {}
        self.lock = threading.Lock()
        self._request_counter = 0          # Used to trigger periodic eviction sweeps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client_key(self, request: Request) -> str:
        """
        Resolve the rate-limit identity key.
        Prefer the raw JWT bearer token so each session is tracked
        independently; fall back to client IP.
        """
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ", 1)[1]
                if token:
                    return token
            except IndexError:          # FIXED: was IndexErr (NameError) — now correct
                pass
        client = request.client
        return client.host if client else "unknown"

    def _evict_stale_keys(self, now: float) -> None:
        """
        Remove keys whose entire history has expired outside the window.
        Called periodically (every 500 requests) to bound memory usage.
        """
        stale = [k for k, ts_list in self.requests.items()
                 if not any(now - t < self.window_seconds for t in ts_list)]
        for k in stale:
            del self.requests[k]
        if stale:
            logger.debug(f"Rate limiter evicted {len(stale)} stale client key(s).")

    def _is_rate_limited(self, client_key: str, now: float, effective_limit: int) -> bool:
        """
        Sliding-window check. Returns True if the client exceeds the limit.
        Evicts the current key's stale timestamps regardless of whether a
        periodic sweep is due.
        """
        history = self.requests.get(client_key, [])
        # Evict timestamps outside the active window
        history = [t for t in history if now - t < self.window_seconds]

        if len(history) >= effective_limit:
            return True

        history.append(now)
        self.requests[client_key] = history
        return False

    # ------------------------------------------------------------------
    # Middleware dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Determine which paths to protect and which limit applies
        is_upload_or_explain = (
            path.startswith("/api/v1/upload") or "/explain" in path
        )
        is_auth = path == "/api/v1/auth/login"

        if is_upload_or_explain or is_auth:
            client_key = self._get_client_key(request)
            effective_limit = self.auth_limit if is_auth else self.limit
            now = time.time()

            with self.lock:
                # Periodic stale-key eviction to bound memory growth (every 500 requests)
                self._request_counter += 1
                if self._request_counter % 500 == 0:
                    self._evict_stale_keys(now)

                if self._is_rate_limited(client_key, now, effective_limit):
                    logger.warning(
                        f"Rate limit exceeded for client {client_key[:16]}... "
                        f"on path {path} (limit={effective_limit}/min)"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"detail": "Too many requests. Please try again later."},
                    )

        return await call_next(request)
