"""Small security helpers: outbound URL checks and in-process rate limiting.

Both are deliberately simple. The rate limiter keeps its counters in memory, so the
API runs as one uvicorn process (docker-compose.yml); with N processes the effective
limit would be N times higher. Put Cloudflare or Caddy rate limiting in front for
anything stronger.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

from fastapi import HTTPException, Request

from envforge.web.settings import get_settings

# ---- outbound URLs (SSRF) ----------------------------------------------------


def _allowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    s = get_settings()
    if any(ip in ipaddress.ip_network(c, strict=False) for c in s.BASE_URL_ALLOW_CIDRS):
        return True
    if s.BASE_URL_ALLOW_PRIVATE:
        return not ip.is_unspecified
    return ip.is_global


def check_base_url(url: str) -> str:
    """Validate a user-supplied model endpoint. Returns the normalised URL or raises
    ValueError with a reason the user can act on.

    Called when the key is saved *and* again by the worker right before use, because
    a hostname can resolve differently later (DNS rebinding).
    """
    url = url.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base URL must look like http(s)://host[:port]/v1")
    if parsed.username or parsed.password:
        raise ValueError("put credentials in the key field, not in the URL")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve {parsed.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not _allowed_ip(ip):
            raise ValueError(
                f"{parsed.hostname} resolves to {ip}, which is not allowed. Use a public "
                "host or a Tailscale address (100.64.0.0/10); an admin can widen "
                "BASE_URL_ALLOW_CIDRS."
            )
    return url


# ---- rate limiting -------------------------------------------------------------


class RateLimiter:
    """Sliding-window counter: at most `limit` hits per `window_s` for each key."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: float) -> bool:
        """Record a hit; False if the key is over its limit (the hit is not recorded)."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


def client_ip(request: Request) -> str:
    """The caller's IP. The API is only reachable through the Next.js proxy (and
    Cloudflare in front of it), so the forwarded headers are trustworthy here."""
    for header in ("cf-connecting-ip", "x-real-ip"):
        if v := request.headers.get(header):
            return v.strip()
    if fwd := request.headers.get("x-forwarded-for"):
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce(key: str, limit: int, window_s: float) -> None:
    """Raise 429 when `key` is over its limit."""
    if not limiter.hit(key, limit, window_s):
        raise HTTPException(429, "too many attempts; wait a minute and try again")
