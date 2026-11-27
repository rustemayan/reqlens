"""HTTP на стандартной библиотеке: без SDK, чтобы установка не ломалась."""

from __future__ import annotations

import json
import os
import socket
import ssl
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

NETWORK = 0      # нет соединения / DNS / таймаут
TLS = -2         # ошибка проверки сертификата
BAD_BODY = -1    # ответ не JSON


class HTTPError(Exception):
    def __init__(self, status: int, body: str, headers: Optional[Dict[str, str]] = None):
        self.status = status
        self.body = body
        self.headers = headers or {}
        super().__init__(f"HTTP {status}: {body[:500]}")


_CTX: Optional[ssl.SSLContext] = None


def _ssl_context() -> ssl.SSLContext:
    """Системное хранилище сертификатов (работает за корпоративным прокси и антивирусом);
    SSL_CERT_FILE / REQUESTS_CA_BUNDLE — явный CA; certifi — если системное хранилище пустое (macOS)."""
    global _CTX
    if _CTX is not None:
        return _CTX
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile and os.path.isfile(cafile):
        _CTX = ssl.create_default_context(cafile=cafile)
        return _CTX
    ctx = ssl.create_default_context()
    try:
        empty = ctx.cert_store_stats().get("x509_ca", 0) == 0
    except Exception:
        empty = False
    if empty:
        try:
            import certifi  # type: ignore

            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
    _CTX = ctx
    return ctx


def post_json(url: str, payload: dict, headers: Dict[str, str], timeout: float) -> Tuple[dict, Dict[str, str]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    for k, v in headers.items():
        req.add_header(k, v)
    ctx = _ssl_context() if url.startswith("https") else None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        hdrs = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
        raise HTTPError(exc.code, body, hdrs) from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
            raise HTTPError(TLS, f"ошибка проверки TLS-сертификата: {reason}") from None
        raise HTTPError(NETWORK, f"сеть недоступна: {reason}") from None
    except ssl.SSLCertVerificationError as exc:
        raise HTTPError(TLS, f"ошибка проверки TLS-сертификата: {exc}") from None
    except (TimeoutError, socket.timeout) as exc:
        raise HTTPError(NETWORK, f"таймаут {timeout:.0f} с: {exc}") from None
    except (ssl.SSLError, ConnectionError, OSError) as exc:
        raise HTTPError(NETWORK, f"{type(exc).__name__}: {exc}") from None
    try:
        return json.loads(body), hdrs
    except json.JSONDecodeError:
        raise HTTPError(BAD_BODY, f"ответ API не JSON: {body[:300]}")
