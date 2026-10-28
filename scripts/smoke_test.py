"""Smoke test a running service: wait for /ready, then check /health and POST /ask.

    python scripts/smoke_test.py http://localhost:8080 [--wait 180]

Standard library only, so CI can run it without installing the package. Exits non-zero on failure.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def request(url: str, payload: dict | None = None, timeout: float = 60) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body or b"{}")
        except json.JSONDecodeError:
            return exc.code, {"raw": body.decode(errors="replace")[:500]}


def wait_until_ready(base: str, wait_s: float) -> dict:
    deadline = time.monotonic() + wait_s
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = request(f"{base}/ready", timeout=10)
            if status == 200:
                return body
            last_error = f"HTTP {status}: {body}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last_error = repr(exc)
        time.sleep(3)
    raise SystemExit(f"FAIL: /ready did not return 200 within {wait_s:.0f}s (last: {last_error})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_url")
    parser.add_argument("--wait", type=float, default=180, help="seconds to wait for /ready")
    parser.add_argument("--question", default="Who directed Toy Story?")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    ready = wait_until_ready(base, args.wait)
    print(f"ready: {ready}")

    status, body = request(f"{base}/health")
    if status != 200 or body.get("status") != "ok":
        print(f"FAIL: /health returned {status}: {body}", file=sys.stderr)
        return 1
    print("health: ok")

    status, body = request(f"{base}/ask", {"question": args.question})
    if status != 200:
        print(f"FAIL: /ask returned {status}: {body}", file=sys.stderr)
        return 1
    if not body.get("answer") or not body.get("retrieved"):
        print(f"FAIL: /ask response is missing answer or retrieved titles: {body}", file=sys.stderr)
        return 1
    print(f"ask: {body['answer']!r}")
    print(f"     top titles: {[row['title'] for row in body['retrieved']]}")
    print(f"     timings_ms: {body['timings_ms']}")

    status, body = request(f"{base}/ask", {"question": "Who directed Oppenheimer (2023)?"})
    if status != 200 or not body.get("refused"):
        print(f"FAIL: the post-2017 question was not refused: {status} {body}", file=sys.stderr)
        return 1
    print("refusal: ok")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
