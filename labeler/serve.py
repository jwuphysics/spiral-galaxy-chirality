#!/usr/bin/env python3
"""Tiny stdlib server for the spiral-chirality labeler.

Serves the project root (so /labeler/index.html and /data/images/*.jpg are
both reachable) and accepts POSTs of label JSON to /labeler/<name>.json,
written atomically (tempfile + os.replace).

Usage:  python3 labeler/serve.py [--port 8801] [--root /path/to/project]
"""
import argparse
import functools
import json
import os
import re
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

POST_PATH = re.compile(r"^/labeler/([A-Za-z0-9._-]+)\.json$")


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        # Labels and manifest must never be served stale.
        if self.path.split("?", 1)[0].endswith((".json", ".js")):
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_POST(self):
        m = POST_PATH.match(self.path.split("?", 1)[0])
        if not m:
            self.send_error(403, "POST is allowed only to /labeler/<name>.json")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            json.loads(body.decode("utf-8"))          # validate before writing
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "Request body must be valid JSON")
            return
        dest_dir = os.path.join(self.directory, "labeler")
        os.makedirs(dest_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest_dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, os.path.join(dest_dir, m.group(1) + ".json"))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_request(self, code="-", size="-"):
        pass  # keep the terminal quiet during labeling; errors still print


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--root", default=os.path.dirname(here),
                    help="directory to serve (default: project root)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(Handler, directory=root))
    print(f"serving  {root}")
    print(f"labeler  http://localhost:{args.port}/labeler/")
    print("labels are saved to labeler/labels.json on every keypress; Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
