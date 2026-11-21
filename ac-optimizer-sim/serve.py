#!/usr/bin/env python3
"""Host the AC Optimizer dashboard on a web server. No extra packages needed.

  python serve.py                 -> http://localhost:8000  (this computer only)
  python serve.py --public        -> reachable from other devices on your network
  python serve.py --port 5000

The dashboard runs the simulation in the visitor's browser, so every slider
change shows its result immediately and the server stays tiny.
"""
import argparse
import functools
import http.server
import socket
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("", "/"):
            self.path = "/dashboard.html"
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "your-computer-ip"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--public", action="store_true", help="listen on all network interfaces")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    host = "0.0.0.0" if a.public else "127.0.0.1"
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer((host, a.port), handler) as srv:
        print(f"Dashboard running at http://localhost:{a.port}")
        if a.public:
            print(f"On your network:      http://{lan_ip()}:{a.port}")
        print("Press Ctrl+C to stop.")
        if not a.no_browser:
            webbrowser.open(f"http://localhost:{a.port}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
