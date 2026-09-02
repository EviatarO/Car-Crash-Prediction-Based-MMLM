"""
serve.py - local static server for the dataset-review website.

WHY NOT `python -m http.server`: its handler does not implement HTTP Range requests,
and without `206 Partial Content` the browser cannot seek inside the mp4s - which is the
whole point of the site (jump to time_of_event - 5 s). This handler adds single-range
support; everything else is stdlib SimpleHTTPRequestHandler behavior.

ROOT: served from the Thesis/ parent directory (two levels above MMLM_AI), because the
raw Nexar mp4s live in the SIBLING project Data-Centric-Crash-Prediction-Using-3LC-and-
MViT/src/Nexar_DataSet/ and a server can only expose paths under its root. Local only -
binds 127.0.0.1, nothing is reachable from other machines.

    python website/serve.py          (then open http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/index.html)
"""
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8765
ROOT = Path(__file__).resolve().parents[3]     # .../PycharmProjects/Thesis
SITE_URL = ("http://localhost:%d/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/"
            "website/index.html" % PORT)

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        """Never let the browser cache anything from this server.

        This is a local authoring server: the pages, the CSS and the generated
        *_data.js files are edited and rebuilt constantly, and a cached copy shows
        stale results that look exactly like a bug in the site. Without this, the
        only fix is a hard refresh, and it is not obvious that one is needed - the
        page renders fine, just with yesterday's numbers or last week's stylesheet.
        Bandwidth is irrelevant over loopback, so correctness wins outright.
        """
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def send_head(self):
        """SimpleHTTPRequestHandler.send_head, plus single-range 206 responses."""
        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        m = RANGE_RE.match(range_header)
        if not m:
            f.close()
            self.send_error(400, "Bad Range header")
            return None
        start_s, end_s = m.groups()
        if start_s == "":                       # suffix form: bytes=-N (last N bytes)
            length = int(end_s)
            start, end = max(0, size - length), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            f.close()
            self.send_response(416)             # Range Not Satisfiable
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        self._range_remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        self._range_remaining = None
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        # mp4 seeking works without caching games, but thumbnails benefit: they never
        # change between rebuilds of site_data.js, so let the browser keep them a bit.
        if self.path.endswith((".jpg", ".png")):
            self.send_header("Cache-Control", "max-age=3600")
        super().end_headers()

    def log_message(self, fmt, *args):          # quiet: one line per request is noise
        pass


def main():
    os.chdir(ROOT)
    handler = partial(RangeHandler, directory=str(ROOT))
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    print(f"serving {ROOT}")
    print(f"open    {SITE_URL}")
    print("Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
