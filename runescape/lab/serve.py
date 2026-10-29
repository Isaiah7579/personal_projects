"""Serve the lab dashboard on http://127.0.0.1:8765 (local only).

    uv run python -m lab.serve [--port 8765]

/                 lab/dashboard/
/api/runs         every run's config + summary + baseline, newest first, and the holdout look count
/runs/<run>/...   files of a run (gens.jsonl, champion.json, evals/...)
/ge.html          the GE-screen replay viewer
/api/replays      replay logs (lab.replay), newest first
/replays/<file>   one replay log
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lab.market import LAB_DIR
from lab.splits import holdout_looks

DASHBOARD = Path(__file__).resolve().parent / "dashboard"
RUNS_DIR = LAB_DIR / "runs"  # = lab.evolve.RUNS_DIR (not imported: that pulls in torch)
REPLAYS_DIR = LAB_DIR / "replays"  # = lab.replay.REPLAYS_DIR


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def runs() -> dict:
    out = []
    for d in sorted(RUNS_DIR.glob("*"), reverse=True) if RUNS_DIR.exists() else []:
        config = read_json(d / "config.json")
        if not config:
            continue
        evals = sorted(p.name for p in (d / "evals").glob("gen_*.json"))
        out.append({"run": d.name, "config": config, "summary": read_json(d / "summary.json"),
                    "baseline": read_json(d / "baseline.json"), "scores": read_json(d / "scores.json"),
                    "evals": evals})
    return {"runs": out, "holdout_looks": holdout_looks()}


def replays() -> list[dict]:
    files = sorted(REPLAYS_DIR.glob("*.json"), reverse=True) if REPLAYS_DIR.exists() else []
    return [{"name": p.name, "bytes": p.stat().st_size} for p in files]


class Handler(SimpleHTTPRequestHandler):
    def send_json(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file_in(self, root: Path, rel: str) -> None:
        target = (root / rel).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            self.send_error(404)
            return
        self.send_json(target.read_bytes())

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/runs":
            self.send_json(json.dumps(runs()).encode())
        elif path == "/api/replays":
            self.send_json(json.dumps(replays()).encode())
        elif path.startswith("/runs/"):
            self.send_file_in(RUNS_DIR, path[len("/runs/"):])
        elif path.startswith("/replays/"):
            self.send_file_in(REPLAYS_DIR, path[len("/replays/"):])
        else:
            super().do_GET()

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(DASHBOARD)))
    print(f"Dashboard: http://127.0.0.1:{args.port}/  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
