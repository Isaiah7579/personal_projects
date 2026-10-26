"""Background sync that keeps the price cache filled, gently, forever.

Meant to start at logon (see install_sync_task.ps1) and never stop. Every API
request counts against a per-request interval and a daily budget, both kept
across restarts in data/sync_state.json. Work is done in priority order:

1. Strategy archive (data/5m, items >= 1m, what the backtests read): every 5m
   bucket the API still serves that isn't cached, newest first. In steady
   state this is just the newest bucket every 5 minutes (288 requests/day);
   after the PC was off it's the catch-up, and it also fills old holes before
   they age out of the API's ~90-day retention.
2. Full archive (data/5m-all, every item): the same buckets with nothing
   dropped, oldest first since those expire soonest. The /5m endpoint returns
   every item for a timestamp in one request, so this is how every item,
   whatever its price or volume, gets full history. (Per-item /timeseries
   history isn't fetched: past 24h it's coarser, and it only reaches back
   as far as the 5m buckets do.)

Tier 2 stops for the day DAILY_RESERVE requests short of the budget so tier 1
can always keep up. The item mapping (buy limits) is refreshed daily.

Usage:
    uv run sync_cache.py              # run with a system tray icon (sync_tray.py): status, pause, quit
    uv run sync_cache.py --no-tray    # run in the terminal (Ctrl+C to stop)
    uv run sync_cache.py --status     # progress and today's request count
Log: data/sync.log. Only one instance runs at a time (data/sync.lock).
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
import threading
import time
from datetime import date, datetime

from backtest_flips import (ARCHIVE_DIR, ARCHIVE_MIN_PRICE, DATA_DIR, DAY, FIVE_MIN, LOCAL_TZ, PUBLISH_LAG_S, Api,
                            compact_bucket, find_earliest_bucket, fmt_ts, read_day, utc_day, write_day)

FULL_ARCHIVE_DIR = DATA_DIR / "5m-all"
STATE_PATH = DATA_DIR / "sync_state.json"
STATUS_PATH = DATA_DIR / "sync_status.json"
LOCK_PATH = DATA_DIR / "sync.lock"
LOG_PATH = DATA_DIR / "sync.log"

REQUEST_INTERVAL_S = 2.0  # 10x gentler than a manual backtest download
DAILY_REQUEST_LIMIT = 10_000  # per local calendar day
DAILY_RESERVE = 400  # kept for tier 1: a day of new buckets (288) + retention/mapping checks
FLUSH_EVERY = 48  # buckets held in memory before writing them to disk
RETRY_AFTER_ERROR_S = 300
RETRY_EMPTY_S = (3600, DAY)  # when to re-ask for a bucket the API answered empty: if < 1h old, otherwise
RETENTION_CHECK_S = DAY  # how often to re-find the oldest bucket the API serves
MAPPING_REFRESH_S = DAY
UI_REFRESH_S = 15  # how often the tray's tooltip and menu are refreshed
PRICE_REFRESH_S = 3600  # how often the status re-reads item prices to describe the filled range

log = logging.getLogger("sync_cache")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def today() -> str:
    return datetime.now(LOCAL_TZ).date().isoformat()


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"requests": {}}


def save_json(path, data, indent: int | None = None) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=indent))
    tmp.replace(path)


def acquire_lock():
    """Hold an OS lock on data/sync.lock for the life of the process; None if another instance holds it."""
    DATA_DIR.mkdir(exist_ok=True)
    handle = open(LOCK_PATH, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def lower_priority() -> None:
    """Background CPU, disk and memory priority (Windows), so the sync never competes with the game."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00100000)  # PROCESS_MODE_BACKGROUND_BEGIN
    else:
        import os
        os.nice(10)


def setup_logging() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")]
    if sys.stdout is not None:  # None under pythonw.exe
        sys.stdout.reconfigure(encoding="utf-8")
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=handlers)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def latest_end() -> int:
    """End of the newest bucket that's safe to cache (same rule as backtest_flips.main)."""
    return (int(time.time()) - PUBLISH_LAG_S - FIVE_MIN) // FIVE_MIN * FIVE_MIN + FIVE_MIN


def cached_buckets(directory) -> set[int]:
    out = set()
    for path in directory.glob("*.json.gz"):
        out.update(int(ts) for ts in read_day(date.fromisoformat(path.name.split(".")[0]), directory))
    return out


def fmt_ago(ts: int, now: float) -> str:
    age = now - ts
    if age < 3600:
        return "now"
    return f"{age / 3600:.0f} hours ago" if age < 2 * DAY else f"{age / DAY:.1f} days ago"


def filled_range(filled: set[int], end: int) -> tuple[int, int] | None:
    """The unbroken run of buckets ending at the newest one in `filled` (before `end`): [start, stop)."""
    newest = next((ts for ts in range(end - FIVE_MIN, end - 365 * DAY, -FIVE_MIN) if ts in filled), None)
    if newest is None:
        return None
    start = newest
    while start - FIVE_MIN in filled:
        start -= FIVE_MIN
    return start, newest + FIVE_MIN


def fmt_price(gp: float) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if gp >= size:
            return f"{gp / size:.3g}{unit}"
    return f"{gp:.0f}"


def latest_prices(end: int) -> dict[str, int]:
    """{item id: latest traded price} for items that traded in the last 24h, from the full archive (the
    strategy archive until the full one has a day)."""
    price: dict[str, int] = {}
    for directory in (FULL_ARCHIVE_DIR, ARCHIVE_DIR):
        buckets = {int(ts): b for day in sorted({utc_day(end - DAY), utc_day(end - 1)})
                   for ts, b in read_day(day, directory).items() if int(ts) >= end - DAY}
        for ts in sorted(buckets):
            for item_id, (high, _, low, _) in buckets[ts].items():
                price[item_id] = max(high or 0, low or 0)
        if price:
            break
    return price


class Syncer:
    def __init__(self, daily_limit: int, interval_s: float) -> None:
        self.api = Api(interval_s)
        self.daily_limit = daily_limit
        self.state = load_state()
        self.cached = {ARCHIVE_DIR: cached_buckets(ARCHIVE_DIR), FULL_ARCHIVE_DIR: cached_buckets(FULL_ARCHIVE_DIR)}
        self.pending: dict[tuple, dict[str, dict]] = {}  # (directory, day) -> buckets not written yet
        self.retry_at: dict[int, float] = {}  # bucket the API answered empty -> when to ask again
        self.counted = 0  # self.api.requests already added to the state's daily count
        self.state_saved = self.progress_logged = 0.0
        self.floor = 0  # oldest bucket the API serves
        self.floor_checked = self.mapping_checked = self.prices_read = 0.0
        self.prices: dict[str, int] = {}
        # Control and display, shared with the tray thread
        self.paused = False
        self.stopping = threading.Event()
        self.wake = threading.Event()  # interrupts a sleep (pause, resume, quit)
        self.activity = "Starting"
        self.filled = ""  # the status message: the price range, items and dates that are completely filled
        self.expanding = ""  # progress towards adding the items outside that range, if any
        self.on_update = lambda: None  # called whenever activity or the filled ranges may have changed

    def pause(self) -> None:
        self.paused = True
        self.wake.set()

    def resume(self) -> None:
        self.paused = False
        self.wake.set()

    def stop(self) -> None:
        self.stopping.set()
        self.wake.set()

    # --- budget ---

    def used_today(self) -> int:
        return self.state["requests"].get(today(), 0)

    def record_usage(self, save: bool = False) -> None:
        counts, day = self.state["requests"], today()
        counts[day] = counts.get(day, 0) + self.api.requests - self.counted
        self.counted = self.api.requests
        for old in sorted(counts)[:-14]:
            del counts[old]
        if save or time.time() - self.state_saved > 60:
            save_json(STATE_PATH, self.state, indent=1)
            self.state_saved = time.time()

    # --- tier 1 & 2: 5m buckets ---

    def next_bucket(self, end: int, directory) -> int | None:
        """Newest (strategy archive) or oldest (full archive) servable bucket missing from `directory`."""
        cached, newest_first, now = self.cached[directory], directory == ARCHIVE_DIR, time.time()
        span = range(end - FIVE_MIN, self.floor - 1, -FIVE_MIN) if newest_first else range(self.floor, end, FIVE_MIN)
        return next((ts for ts in span if ts not in cached and self.retry_at.get(ts, 0) <= now), None)

    def fetch_bucket(self, ts: int) -> None:
        data = self.api.get("5m", {"timestamp": ts})["data"]
        if not data:
            # Not published yet, past the retention edge (it moves forward between checks), or an API outage
            recent = ts > time.time() - RETRY_EMPTY_S[0]
            self.retry_at[ts] = time.time() + RETRY_EMPTY_S[0 if recent else 1]
            if ts < self.floor + DAY:
                self.floor = ts + FIVE_MIN
            return
        full = {item_id: [v.get("avgHighPrice"), v.get("highPriceVolume") or 0, v.get("avgLowPrice"),
                          v.get("lowPriceVolume") or 0] for item_id, v in data.items()}
        for directory, buckets in ((ARCHIVE_DIR, compact_bucket(data)), (FULL_ARCHIVE_DIR, full)):
            if ts not in self.cached[directory]:
                self.cached[directory].add(ts)
                self.pending.setdefault((directory, utc_day(ts)), {})[str(ts)] = buckets
        if sum(len(b) for b in self.pending.values()) >= FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        for (directory, day), buckets in self.pending.items():
            write_day(day, buckets, directory)
        self.pending.clear()

    # --- main loop ---

    def step(self) -> float:
        """Do one request's worth of work; return how long to sleep before the next step (0 = go on)."""
        now, end = time.time(), latest_end()
        next_publish = end + FIVE_MIN + PUBLISH_LAG_S - now + 5
        used = self.used_today()
        if used >= self.daily_limit:
            self.activity = "Daily request limit reached; waiting for midnight"
            self.flush()
            return max(next_publish, 60)

        if now - self.floor_checked > RETENTION_CHECK_S:
            self.activity = "Checking how far back the API serves"
            self.on_update()
            self.floor, self.floor_checked = find_earliest_bucket(self.api, end), now
            log.info(f"API serves 5m buckets from {fmt_ts(self.floor)}")
            return 0
        if now - self.mapping_checked > MAPPING_REFRESH_S:
            self.activity = "Refreshing the item mapping"
            save_json(DATA_DIR / "mapping.json", self.api.get("mapping"))
            self.mapping_checked = now
            return 0

        ts = self.next_bucket(end, ARCHIVE_DIR)
        if ts is not None:
            if end - ts > 2 * FIVE_MIN + PUBLISH_LAG_S:
                self.activity = f"Downloading 1M+ gp items: {fmt_ago(ts, time.time())}"
                self.log_progress(f"backfilling strategy archive at {fmt_ts(ts)}")
            self.fetch_bucket(ts)
            if self.next_bucket(end, ARCHIVE_DIR) is None:
                self.flush()  # caught up: make the newest bucket visible to backtests now
            return 0

        if used >= self.daily_limit - DAILY_RESERVE:
            self.activity = "Waiting for tomorrow's request budget to download more"
            self.flush()
            return next_publish
        ts = self.next_bucket(end, FULL_ARCHIVE_DIR)
        if ts is not None:
            self.activity = f"Downloading all items: {fmt_ago(ts, time.time())}"
            self.log_progress(f"filling full archive at {fmt_ts(ts)}")
            self.fetch_bucket(ts)
            return 0

        self.activity = "Up to date"
        self.flush()
        return next_publish

    def log_progress(self, what: str) -> None:
        """Log at most one line every 10 minutes during long backfills."""
        if time.time() - self.progress_logged > 600:
            log.info(f"{what} ({self.used_today():,} requests today)")
            self.progress_logged = time.time()

    def write_status(self) -> None:
        """Describe the one range of prices, volumes and dates that's completely filled.

        Items >= ARCHIVE_MIN_PRICE are in both archives, the rest only in the full one, so the range is either
        those items or, once the full archive reaches back as far, every item. Volume never limits it.
        """
        end, now = latest_end(), time.time()
        if now - self.prices_read > PRICE_REFRESH_S:
            self.prices, self.prices_read = latest_prices(end), now
        empty = set(self.retry_at)  # buckets the API answered empty have nothing to fill
        full = self.cached[FULL_ARCHIVE_DIR] | empty
        strategy_span, full_span = filled_range(self.cached[ARCHIVE_DIR] | full, end), filled_range(full, end)
        everything = bool(full_span and strategy_span and full_span[0] <= strategy_span[0])
        span = full_span if everything else strategy_span
        min_price = 0 if everything else ARCHIVE_MIN_PRICE
        prices = [p for p in self.prices.values() if p >= min_price]
        if span and prices:
            low = min(prices) if everything else min_price
            self.filled = (f"{fmt_price(low)}-{fmt_price(max(prices))} gp, any volume, {len(prices):,} items, "
                           f"{fmt_ago(span[0], now)} - {fmt_ago(span[1], now)}")
        self.expanding = ""
        if not everything and self.floor:
            servable = range(self.floor, end, FIVE_MIN)
            done = sum(ts in full for ts in servable) / len(servable)
            self.expanding = (f"Adding the {len(self.prices) - len(prices):,} items under "
                              f"{fmt_price(ARCHIVE_MIN_PRICE)} gp: {done:.0%} downloaded")
        save_json(STATUS_PATH, indent=2, data={
            "updated": fmt_ts(int(now)),
            "filled": self.filled,
            "expanding": self.expanding,
            "activity": self.activity,
            "requests_today": self.used_today(),
            "daily_limit": self.daily_limit,
            "api_serves_from": fmt_ts(self.floor) if self.floor else None,
        })

    def run(self) -> None:
        log.info(f"started: {self.daily_limit:,} requests/day, one per {self.api.interval_s:g}s; "
                 f"{len(self.cached[ARCHIVE_DIR]):,} strategy and {len(self.cached[FULL_ARCHIVE_DIR]):,} "
                 f"full buckets cached")
        last_status = last_ui = 0.0
        try:
            while not self.stopping.is_set():
                if self.paused:
                    self.activity = "Paused"
                    wait = None  # until resumed or stopped
                else:
                    try:
                        wait = self.step()
                    except Exception:
                        log.exception(f"sync failed; retrying in {RETRY_AFTER_ERROR_S}s")
                        self.activity = f"Error (see log); retrying in {RETRY_AFTER_ERROR_S // 60} min"
                        wait = RETRY_AFTER_ERROR_S
                if wait != 0:
                    self.flush()
                self.record_usage(save=wait != 0)
                if wait != 0 or time.time() - last_status > 60:
                    self.write_status()
                    last_status = time.time()
                if wait != 0 or time.time() - last_ui > UI_REFRESH_S:
                    self.on_update()
                    last_ui = time.time()
                if wait != 0:
                    self.wake.wait(wait)
                    self.wake.clear()
        finally:
            self.flush()
            self.record_usage(save=True)
            log.info("stopped")


def print_status() -> None:
    try:
        status = json.loads(STATUS_PATH.read_text())
    except FileNotFoundError:
        raise SystemExit("No status yet; sync_cache.py hasn't run.")
    if "expanding" not in status:
        raise SystemExit("The status file is from an older version; try again in a minute.")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Completely filled (as of {status['updated']}; items that traded in the last 24h):")
    print(f"  {status['filled'] or '(working it out...)'}")
    if status["expanding"]:
        print(f"Next: {status['expanding']}")
    print(f"Now: {status['activity']}")
    print(f"Requests today: {status['requests_today']:,} / {status['daily_limit']:,}")
    print(f"API serves 5m buckets from {status['api_serves_from']}")
    used = load_state()["requests"]
    print("Requests per day:", ", ".join(f"{d}: {n:,}" for d, n in sorted(used.items())[-7:]))


def already_running(quiet: bool) -> None:
    log.info("another instance is already running; exiting")
    if quiet:
        return
    if sys.stdout is None and sys.platform == "win32":  # windowless (Start menu shortcut): tell the user anyway
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, "RS GE Sync is already running; look for its icon in the system "
                                         "tray (it may be under the ^ overflow).", "RS GE Sync", 0x40)
    else:
        print("sync_cache.py is already running.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--status", action="store_true", help="print progress from the running sync and exit")
    parser.add_argument("--no-tray", action="store_true", help="run in the terminal without a tray icon")
    parser.add_argument("--background", action="store_true",
                        help="started at logon: exit quietly (code 0) if already running")
    parser.add_argument("--daily-limit", type=int, default=DAILY_REQUEST_LIMIT, help="max API requests per day")
    parser.add_argument("--interval", type=float, default=REQUEST_INTERVAL_S, help="min seconds between requests")
    args = parser.parse_args()
    if args.status:
        print_status()
        return
    setup_logging()
    lock = acquire_lock()
    if lock is None:
        already_running(args.background)
        return
    lower_priority()
    syncer = Syncer(args.daily_limit, args.interval)
    try:
        if args.no_tray:
            syncer.run()
        else:
            from sync_tray import run_tray
            run_tray(syncer, LOG_PATH)
    except KeyboardInterrupt:
        pass
    finally:
        lock.close()


if __name__ == "__main__":
    main()
