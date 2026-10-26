"""System tray icon for sync_cache.py: what's completely filled, pause/resume, open the log, quit.

The tray owns the main thread (pystray needs it); the sync loop runs in a worker thread and is controlled only
through Syncer.pause/resume/stop, so a quit still flushes buffered buckets to disk.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pystray
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from sync_cache import Syncer

APP_NAME = "RS GE Sync"
GOLD, GOLD_DARK, PAUSED_GREY, PAUSED_DARK = "#e8b923", "#7a5a00", "#a0a0a0", "#505050"
STOP_TIMEOUT_S = 60  # longest a quit waits for an in-flight request before giving up on the flush


def icon_image(paused: bool = False, size: int = 256) -> Image.Image:
    """A gold coin marked GE (grey while paused)."""
    face, edge = (PAUSED_GREY, PAUSED_DARK) if paused else (GOLD, GOLD_DARK)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    border = max(2, size // 12)
    draw.ellipse((1, 1, size - 2, size - 2), fill=edge)
    draw.ellipse((border, border, size - 1 - border, size - 1 - border), fill=face)
    try:
        font = ImageFont.truetype("arialbd.ttf", int(size * 0.42))
    except OSError:
        font = ImageFont.load_default(int(size * 0.42))
    draw.text((size / 2, size / 2), "GE", font=font, fill=edge, anchor="mm")
    return img


def run_tray(syncer: Syncer, log_path: Path) -> None:
    images = {False: icon_image(False, 64), True: icon_image(True, 64)}

    def toggle_pause(icon, item) -> None:
        syncer.resume() if syncer.paused else syncer.pause()
        refresh()

    def quit_app(icon, item) -> None:
        icon.stop()

    def info(text):
        return pystray.MenuItem(text, None, enabled=False)

    def menu_items():
        yield info("Completely filled (items that traded in the last 24h):")
        yield info(f"    {syncer.filled or '(working it out...)'}")
        if syncer.expanding:
            yield info(f"Next: {syncer.expanding}")
        yield pystray.Menu.SEPARATOR
        yield info(f"Now: {syncer.activity}")
        yield info(f"Requests today: {syncer.used_today():,} / {syncer.daily_limit:,}")
        yield pystray.Menu.SEPARATOR
        yield pystray.MenuItem(lambda item: "Resume" if syncer.paused else "Pause", toggle_pause)
        yield pystray.MenuItem("Open log", lambda: os.startfile(log_path))
        yield pystray.MenuItem("Open data folder", lambda: os.startfile(log_path.parent))
        yield pystray.MenuItem("Quit", quit_app)

    icon = pystray.Icon("rs_ge_sync", images[False], APP_NAME, menu=pystray.Menu(menu_items))

    def refresh() -> None:
        icon.icon = images[syncer.paused]
        filled = syncer.filled or syncer.activity
        icon.title = f"{APP_NAME}{' (paused)' if syncer.paused else ''}\n{filled}"[:127]  # Windows tooltip limit
        icon.update_menu()

    syncer.on_update = refresh
    worker = threading.Thread(target=syncer.run, name="sync", daemon=True)

    def setup(icon) -> None:
        icon.visible = True
        worker.start()

    icon.run(setup)
    syncer.stop()
    worker.join(STOP_TIMEOUT_S)
